"""Speaker diarization: turns a plain transcript into "Speaker N: ..." lines.

Pipeline: VAD (faster-whisper's bundled Silero) -> ECAPA speaker embeddings
(SpeechBrain, CPU, ungated -- no license form) -> two-stage agglomerative
clustering with a per-meeting threshold (best silhouette) -> re-segmentation of
long segments that hold two speakers -> merge with Whisper's word timestamps.
Runs unconditionally as part of every recording (see
``cli._transcribe_recording``); there is no plain, undiarized transcription
path anymore. Always clusters the single mixed-file audio stream -- there is
no channel-aware mode.

OFFLINE-FIRST: normal runs make no network calls (same guarantee as
``transcription.py``). The one-time model fetch is :func:`download_embedder`,
called from ``transcription.download_model`` (the ``--download-model`` flag).

Speaker labels are just cluster IDs ("Speaker 0", "Speaker 1", ...), not real
names. The settings were tuned on a small set of hand-labelled meetings (see
config.py) -- treat the speaker count as a best-effort estimate.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

import numpy as np

from meeting_recorder.config import (
    DIARIZE_CACHE_DIR,
    DIARIZE_CLUSTER_THRESHOLD,
    DIARIZE_MIN_SEGMENT_S,
    DIARIZE_MODEL,
    DIARIZE_RESEG_HOP_S,
    DIARIZE_RESEG_MARGIN,
    DIARIZE_RESEG_MIN_S,
    DIARIZE_RESEG_RUN_S,
    DIARIZE_RESEG_WIN_S,
    DIARIZE_THRESHOLD_SWEEP,
    DIARIZE_TWO_STAGE_MIN_S,
    DIARIZE_VAD_MIN_SILENCE_MS,
)
from meeting_recorder.transcription import (
    _DEVICE_COMPUTE,
    TranscriptionResult,
    _candidate_devices,
    _enable_offline,
    _inject_system_trust_store,
    _load_whisper_model,
    _register_cuda_dll_dirs,
)

log = logging.getLogger(__name__)

# speechbrain logs its model-fetch/quirks chatter at INFO; it's noise for this
# app's users, not actionable, so keep it at WARNING and above.
logging.getLogger("speechbrain").setLevel(logging.WARNING)

SAMPLE_RATE = 16000  # faster-whisper's internal decode/VAD rate


@contextmanager
def _quiet_model_load() -> Iterator[None]:
    """Silence warnings that are noise here: an unrelated requests/urllib3
    version mismatch, and torch's upstream weights_only=False advisory.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*doesn't match a supported version.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=r".*weights_only.*")
        yield


@dataclass
class _SpeechSegment:
    start: float
    end: float
    label: str | None = None


def _run_vad(audio: np.ndarray) -> list[_SpeechSegment]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_silence_duration_ms=DIARIZE_VAD_MIN_SILENCE_MS)
    timestamps = get_speech_timestamps(audio, opts)
    return [
        _SpeechSegment(start=ts["start"] / SAMPLE_RATE, end=ts["end"] / SAMPLE_RATE)
        for ts in timestamps
    ]


def _load_embedder() -> Any:
    """Construct the SpeechBrain ECAPA embedder (isolated for testability).

    Runs on CPU on purpose: Whisper already owns the GPU, and embedding a
    meeting's worth of audio on CPU costs no VRAM.
    """
    with _quiet_model_load():
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        return EncoderClassifier.from_hparams(
            source=DIARIZE_MODEL,
            savedir=str(DIARIZE_CACHE_DIR),
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,  # avoid symlinks (need elevation on Windows)
        )


def download_embedder() -> None:
    """Fetch the diarization embedder into the local cache (online, one-time)."""
    _inject_system_trust_store()
    with _quiet_model_load():
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        EncoderClassifier.from_hparams(
            source=DIARIZE_MODEL,
            savedir=str(DIARIZE_CACHE_DIR),
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,  # avoid symlinks (need elevation on Windows)
        )


def _embed_segments(embedder: Any, audio: np.ndarray, segments: list[_SpeechSegment]) -> np.ndarray:
    import torch

    vectors = []
    for seg in segments:
        chunk = audio[int(seg.start * SAMPLE_RATE) : int(seg.end * SAMPLE_RATE)]
        tensor = torch.from_numpy(chunk).float().unsqueeze(0)
        with torch.no_grad():
            emb = embedder.encode_batch(tensor).squeeze().numpy()
        vectors.append(emb)
    return np.stack(vectors)


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize so Euclidean distance behaves like cosine distance."""
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # type: ignore[no-any-return]


def _ahc(normed: np.ndarray, threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="euclidean", linkage="average"
    )
    return clustering.fit_predict(normed)  # type: ignore[no-any-return]


def _cluster(embeddings: np.ndarray) -> np.ndarray:
    """Cluster with the per-meeting threshold that gives the best cosine silhouette.

    Tries every threshold in ``DIARIZE_THRESHOLD_SWEEP``; only clusterings with
    2..n-1 clusters can be scored. Falls back to ``DIARIZE_CLUSTER_THRESHOLD``
    when none qualifies (e.g. very few segments, or a single speaker).
    """
    from sklearn.metrics import silhouette_score

    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)
    normed = _normalize(embeddings)
    lo, hi, step = DIARIZE_THRESHOLD_SWEEP
    best_score, best_labels = -2.0, None
    for threshold in np.arange(lo, hi + 1e-3, step):
        labels = _ahc(normed, float(threshold))
        if 2 <= len(set(labels)) < len(labels):
            score = silhouette_score(normed, labels, metric="cosine")
            if score > best_score:
                best_score, best_labels = score, labels
    return best_labels if best_labels is not None else _ahc(normed, DIARIZE_CLUSTER_THRESHOLD)


def _centroids(
    labels: np.ndarray, normed: np.ndarray, durs: np.ndarray
) -> tuple[list[int], np.ndarray]:
    """Cluster ids (labels >= 0) and their duration-weighted, L2-normalized centroids."""
    ids = sorted({int(c) for c in labels if c >= 0})
    cent = np.stack([(normed[labels == c] * durs[labels == c, None]).sum(0) for c in ids])
    return ids, _normalize(cent)


def _two_stage_labels(embeddings: np.ndarray, durs: np.ndarray) -> np.ndarray:
    """Cluster only segments >= ``DIARIZE_TWO_STAGE_MIN_S``; the rest join a cluster.

    Short segments have noisy embeddings: clustered directly they bridge
    speakers or form phantom one-segment clusters, so they take the label of
    the nearest centroid of the long ones instead. With fewer than two long
    segments (or no short ones) everything is clustered together.
    """
    long = durs >= DIARIZE_TWO_STAGE_MIN_S
    if long.sum() < 2 or long.all():
        return np.asarray(_cluster(embeddings))
    normed = _normalize(embeddings)
    labels = np.full(len(embeddings), -1)
    labels[long] = np.asarray(_cluster(embeddings[long]))
    ids, cent = _centroids(labels[long], normed[long], durs[long])
    labels[~long] = np.array(ids)[np.argmax(normed[~long] @ cent.T, axis=1)]
    return labels


def _window_spans(start: float, end: float) -> list[tuple[float, float]]:
    """Re-segmentation windows over [start, end]; the last one ends exactly at ``end``."""
    w, h = DIARIZE_RESEG_WIN_S, DIARIZE_RESEG_HOP_S
    if end - start <= w:
        return [(start, end)]
    starts = list(np.arange(start, end - w + 1e-6, h))
    if starts[-1] + w < end - 1e-6:
        starts.append(end - w)
    return [(float(t), float(t) + w) for t in starts]


_WINDOW_BATCH = 32


def _embed_windows(
    embedder: Any, audio: np.ndarray, spans: list[tuple[float, float]]
) -> np.ndarray:
    """ECAPA embeddings of fixed-length windows (zero-padded past the audio end), batched."""
    import torch

    n = int(round(DIARIZE_RESEG_WIN_S * SAMPLE_RATE))
    out = []
    for k in range(0, len(spans), _WINDOW_BATCH):
        chunks = []
        for start, _ in spans[k : k + _WINDOW_BATCH]:
            chunk = audio[int(start * SAMPLE_RATE) : int(start * SAMPLE_RATE) + n]
            chunks.append(np.pad(chunk, (0, n - len(chunk))) if len(chunk) < n else chunk)
        with torch.no_grad():
            emb = embedder.encode_batch(torch.from_numpy(np.stack(chunks)).float())
        out.append(emb.squeeze(1).numpy())
    return np.concatenate(out)


def _split_by_votes(
    seg: _SpeechSegment,
    spans: list[tuple[float, float]],
    votes: np.ndarray,
    own: int,
    ids: list[int],
) -> list[_SpeechSegment]:
    """Split ``seg`` where windows vote for another cluster for >= ``DIARIZE_RESEG_RUN_S``.

    ``votes`` holds one index into ``ids`` per window. A run of votes for
    another cluster shorter than the minimum reverts to ``own``. Piece
    boundaries sit halfway between neighbouring window centres, so the pieces
    tile the original segment exactly.
    """
    centres = [(a + b) / 2 for a, b in spans]
    edges = [seg.start, *((x + y) / 2 for x, y in zip(centres, centres[1:], strict=False))]
    edges.append(seg.end)
    final = votes.copy()
    k = 0
    while k < len(votes):
        j = k
        while j + 1 < len(votes) and votes[j + 1] == votes[k]:
            j += 1
        if votes[k] != own and edges[j + 1] - edges[k] < DIARIZE_RESEG_RUN_S - 1e-9:
            final[k : j + 1] = own
        k = j + 1
    if (final == own).all():
        return [seg]
    pieces = []
    k = 0
    while k < len(final):
        j = k
        while j + 1 < len(final) and final[j + 1] == final[k]:
            j += 1
        pieces.append(_SpeechSegment(edges[k], edges[j + 1], f"Speaker {ids[final[k]]}"))
        k = j + 1
    return pieces


def _resegment(
    embedder: Any,
    audio: np.ndarray,
    segments: list[_SpeechSegment],
    labels: np.ndarray,
    normed: np.ndarray,
    durs: np.ndarray,
) -> list[_SpeechSegment]:
    """Split long segments that hold two speakers, using the existing clusters only.

    Each segment >= ``DIARIZE_RESEG_MIN_S`` is re-embedded in sliding windows.
    A window votes for another cluster when that cluster's centroid is the
    nearest and beats the segment's own cluster by >= ``DIARIZE_RESEG_MARGIN``
    (cosine); :func:`_split_by_votes` turns the votes into pieces. No new
    clusters can appear.
    """
    long_idx = [i for i in range(len(segments)) if durs[i] >= DIARIZE_RESEG_MIN_S]
    spans: list[tuple[float, float]] = []
    owner: list[int] = []
    for i in long_idx:
        for span in _window_spans(segments[i].start, segments[i].end):
            spans.append(span)
            owner.append(i)
    if not spans:
        return segments
    ids, cent = _centroids(labels, normed, durs)
    sims = _normalize(_embed_windows(embedder, audio, spans)) @ cent.T
    owners = np.array(owner)
    long_set = set(long_idx)
    out: list[_SpeechSegment] = []
    for i, seg in enumerate(segments):
        if i not in long_set:
            out.append(seg)
            continue
        rows = np.flatnonzero(owners == i)
        s = sims[rows]
        own = ids.index(int(labels[i]))
        best = np.argmax(s, axis=1)
        wins = (best != own) & (s[np.arange(len(rows)), best] - s[:, own] >= DIARIZE_RESEG_MARGIN)
        votes = np.where(wins, best, own)
        out.extend(_split_by_votes(seg, [spans[r] for r in rows], votes, own, ids))
    return out


def _diarize_single_track(audio: np.ndarray) -> list[_SpeechSegment]:
    """VAD + embed + cluster + re-segment one audio stream into labeled segments.

    Segments shorter than ``DIARIZE_MIN_SEGMENT_S`` are dropped before
    embedding -- ECAPA needs a minimum amount of audio for a reliable speaker
    embedding, so a too-short sliver is left unlabeled (renders as
    "Speaker None") rather than risk a garbage cluster assignment.
    """
    segments = [s for s in _run_vad(audio) if (s.end - s.start) >= DIARIZE_MIN_SEGMENT_S]
    if not segments:
        return segments
    embedder = _load_embedder()
    embeddings = _embed_segments(embedder, audio, segments)
    durs = np.array([s.end - s.start for s in segments])
    labels = _two_stage_labels(embeddings, durs)
    for seg, label in zip(segments, labels, strict=True):
        seg.label = f"Speaker {int(label)}"
    return _resegment(embedder, audio, segments, labels, _normalize(embeddings), durs)


# Words farther than this from any segment stay unlabeled (often Whisper hallucinations).
_WORD_SNAP_TOLERANCE_S = 0.5


def _assign_word_labels(
    words: list[dict[str, Any]], segments: list[_SpeechSegment]
) -> list[dict[str, Any]]:
    """Attach every label whose segment covers each word's midpoint (0, 1, or
    more -- more than one means overlapping speech, e.g. mic + system both
    active). A word with no such segment falls back to the segment it overlaps
    most, then to the nearest segment within ``_WORD_SNAP_TOLERANCE_S``;
    otherwise it stays unlabeled.
    """
    labeled = [s for s in segments if s.label]
    for word in words:
        start, end = word["start"], word["end"]
        mid = (start + end) / 2
        word["labels"] = [s.label for s in labeled if s.start <= mid <= s.end]
        if word["labels"] or not labeled:
            continue
        best = max(labeled, key=lambda s: min(end, s.end) - max(start, s.start))
        if min(end, best.end) - max(start, best.start) > 0:
            word["labels"] = [best.label]
            continue
        nearest = min(labeled, key=lambda s: max(s.start - mid, mid - s.end))
        if max(nearest.start - mid, mid - nearest.end) <= _WORD_SNAP_TOLERANCE_S:
            word["labels"] = [nearest.label]
    return words


def _render_transcript(words: list[dict[str, Any]]) -> str:
    """Group consecutive words by their active label set into lines.

    Overlapping speech (more than one active label) renders as one line per
    label, each carrying the same shared words -- there is only one
    transcribed word stream, so there is no way to split the text itself
    between the overlapping speakers.
    """
    lines: list[str] = []
    current_labels: tuple[str, ...] = ()
    buffer: list[str] = []

    def flush() -> None:
        text = "".join(buffer).strip()
        if not text:
            return
        if not current_labels:
            lines.append(f"Speaker None: {text}")
        else:
            lines.extend(f"{label}: {text}" for label in current_labels)

    for word in words:
        labels = tuple(word["labels"])
        if labels != current_labels:
            flush()
            current_labels = labels
            buffer = []
        buffer.append(word["text"])
    flush()
    return _dedup_consecutive_duplicates("\n".join(lines))


_DUPLICATE_SIMILARITY_THRESHOLD = 0.9
_TRAILING_PUNCTUATION = ".,!?;:"


def _normalize_for_comparison(text: str) -> str:
    """Strip trailing punctuation/case noise that Whisper applies inconsistently
    to the same utterance when it's split across overlapping segments.
    """
    return text.strip().rstrip(_TRAILING_PUNCTUATION).strip().lower()


def _is_near_duplicate(a: str, b: str) -> bool:
    """True if ``a`` and ``b`` are the same utterance modulo minor transcription
    noise (trailing punctuation, filler words picked up differently, etc).
    """
    norm_a, norm_b = _normalize_for_comparison(a), _normalize_for_comparison(b)
    if not norm_a or not norm_b:
        return norm_a == norm_b
    if norm_a == norm_b:
        return True
    return SequenceMatcher(None, norm_a, norm_b).ratio() >= _DUPLICATE_SIMILARITY_THRESHOLD


def _dedup_consecutive_duplicates(transcript: str) -> str:
    """Remove runs of consecutive lines that repeat the same (or near-identical)
    text under different speaker labels.

    Artifact pattern: Speaker A says X, then one or more other speakers echo the
    same X -- verbatim or with minor transcription noise -- (mis-segmentation
    from overlapping mic/system audio). The whole echoed run is dropped,
    keeping only A's original line, regardless of which speaker continues
    afterward -- real-world testing showed this echo is a mic/system
    bleed-through artifact even for short acknowledgments, not genuine
    simultaneous speech worth preserving.
    """
    lines = transcript.split("\n")
    if len(lines) < 2:
        return transcript

    parsed = [_parse_speaker_line(line) for line in lines]
    deduped: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        text_i = parsed[i][1]
        deduped.append(lines[i])

        j = i + 1
        while j < n and text_i.strip() and _is_near_duplicate(parsed[j][1], text_i):
            j += 1

        # lines[i+1:j] all echo text_i; drop the whole run unconditionally.
        if j > i + 1:
            i = j
        else:
            i += 1

    return "\n".join(deduped)


def _parse_speaker_line(line: str) -> tuple[str, str]:
    """Extract speaker label and text from a 'Speaker X: text' line."""
    if ": " not in line:
        return ("", line)
    speaker, _, text = line.partition(": ")
    return (speaker, text)


def transcribe_with_speakers(
    wav_path: Path,
    *,
    model: str = "small.en",
    device: str = "auto",
    language: str = "en",
) -> TranscriptionResult:
    """Transcribe ``wav_path`` with "Speaker N: ..." labels and return it.

    Clusters the audio directly (see :func:`_diarize_single_track`).

    Runs fully offline (no network); see module docstring. Raises
    ``FileNotFoundError`` if the audio is missing, or ``RuntimeError`` if no
    Whisper backend could be loaded.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    _enable_offline()  # no network during a normal recording
    _register_cuda_dll_dirs()

    from faster_whisper.audio import decode_audio

    audio = cast(np.ndarray, decode_audio(str(wav_path), sampling_rate=SAMPLE_RATE))
    segments = _diarize_single_track(audio)

    last_error: Exception | None = None
    for dev in _candidate_devices(device):
        compute_type = _DEVICE_COMPUTE.get(dev, "int8")
        try:
            whisper = _load_whisper_model(model, dev, compute_type)
        except Exception as exc:  # noqa: BLE001 - report and try the next device
            last_error = exc
            log.warning("Could not initialize the %s backend (%s); trying next.", dev, exc)
            continue

        log.info("Transcribing %s with %s on %s (diarized)...", wav_path.name, model, dev)
        whisper_segments, info = whisper.transcribe(
            str(wav_path), language=language, word_timestamps=True
        )
        words = [
            {"start": w.start, "end": w.end, "text": w.word}
            for seg in whisper_segments
            for w in seg.words or []
        ]
        words = _assign_word_labels(words, segments)
        return TranscriptionResult(
            text=_render_transcript(words),
            language=getattr(info, "language", None),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            device=dev,
        )

    raise RuntimeError(f"No transcription backend could be loaded (last error: {last_error})")
