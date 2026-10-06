"""Speaker diarization: turns a plain transcript into "Speaker N: ..." lines.

Pipeline: VAD (faster-whisper's bundled Silero) -> ECAPA speaker embeddings
(SpeechBrain, CPU, ungated -- no license form) -> agglomerative clustering ->
merge with Whisper's word timestamps. Runs unconditionally as part of every
recording (see ``cli._transcribe_recording``); there is no plain, undiarized
transcription path anymore.

Channel-aware mode (when the raw mic/system tracks are available): the mic
track is a known fact -- every segment on it is labeled directly as the local
speaker, no embedding/clustering needed -- and only the system/loopback
track is clustered, so it only ever has to tell remote participants apart
from each other. Falls back to single-track clustering on the mixed file
when the raw tracks aren't available.

OFFLINE-FIRST: normal runs make no network calls (same guarantee as
``transcription.py``). The one-time model fetch is :func:`download_embedder`,
called from ``transcription.download_model`` (the ``--download-model`` flag).

Speaker labels are just cluster IDs ("Speaker 0", "Speaker 1", ...) or the
fixed local-speaker label, not real names, and the clustering threshold is
calibrated on a small number of recordings -- treat the speaker count as a
best-effort estimate.
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
    DIARIZE_LOCAL_SPEAKER_LABEL,
    DIARIZE_MIN_SEGMENT_S,
    DIARIZE_MODEL,
)
from meeting_recorder.mixing import resolve_alignment_offset_s
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

    opts = VadOptions(min_silence_duration_ms=500)
    timestamps = get_speech_timestamps(audio, opts)
    segments = [
        _SpeechSegment(start=ts["start"] / SAMPLE_RATE, end=ts["end"] / SAMPLE_RATE)
        for ts in timestamps
    ]
    return [s for s in segments if (s.end - s.start) >= DIARIZE_MIN_SEGMENT_S]


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


def _cluster(embeddings: np.ndarray) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    # L2-normalize so Euclidean distance behaves like cosine distance.
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DIARIZE_CLUSTER_THRESHOLD,
        metric="euclidean",
        linkage="average",
    )
    return clustering.fit_predict(normed)  # type: ignore[no-any-return]


def _diarize_single_track(audio: np.ndarray) -> list[_SpeechSegment]:
    """VAD + embed + cluster one audio stream into labeled segments."""
    segments = _run_vad(audio)
    if not segments:
        return segments
    embedder = _load_embedder()
    embeddings = _embed_segments(embedder, audio, segments)
    labels = _cluster(embeddings)
    for seg, label in zip(segments, labels, strict=True):
        seg.label = f"Speaker {int(label)}"
    return segments


def _diarize_channel_aware(mic_path: Path, sys_path: Path) -> list[_SpeechSegment]:
    """Label the mic track directly (it's always the local speaker) and only
    cluster the system/loopback track, so clustering only has to tell remote
    participants apart from each other.

    The two raw tracks can start at slightly different wall-clock times (see
    ``mixing.create_mixed_file``); segments are shifted by the same
    duration-based offset used to build the mixed file, so they land on the
    mixed file's timeline (the one Whisper's word timestamps use).
    """
    from faster_whisper.audio import decode_audio

    # split_stereo defaults to False, so this always returns a plain ndarray,
    # never the stereo tuple variant in decode_audio's return type.
    mic_audio = cast(np.ndarray, decode_audio(str(mic_path), sampling_rate=SAMPLE_RATE))
    sys_audio = cast(np.ndarray, decode_audio(str(sys_path), sampling_rate=SAMPLE_RATE))
    offset_sec = resolve_alignment_offset_s(
        len(mic_audio) / SAMPLE_RATE, len(sys_audio) / SAMPLE_RATE, SAMPLE_RATE
    )
    mic_pad, sys_pad = max(0.0, -offset_sec), max(0.0, offset_sec)

    mic_segments = _run_vad(mic_audio)
    for seg in mic_segments:
        seg.start += mic_pad
        seg.end += mic_pad
        seg.label = DIARIZE_LOCAL_SPEAKER_LABEL

    sys_segments = _diarize_single_track(sys_audio)
    for seg in sys_segments:
        seg.start += sys_pad
        seg.end += sys_pad

    return mic_segments + sys_segments


def _assign_word_labels(
    words: list[dict[str, Any]], segments: list[_SpeechSegment]
) -> list[dict[str, Any]]:
    """Attach every label whose segment covers each word (0, 1, or more --
    more than one means overlapping speech, e.g. mic + system both active).
    """
    for word in words:
        mid = (word["start"] + word["end"]) / 2
        word["labels"] = [s.label for s in segments if s.start <= mid <= s.end and s.label]
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
    from overlapping mic/system audio), then A resumes with new text. The whole
    echoed run is dropped, keeping only A's original line.

    Genuine overlap (kept as-is): the repeated line is not followed by the
    original speaker resuming with new text -- e.g. two people say "yeah" at
    the same time and the conversation moves on from there instead.
    """
    lines = transcript.split("\n")
    if len(lines) < 2:
        return transcript

    parsed = [_parse_speaker_line(line) for line in lines]
    deduped: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        speaker_i, text_i = parsed[i]
        deduped.append(lines[i])

        j = i + 1
        while j < n and text_i.strip() and _is_near_duplicate(parsed[j][1], text_i):
            j += 1

        # lines[i+1:j] all echo text_i; drop the run only if the original
        # speaker resumes with genuinely different text right after it.
        if (
            j > i + 1
            and j < n
            and parsed[j][0] == speaker_i
            and not _is_near_duplicate(parsed[j][1], text_i)
        ):
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
    mic_path: Path | None = None,
    sys_path: Path | None = None,
) -> TranscriptionResult:
    """Transcribe ``wav_path`` with "Speaker N: ..." labels and return it.

    When ``mic_path``/``sys_path`` are given and exist, uses channel-aware
    diarization (see :func:`_diarize_channel_aware`); otherwise clusters the
    mixed file directly.

    Runs fully offline (no network); see module docstring. Raises
    ``FileNotFoundError`` if the audio is missing, or ``RuntimeError`` if no
    Whisper backend could be loaded.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    _enable_offline()  # no network during a normal recording
    _register_cuda_dll_dirs()

    if mic_path is not None and sys_path is not None and mic_path.exists() and sys_path.exists():
        segments = _diarize_channel_aware(mic_path, sys_path)
    else:
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
