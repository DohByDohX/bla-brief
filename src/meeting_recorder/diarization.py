"""Speaker diarization: turns a plain transcript into "Speaker N: ..." lines.

Pipeline: VAD (faster-whisper's bundled Silero) -> ECAPA speaker embeddings
(SpeechBrain, CPU, ungated -- no license form) -> agglomerative clustering ->
merge with Whisper's word timestamps. Runs unconditionally as part of every
recording (see ``cli._transcribe_recording``); there is no plain, undiarized
transcription path anymore.

OFFLINE-FIRST: normal runs make no network calls (same guarantee as
``transcription.py``). The one-time model fetch is :func:`download_embedder`,
called from ``transcription.download_model`` (the ``--download-model`` flag).

Speaker labels are just cluster IDs ("Speaker 0", "Speaker 1", ...), not real
names, and the clustering threshold is calibrated on a small number of
recordings -- treat the speaker count as a best-effort estimate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from meeting_recorder.config import (
    DIARIZE_CACHE_DIR,
    DIARIZE_CLUSTER_THRESHOLD,
    DIARIZE_MIN_SEGMENT_S,
    DIARIZE_MODEL,
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

SAMPLE_RATE = 16000  # faster-whisper's internal decode/VAD rate


@dataclass
class _SpeechSegment:
    start: float
    end: float
    speaker: int | None = None


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
    from speechbrain.utils.fetching import LocalStrategy

    _inject_system_trust_store()
    from speechbrain.inference.speaker import EncoderClassifier

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


def _assign_word_speakers(
    words: list[dict[str, Any]], segments: list[_SpeechSegment]
) -> list[dict[str, Any]]:
    for word in words:
        mid = (word["start"] + word["end"]) / 2
        speaker = next((s.speaker for s in segments if s.start <= mid <= s.end), None)
        word["speaker"] = speaker
    return words


def _render_transcript(words: list[dict[str, Any]]) -> str:
    lines = []
    current_speaker: object = object()
    buffer: list[str] = []
    for word in words:
        if word["speaker"] != current_speaker:
            if buffer:
                lines.append(f"Speaker {current_speaker}: {''.join(buffer).strip()}")
            current_speaker = word["speaker"]
            buffer = []
        buffer.append(word["text"])
    if buffer:
        lines.append(f"Speaker {current_speaker}: {''.join(buffer).strip()}")
    return "\n".join(lines)


def transcribe_with_speakers(
    wav_path: Path,
    *,
    model: str = "small.en",
    device: str = "auto",
    language: str = "en",
) -> TranscriptionResult:
    """Transcribe ``wav_path`` with "Speaker N: ..." labels and return it.

    Runs fully offline (no network); see module docstring. Raises
    ``FileNotFoundError`` if the audio is missing, or ``RuntimeError`` if no
    Whisper backend could be loaded.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    _enable_offline()  # no network during a normal recording
    _register_cuda_dll_dirs()

    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(wav_path), sampling_rate=SAMPLE_RATE)
    segments = _run_vad(audio)
    embedder = _load_embedder()
    embeddings = _embed_segments(embedder, audio, segments)
    labels = _cluster(embeddings)
    for seg, label in zip(segments, labels, strict=True):
        seg.speaker = int(label)

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
        words = _assign_word_speakers(words, segments)
        return TranscriptionResult(
            text=_render_transcript(words),
            language=getattr(info, "language", None),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            device=dev,
        )

    raise RuntimeError(f"No transcription backend could be loaded (last error: {last_error})")
