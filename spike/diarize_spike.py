"""Throwaway spike: ungated speaker diarization (plain single-file).

Pipeline: VAD (faster-whisper's bundled Silero) -> ECAPA speaker embeddings
(SpeechBrain, CPU) -> agglomerative clustering -> merge with a Whisper
transcript by timestamp -> "Speaker N: ..." lines.

Not wired into the shipping recorder/CLI. Run directly:
    python spike/diarize_spike.py <path-to-wav>

Requires (spike-only, not in pyproject): speechbrain, torch, torchaudio,
scikit-learn. faster-whisper (and its bundled VAD) is already a project dep.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps
from sklearn.cluster import AgglomerativeClustering

SAMPLE_RATE = 16000  # faster-whisper's internal decode/VAD rate
# Agglomerative clustering distance threshold (euclidean, on L2-normalized
# embeddings). Lower = more speakers (over-splits); higher = fewer speakers
# (over-merges). This is the main tuning knob called out in the plan.
# Override at the command line for fast iteration: DIARIZE_THRESHOLD=1.1
# Spike finding on 2026-07-10_1201_daily-management-sync.wav: hand-counted
# ground truth is 5 speakers, matched at threshold=1.3 (1.2 under-split at 8).
CLUSTER_DISTANCE_THRESHOLD = float(os.environ.get("DIARIZE_THRESHOLD", "1.3"))
MIN_SEGMENT_S = 0.3  # drop VAD slivers too short to embed meaningfully


@dataclass
class SpeechSegment:
    start: float
    end: float
    speaker: int | None = None


def _run_vad(audio: np.ndarray) -> list[SpeechSegment]:
    opts = VadOptions(min_silence_duration_ms=500)
    timestamps = get_speech_timestamps(audio, opts)
    segments = [
        SpeechSegment(start=ts["start"] / SAMPLE_RATE, end=ts["end"] / SAMPLE_RATE)
        for ts in timestamps
    ]
    return [s for s in segments if (s.end - s.start) >= MIN_SEGMENT_S]


def _load_embedder():
    import truststore
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    truststore.inject_into_ssl()  # corporate TLS intercept; same fix as transcription.py

    # CPU on purpose: whisper already owns the GPU; embedding a short meeting
    # on CPU costs no VRAM and is fast enough for a spike.
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="spike/.models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,  # avoid symlinks (need elevation on Windows)
    )


def _embed_segments(embedder, audio: np.ndarray, segments: list[SpeechSegment]) -> np.ndarray:
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
    # Cosine-normalize so Euclidean distance in AgglomerativeClustering behaves
    # like cosine distance.
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
        metric="euclidean",
        linkage="average",
    )
    return clustering.fit_predict(normed)


def _transcribe(wav_path: Path) -> list[dict]:
    from faster_whisper import WhisperModel

    from meeting_recorder.transcription import _register_cuda_dll_dirs

    _register_cuda_dll_dirs()  # GPU only; no CPU fallback for this spike
    model = WhisperModel("base.en", device="cuda", compute_type="int8_float16")
    segments, _info = model.transcribe(str(wav_path), language="en", word_timestamps=True)
    words = []
    for seg in segments:
        for word in seg.words or []:
            words.append({"start": word.start, "end": word.end, "text": word.word})
    return words


def _assign_word_speakers(words: list[dict], segments: list[SpeechSegment]) -> list[dict]:
    for word in words:
        mid = (word["start"] + word["end"]) / 2
        speaker = next((s.speaker for s in segments if s.start <= mid <= s.end), None)
        word["speaker"] = speaker
    return words


def _render_transcript(words: list[dict]) -> str:
    lines = []
    current_speaker = object()
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


def _cached_vad_embeddings_and_words(
    wav_path: Path,
) -> tuple[list[SpeechSegment], np.ndarray, list[dict]]:
    """Cache VAD + embeddings + transcript so threshold tuning is instant."""
    cache_path = Path("spike") / f".{wav_path.stem}.cache.pkl"
    if cache_path.exists():
        print(f"Using cached VAD/embeddings/transcript from {cache_path}")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    audio = decode_audio(str(wav_path), sampling_rate=SAMPLE_RATE)

    print("Running VAD...")
    segments = _run_vad(audio)
    print(f"  {len(segments)} speech segments found.")

    print("Loading ECAPA embedder (CPU)...")
    embedder = _load_embedder()
    print("Embedding segments...")
    embeddings = _embed_segments(embedder, audio, segments)

    print("Transcribing (Whisper base.en, GPU)...")
    words = _transcribe(wav_path)

    with cache_path.open("wb") as f:
        pickle.dump((segments, embeddings, words), f)
    return segments, embeddings, words


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python spike/diarize_spike.py <path-to-wav>")
        sys.exit(1)
    wav_path = Path(sys.argv[1])

    t0 = time.monotonic()
    segments, embeddings, words = _cached_vad_embeddings_and_words(wav_path)

    print(f"Clustering speakers (threshold={CLUSTER_DISTANCE_THRESHOLD})...")
    labels = _cluster(embeddings)
    for seg, label in zip(segments, labels, strict=True):
        seg.speaker = int(label)
    print(f"  {len(set(labels))} speakers identified.")

    words = _assign_word_speakers(words, segments)

    transcript = _render_transcript(words)
    elapsed = time.monotonic() - t0

    out_path = Path("spike") / f"{wav_path.stem}.diarized.md"
    out_path.write_text(transcript + "\n", encoding="utf-8")

    print(f"\nDone in {elapsed:.1f}s. Transcript written to {out_path}\n")
    print(transcript)


if __name__ == "__main__":
    main()
