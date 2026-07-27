"""Local speech-to-text via faster-whisper (CTranslate2 backend).

Transcription runs entirely in-process and exits with the recorder, so there is
no resident service holding a model in memory between meetings. The GPU (CUDA)
backend is used when its libraries load successfully; otherwise the code falls
back to CPU transparently, so a machine without a working CUDA stack still
produces transcripts (just slower).

``faster-whisper`` is an *optional* dependency (the ``transcribe`` extra). It is
imported lazily so the core recorder keeps working when the extra isn't
installed; calling :func:`transcribe_file` without it raises a clear error.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Preferred CTranslate2 compute type per device. ``int8_float16`` keeps GPU
#: memory low (comfortably inside a 4 GB card); CPU uses plain ``int8``.
_DEVICE_COMPUTE: dict[str, str] = {"cuda": "int8_float16", "cpu": "int8"}

#: nvidia-* wheels that ship the CUDA cuBLAS/cuDNN DLLs CTranslate2 needs. On
#: Windows their ``bin`` dirs must be on the DLL search path before the GPU
#: backend can load.
_CUDA_DLL_PACKAGES = ("nvidia.cublas", "nvidia.cudnn")


@dataclass(frozen=True)
class TranscriptionResult:
    """Outcome of one transcription."""

    text: str
    language: str | None
    duration: float
    device: str  # the backend actually used ("cuda" or "cpu")


def _register_cuda_dll_dirs() -> None:
    """Make the nvidia wheel ``bin`` dirs discoverable for CUDA DLL loading.

    The ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` wheels install their DLLs
    under ``site-packages/nvidia/<pkg>/bin``. CTranslate2 does not find them
    there automatically on Windows, and ``os.add_dll_directory`` alone is not
    enough for how CTranslate2 resolves its CUDA dependencies -- the bin dirs
    must also be on ``PATH``. We do both. Best-effort and idempotent: any
    missing package or unsupported platform is simply skipped (CPU still works).
    """
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:  # non-Windows: nothing to do
        return
    for pkg in _CUDA_DLL_PACKAGES:
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError):
            continue
        if not spec or not spec.submodule_search_locations:
            continue
        bin_dir = Path(next(iter(spec.submodule_search_locations))) / "bin"
        if not bin_dir.is_dir():
            continue
        bin_str = str(bin_dir)
        try:
            add_dll_directory(bin_str)
        except OSError:
            log.debug("Could not register CUDA DLL dir %s", bin_dir, exc_info=True)
        # CTranslate2 resolves cuBLAS/cuDNN via PATH; prepend if not already present.
        path = os.environ.get("PATH", "")
        if bin_str not in path.split(os.pathsep):
            os.environ["PATH"] = bin_str + os.pathsep + path


def _inject_system_trust_store() -> None:
    """Route Python's SSL through the OS certificate store, if available.

    On networks that intercept TLS with a corporate root CA, ``huggingface_hub``
    (which uses certifi's bundle) cannot verify the model download. ``truststore``
    makes Python trust the Windows certificate store instead, where that CA
    already lives. Best-effort: absence of ``truststore`` is silently ignored.
    """
    try:
        import truststore
    except ModuleNotFoundError:
        return
    try:
        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001 - never let trust-store setup break a run
        log.debug("truststore injection failed", exc_info=True)


def _candidate_devices(device: str) -> list[str]:
    """Resolve a device preference into an ordered list to attempt.

    ``"auto"`` tries CUDA first then CPU; an explicit ``"cuda"``/``"cpu"`` is
    used as-is (no silent fallback, so an explicit GPU request that fails is a
    visible error rather than a slow surprise).
    """
    if device == "auto":
        return ["cuda", "cpu"]
    return [device]


def _load_whisper_model(model_size: str, device: str, compute_type: str) -> Any:
    """Construct a faster-whisper ``WhisperModel`` (isolated for testability).

    Kept as a thin seam so unit tests can substitute a fake without needing the
    optional dependency or a downloaded model.
    """
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:  # optional dependency missing
        raise RuntimeError(
            "faster-whisper is not installed. Install the transcription extra: "
            'pip install "meeting-recorder[transcribe]"'
        ) from exc
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_file(
    wav_path: Path,
    *,
    model: str = "base.en",
    device: str = "auto",
    language: str = "en",
) -> TranscriptionResult:
    """Transcribe ``wav_path`` and return the recognized text plus metadata.

    Attempts each candidate device in order (see :func:`_candidate_devices`),
    falling back to the next when a backend cannot be initialized. Raises
    ``FileNotFoundError`` if the audio is missing, or ``RuntimeError`` if no
    backend could be loaded at all.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    _inject_system_trust_store()
    _register_cuda_dll_dirs()

    last_error: Exception | None = None
    for dev in _candidate_devices(device):
        compute_type = _DEVICE_COMPUTE.get(dev, "int8")
        try:
            whisper = _load_whisper_model(model, dev, compute_type)
        except Exception as exc:  # noqa: BLE001 - report and try the next device
            last_error = exc
            log.warning("Could not initialize the %s backend (%s); trying next.", dev, exc)
            continue

        log.info("Transcribing %s with %s on %s...", wav_path.name, model, dev)
        segments, info = whisper.transcribe(str(wav_path), language=language)
        text = "".join(segment.text for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            device=dev,
        )

    raise RuntimeError(f"No transcription backend could be loaded (last error: {last_error})")


def write_transcript(text: str, dest_md: Path) -> Path:
    """Write ``text`` to ``dest_md`` atomically and return the final path.

    Writes to a hidden ``.part`` sibling first, then renames, so a folder
    watcher (or the catch-up automation) never observes a half-written file.
    The body is written verbatim (no frontmatter) to match what the previous
    transcriber emitted.
    """
    dest_md.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_md.with_name(f".{dest_md.name}.part")
    tmp.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
    os.replace(tmp, dest_md)
    return dest_md
