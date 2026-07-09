"""Audio device discovery for the dual-source recorder.

Everything derives from the WASAPI **default output device** so that when you
connect a Bluetooth/USB headset and Windows routes audio to it, the recorder
automatically captures that device's loopback, plays the keepalive on it, and
prefers its microphone.
"""

from __future__ import annotations

import logging
from typing import cast

import pyaudiowpatch as pyaudio

log = logging.getLogger(__name__)

DeviceInfo = dict


def _info(pa: pyaudio.PyAudio, index: int) -> DeviceInfo:
    """Typed wrapper around the (untyped) PyAudio device lookup."""
    return cast(DeviceInfo, pa.get_device_info_by_index(index))


def _find_wasapi_api_index(pa: pyaudio.PyAudio) -> int | None:
    """Return the host API index for WASAPI, or None if unavailable."""
    for i in range(pa.get_host_api_count()):
        api = pa.get_host_api_info_by_index(i)
        if "WASAPI" in api["name"]:
            return i
    return None


def find_devices(
    pa: pyaudio.PyAudio,
    mic_index: int | None = None,
    loopback_index: int | None = None,
) -> tuple[DeviceInfo | None, DeviceInfo | None, DeviceInfo | None]:
    """Find the mic, loopback, and output devices.

    Key design: all three devices are derived from the WASAPI default output
    device. When you connect a Bluetooth/USB headset and Windows routes audio
    to it, we automatically:
      - Capture system audio from THAT device's loopback
      - Play silence keepalive on THAT output (so loopback stays active)
      - Use THAT headset's mic (if it has one) instead of the built-in mic

    Mic priority (when ``mic_index`` is None):
      1. Explicit ``--mic`` override
      2. Mic paired with the default output device (same hardware name)
      3. WASAPI default input device
      4. Any WASAPI input device

    When ``loopback_index`` is given it overrides the auto-detected system
    (loopback) device.

    Returns:
        ``(mic_info, loopback_info, output_info)``; any element may be None.
    """
    wasapi_api_idx = _find_wasapi_api_index(pa)

    # Enumerate all devices once and reuse across the output / loopback / mic
    # searches below (avoids rescanning the full device list ~6 times).
    devices = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]

    output_info = _find_output(pa, devices, wasapi_api_idx)
    if loopback_index is not None:
        loopback_info: DeviceInfo | None = _info(pa, loopback_index)
    else:
        loopback_info = _find_loopback(devices, output_info)
    mic_info = _find_mic(pa, devices, output_info, wasapi_api_idx, mic_index)

    return mic_info, loopback_info, output_info


def _find_output(
    pa: pyaudio.PyAudio, devices: list[DeviceInfo], wasapi_api_idx: int | None
) -> DeviceInfo | None:
    """The anchor device — everything else derives from this."""
    if wasapi_api_idx is not None:
        api = pa.get_host_api_info_by_index(wasapi_api_idx)
        default_out = api.get("defaultOutputDevice", -1)
        if default_out >= 0:
            return _info(pa, default_out)
    for d in devices:
        if d["maxOutputChannels"] > 0 and not d.get("isLoopbackDevice", False):
            return d
    return None


def _find_loopback(devices: list[DeviceInfo], output_info: DeviceInfo | None) -> DeviceInfo | None:
    """Loopback device that matches the chosen output device."""
    if output_info:
        out_name = output_info["name"]
        # Prefer the exact "<output> [Loopback]" pairing PyAudioWPatch creates;
        # fall back to a prefix match so a shorter name can't match the wrong
        # (longer-named) device via a loose substring test.
        exact = f"{out_name} [Loopback]"
        for d in devices:
            if d.get("isLoopbackDevice", False) and d["name"] == exact:
                return d
        for d in devices:
            if d.get("isLoopbackDevice", False) and d["name"].startswith(out_name):
                return d
    # Fallback: first loopback device.
    for d in devices:
        if d.get("isLoopbackDevice", False):
            return d
    return None


def _find_mic(
    pa: pyaudio.PyAudio,
    devices: list[DeviceInfo],
    output_info: DeviceInfo | None,
    wasapi_api_idx: int | None,
    mic_index: int | None,
) -> DeviceInfo | None:
    """Resolve the microphone by the documented 4-step priority."""
    # 1. Explicit override.
    if mic_index is not None:
        return _info(pa, mic_index)

    # 2. Mic belonging to the same hardware as the default output, e.g. output
    #    "Headphones (Lenovo ...)" -> input "Headset (Lenovo ...)".
    if output_info:
        out_name = output_info["name"]
        hw_id = ""
        if "(" in out_name and ")" in out_name:
            hw_id = out_name[out_name.index("(") + 1 : out_name.rindex(")")]
        if hw_id and hw_id != "R":  # skip single-letter matches like "Realtek(R)"
            for d in devices:
                if (
                    d["maxInputChannels"] > 0
                    and not d.get("isLoopbackDevice", False)
                    and hw_id in d["name"]
                    and d.get("hostApi") == wasapi_api_idx
                ):
                    return d

    # 3. WASAPI default input.
    if wasapi_api_idx is not None:
        api = pa.get_host_api_info_by_index(wasapi_api_idx)
        default_in = api.get("defaultInputDevice", -1)
        if default_in >= 0:
            return _info(pa, default_in)

    # 4. Any input device.
    for d in devices:
        if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice", False):
            return d
    return None


def list_devices() -> None:
    """Print all available input and loopback devices, marking the selected ones."""
    pa = pyaudio.PyAudio()
    try:
        mic_info, loopback_info, _ = find_devices(pa)
        devices = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]

        print("\n  === Input devices ===\n")
        for d in devices:
            if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice", False):
                tag = (
                    "  <-- SELECTED MIC"
                    if mic_info and int(d["index"]) == int(mic_info["index"])
                    else ""
                )
                print(
                    f"  [{int(d['index']):>2}] {d['name']:<55} "
                    f"{d['maxInputChannels']}ch  {int(d['defaultSampleRate'])}Hz{tag}"
                )

        print("\n  === Loopback devices (system audio) ===\n")
        for d in devices:
            if d.get("isLoopbackDevice", False):
                tag = (
                    "  <-- SELECTED"
                    if loopback_info and int(d["index"]) == int(loopback_info["index"])
                    else ""
                )
                print(
                    f"  [{int(d['index']):>2}] {d['name']:<55} "
                    f"{d['maxInputChannels']}ch  {int(d['defaultSampleRate'])}Hz{tag}"
                )
        print()
    finally:
        pa.terminate()


def _prompt_device(kind: str, options: list[DeviceInfo], default_index: int | None) -> int | None:
    """Prompt for one device choice from ``options``.

    Returns the chosen device index, or None to accept the auto-detected
    default (which lets :func:`find_devices` keep its full auto behavior).
    Falls back to the default on empty input, an invalid entry, or EOF/Ctrl+C.
    """
    valid = {int(d["index"]) for d in options}
    default_hint = f" [{default_index}]" if default_index is not None else ""
    try:
        raw = input(f"  Choose {kind} device by number{default_hint} (Enter = default): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return None
    try:
        choice = int(raw)
    except ValueError:
        print(f"  Not a number; using default {kind} device.")
        return None
    if choice not in valid:
        print(f"  {choice} is not a valid {kind} device; using default.")
        return None
    # Accepting the default explicitly still returns None so auto logic applies.
    return None if choice == default_index else choice


def select_devices_interactive(
    pa: pyaudio.PyAudio,
) -> tuple[int | None, int | None]:
    """Interactively pick the mic and system (loopback) devices.

    Prints the available input and loopback devices with the auto-detected
    choices marked, then prompts for each. Returns
    ``(mic_index, loopback_index)`` where either may be None to mean "use the
    auto-detected default".
    """
    mic_info, loopback_info, _ = find_devices(pa)
    devices = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]

    inputs = [
        d for d in devices if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice", False)
    ]
    loopbacks = [d for d in devices if d.get("isLoopbackDevice", False)]

    print("\n  === Input (mic) devices ===\n")
    for d in inputs:
        tag = "  <-- default" if mic_info and int(d["index"]) == int(mic_info["index"]) else ""
        print(f"  [{int(d['index']):>2}] {d['name']:<55} {int(d['defaultSampleRate'])}Hz{tag}")
    mic_default = int(mic_info["index"]) if mic_info else None
    mic_choice = _prompt_device("mic", inputs, mic_default)

    print("\n  === System (loopback) devices ===\n")
    for d in loopbacks:
        tag = (
            "  <-- default"
            if loopback_info and int(d["index"]) == int(loopback_info["index"])
            else ""
        )
        print(f"  [{int(d['index']):>2}] {d['name']:<55} {int(d['defaultSampleRate'])}Hz{tag}")
    sys_default = int(loopback_info["index"]) if loopback_info else None
    sys_choice = _prompt_device("system", loopbacks, sys_default)
    print()

    return mic_choice, sys_choice


def select_devices() -> tuple[int | None, int | None]:
    """Open a temporary PyAudio, run the interactive picker, and clean up."""
    pa = pyaudio.PyAudio()
    try:
        return select_devices_interactive(pa)
    finally:
        pa.terminate()
