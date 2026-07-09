"""Unit tests for device-selection logic using a fake PyAudio backend."""

from __future__ import annotations

from support import FakePyAudio

from meeting_recorder.devices import find_devices


def _dev(index, name, *, out=0, inp=0, loopback=False, host=1, rate=48000):
    return {
        "index": index,
        "name": name,
        "maxOutputChannels": out,
        "maxInputChannels": inp,
        "isLoopbackDevice": loopback,
        "hostApi": host,
        "defaultSampleRate": rate,
    }


def _wasapi(default_out=-1, default_in=-1):
    return [
        {"name": "MME"},
        {
            "name": "Windows WASAPI",
            "defaultOutputDevice": default_out,
            "defaultInputDevice": default_in,
        },
    ]


def _headset_scenario(default_in=5):
    """Output routed to Lenovo headset; a matching mic and both loopbacks exist."""
    devices = [
        _dev(0, "Speakers (Realtek(R) Audio)", out=2),
        _dev(1, "Headphones (Lenovo Wireless VoIP Headset)", out=2),
        _dev(2, "Headphones (Lenovo Wireless VoIP Headset) [Loopback]", inp=2, loopback=True),
        _dev(3, "Speakers (Realtek(R) Audio) [Loopback]", inp=2, loopback=True),
        _dev(4, "Headset (Lenovo Wireless VoIP Headset)", inp=1, rate=16000),
        _dev(5, "Microphone Array (Intel Smart Sound)", inp=2),
    ]
    return devices, _wasapi(default_out=1, default_in=default_in)


def test_selects_hardware_matched_mic():
    devices, apis = _headset_scenario()
    pa = FakePyAudio(devices, apis)
    mic, loopback, output = find_devices(pa)
    assert output["index"] == 1
    assert mic["index"] == 4  # Lenovo Headset mic, matched by hardware name


def test_loopback_exact_pairing_preferred():
    devices, apis = _headset_scenario()
    pa = FakePyAudio(devices, apis)
    _, loopback, _ = find_devices(pa)
    assert loopback["name"] == "Headphones (Lenovo Wireless VoIP Headset) [Loopback]"


def test_explicit_mic_override_wins():
    devices, apis = _headset_scenario()
    pa = FakePyAudio(devices, apis)
    mic, _, _ = find_devices(pa, mic_index=5)
    assert mic["index"] == 5


def test_falls_back_to_default_input_when_no_hardware_match():
    # Output is Realtek speakers with no matching-hardware mic -> use WASAPI
    # default input device (index 2).
    devices = [
        _dev(0, "Speakers (Realtek(R) Audio)", out=2),
        _dev(1, "Speakers (Realtek(R) Audio) [Loopback]", inp=2, loopback=True),
        _dev(2, "Microphone Array (Intel Smart Sound)", inp=2),
    ]
    apis = _wasapi(default_out=0, default_in=2)
    pa = FakePyAudio(devices, apis)
    mic, loopback, output = find_devices(pa)
    assert output["index"] == 0
    assert loopback["index"] == 1
    assert mic["index"] == 2


def test_loopback_prefix_fallback_when_no_exact_pairing():
    # No "<output> [Loopback]" exact name; a prefix-matching loopback is used.
    devices = [
        _dev(0, "Speakers (Realtek(R) Audio)", out=2),
        _dev(1, "Speakers (Realtek(R) Audio) [Loopback] (2)", inp=2, loopback=True),
        _dev(2, "Microphone Array (Intel Smart Sound)", inp=2),
    ]
    apis = _wasapi(default_out=0, default_in=2)
    pa = FakePyAudio(devices, apis)
    _, loopback, _ = find_devices(pa)
    assert loopback["index"] == 1


def test_explicit_loopback_moves_keepalive_output_to_match():
    # Default output is the Lenovo headset (index 1), but the user explicitly
    # picks the Realtek loopback (index 3). The keepalive/output device must
    # follow that choice to its feeding output (index 0), otherwise the chosen
    # endpoint stays idle and the system track comes back empty.
    devices, apis = _headset_scenario()
    pa = FakePyAudio(devices, apis)
    _, loopback, output = find_devices(pa, loopback_index=3)
    assert loopback["index"] == 3
    assert output["index"] == 0  # Speakers (Realtek), not the default headset


def test_explicit_loopback_without_matching_output_keeps_default():
    # If no output matches the chosen loopback's name, fall back to the default
    # output rather than losing the keepalive entirely.
    devices = [
        _dev(0, "Speakers (Realtek(R) Audio)", out=2),
        _dev(1, "Orphan Loopback [Loopback]", inp=2, loopback=True),
        _dev(2, "Microphone Array (Intel Smart Sound)", inp=2),
    ]
    apis = _wasapi(default_out=0, default_in=2)
    pa = FakePyAudio(devices, apis)
    _, loopback, output = find_devices(pa, loopback_index=1)
    assert loopback["index"] == 1
    assert output["index"] == 0  # default output preserved
