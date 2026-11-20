from __future__ import annotations

import io

import av
import numpy as np
import pytest
import soundfile as sf

from zhms_s2s.audio import decode_audio_bytes
from zhms_s2s.errors import InvalidAudioError


def wav_bytes(sample_rate: int = 8000, seconds: float = 0.2) -> bytes:
    samples = np.sin(2 * np.pi * 220 * np.arange(int(sample_rate * seconds)) / sample_rate).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    return buffer.getvalue()


def compressed_audio_bytes(container_format: str, codec: str) -> bytes:
    sample_rate = 16000
    buffer = io.BytesIO()
    output = av.open(buffer, mode="w", format=container_format)
    stream = output.add_stream(codec, rate=sample_rate)
    stream.layout = "mono"
    samples = (
        np.sin(2 * np.pi * 220 * np.arange(sample_rate // 5) / sample_rate)
        * 10000
    ).astype(np.int16).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
    frame.sample_rate = sample_rate
    for packet in stream.encode(frame):
        output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()
    return buffer.getvalue()


def test_decode_resamples_to_16khz():
    result = decode_audio_bytes(wav_bytes(), filename="sample.wav")
    assert result.sample_rate == 16000
    assert result.samples.ndim == 1
    assert result.duration_s == pytest.approx(0.2, rel=0.02)


@pytest.mark.parametrize(
    ("container_format", "codec", "filename"),
    [
        ("mp4", "aac", "recording.m4a"),
        ("adts", "aac", "recording.aac"),
        ("mp3", "mp3", "recording.mp3"),
        ("mp4", "aac", "recording.wav"),
    ],
)
def test_decode_compressed_audio(container_format, codec, filename):
    result = decode_audio_bytes(
        compressed_audio_bytes(container_format, codec),
        filename=filename,
    )
    assert result.sample_rate == 16000
    assert result.samples.ndim == 1
    assert result.duration_s > 0.15


@pytest.mark.parametrize(
    ("data", "filename"),
    [
        (b"", "empty.wav"),
        (b"not audio", "bad.wav"),
        (b"not audio", "bad.mp3"),
        (wav_bytes(), "sample.txt"),
    ],
)
def test_invalid_audio_is_rejected(data, filename):
    with pytest.raises(InvalidAudioError):
        decode_audio_bytes(data, filename=filename)
