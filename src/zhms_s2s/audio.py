from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .errors import InvalidAudioError


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".flac",
    ".ogg",
    ".aiff",
    ".aif",
    ".m4a",
    ".aac",
    ".mp3",
}

DECODE_ERROR = "Audio could not be decoded as WAV/FLAC/OGG/AIFF/M4A/AAC/MP3."


@dataclass(slots=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    duration_s: float


def preprocess_audio(
    audio: AudioData,
    *,
    low_rms_threshold: float = 0.025,
    target_rms: float = 0.08,
    max_gain: float = 4.0,
) -> AudioData:
    """Apply conservative ASR input conditioning without denoising speech.

    DC removal is always safe for microphone input. Gain is only raised for
    demonstrably quiet recordings and is peak-limited, so ordinary recordings
    are not compressed or otherwise reshaped.
    """

    samples = np.asarray(audio.samples, dtype=np.float32)
    centered = samples - np.mean(samples, dtype=np.float64)
    rms = float(np.sqrt(np.mean(centered * centered) + 1e-12))
    peak = float(np.max(np.abs(centered))) if centered.size else 0.0
    gain = 1.0
    if 1e-5 < rms < low_rms_threshold and peak > 0:
        gain = min(max_gain, target_rms / rms, 0.95 / peak)
    processed = np.ascontiguousarray(centered * gain, dtype=np.float32)
    return AudioData(
        samples=processed,
        sample_rate=audio.sample_rate,
        duration_s=audio.duration_s,
    )


def _decode_with_pyav(
    data: bytes,
    *,
    target_sample_rate: int,
    max_seconds: float,
) -> tuple[np.ndarray, int]:
    container = None
    try:
        container = av.open(io.BytesIO(data), mode="r")
        if not container.streams.audio:
            raise InvalidAudioError(DECODE_ERROR)

        stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="fltp",
            layout="mono",
            rate=target_sample_rate,
        )
        chunks: list[np.ndarray] = []
        sample_count = 0
        max_samples = int(target_sample_rate * max_seconds)

        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunk = np.asarray(
                    resampled.to_ndarray(), dtype=np.float32
                ).reshape(-1)
                sample_count += len(chunk)
                if sample_count > max_samples:
                    duration = sample_count / float(target_sample_rate)
                    raise InvalidAudioError(
                        f"Audio duration {duration:.2f}s exceeds the "
                        f"{max_seconds:.0f}s limit."
                    )
                chunks.append(chunk)

        for resampled in resampler.resample(None):
            chunk = np.asarray(resampled.to_ndarray(), dtype=np.float32).reshape(-1)
            sample_count += len(chunk)
            if sample_count > max_samples:
                duration = sample_count / float(target_sample_rate)
                raise InvalidAudioError(
                    f"Audio duration {duration:.2f}s exceeds the "
                    f"{max_seconds:.0f}s limit."
                )
            chunks.append(chunk)

        if not chunks:
            raise InvalidAudioError("Decoded audio contains no samples.")
        return np.concatenate(chunks), target_sample_rate
    except InvalidAudioError:
        raise
    except Exception as exc:
        raise InvalidAudioError(DECODE_ERROR) from exc
    finally:
        if container is not None:
            container.close()


def decode_audio_bytes(
    data: bytes,
    *,
    filename: str | None,
    target_sample_rate: int = 16000,
    max_seconds: float = 300.0,
) -> AudioData:
    if not data:
        raise InvalidAudioError("Uploaded audio is empty.")
    suffix = Path(filename or "audio.wav").suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise InvalidAudioError(
            f"Unsupported audio format '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    try:
        samples, sample_rate = sf.read(
            io.BytesIO(data), dtype="float32", always_2d=True
        )
    except Exception:
        samples, sample_rate = _decode_with_pyav(
            data,
            target_sample_rate=target_sample_rate,
            max_seconds=max_seconds,
        )
    if samples.size == 0 or samples.shape[0] == 0:
        raise InvalidAudioError("Decoded audio contains no samples.")
    if not np.isfinite(samples).all():
        raise InvalidAudioError("Audio contains invalid numeric samples.")

    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    duration = len(mono) / float(sample_rate)
    if duration <= 0.02:
        raise InvalidAudioError("Audio is too short; provide at least 20 ms.")
    if duration > max_seconds:
        raise InvalidAudioError(
            f"Audio duration {duration:.2f}s exceeds the {max_seconds:.0f}s limit."
        )
    if float(np.max(np.abs(mono))) < 1e-5:
        raise InvalidAudioError("Audio is silent.")

    if sample_rate != target_sample_rate:
        from math import gcd

        divisor = gcd(sample_rate, target_sample_rate)
        mono = resample_poly(
            mono, target_sample_rate // divisor, sample_rate // divisor
        ).astype(np.float32)
        sample_rate = target_sample_rate
    return AudioData(
        samples=np.ascontiguousarray(mono, dtype=np.float32),
        sample_rate=sample_rate,
        duration_s=duration,
    )


def audio_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise InvalidAudioError(f"Generated audio is unreadable: {path}") from exc
    duration = float(info.frames) / float(info.samplerate)
    if info.frames <= 0 or duration <= 0:
        raise InvalidAudioError(f"Generated audio contains no frames: {path}")
    return duration
