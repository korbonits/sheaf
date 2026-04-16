"""Shared audio encoding/decoding utilities for audio backends."""

from __future__ import annotations

import io
import struct
import tempfile

import numpy as np

_WHISPER_SAMPLE_RATE = 16000


def decode_audio(audio_bytes: bytes) -> np.ndarray | str:
    """Decode audio bytes to a float32 numpy array at 16kHz if possible.

    WAV PCM (16 or 32-bit, mono or stereo) is decoded without ffmpeg.
    All other formats fall back to a named temp file — the calling backend
    passes the path to the model, which invokes ffmpeg internally.

    Returns:
        np.ndarray: float32 waveform at 16kHz for WAV input.
        str: temp file path for non-WAV formats (caller owns cleanup).
    """
    try:
        return _decode_wav(audio_bytes)
    except Exception:
        tmp = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        return tmp.name


def _decode_wav(data: bytes) -> np.ndarray:
    """Parse a 16-bit or 32-bit PCM WAV and return float32 at 16kHz."""
    buf = io.BytesIO(data)
    riff, _, wave = struct.unpack("<4sI4s", buf.read(12))
    if riff != b"RIFF" or wave != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file")

    num_channels = sample_rate = bits_per_sample = None
    audio_data: bytes = b""
    while True:
        chunk_header = buf.read(8)
        if len(chunk_header) < 8:
            break
        chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
        chunk_data = buf.read(chunk_size)
        if chunk_id == b"fmt ":
            audio_fmt, num_channels, sample_rate, _, _, bits_per_sample = struct.unpack(
                "<HHIIHH", chunk_data[:16]
            )
            if audio_fmt != 1:
                raise ValueError(f"Non-PCM WAV format {audio_fmt}")
        elif chunk_id == b"data":
            audio_data = chunk_data

    if bits_per_sample not in (16, 32) or not audio_data:
        raise ValueError("Unsupported WAV: need 16 or 32-bit PCM with data")

    dtype = np.int16 if bits_per_sample == 16 else np.int32
    samples = np.frombuffer(audio_data, dtype=dtype).astype(np.float32)
    samples /= np.iinfo(dtype).max

    if num_channels and num_channels > 1:
        samples = samples.reshape(-1, num_channels).mean(axis=1)

    if sample_rate and sample_rate != _WHISPER_SAMPLE_RATE:
        target_len = int(len(samples) * _WHISPER_SAMPLE_RATE / sample_rate)
        samples = np.interp(
            np.linspace(0, len(samples), target_len),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)

    return samples


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 mono audio array to 16-bit PCM WAV bytes.

    Args:
        audio: 1-D float32 numpy array with values in [-1, 1].
        sample_rate: Sample rate in Hz.

    Returns:
        Raw WAV file bytes (RIFF/WAVE, 16-bit PCM, mono).
    """
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    audio_data = audio_int16.tobytes()
    data_size = len(audio_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + audio_data
