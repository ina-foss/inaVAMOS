"""Audio decoding to 16 kHz mono float32."""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

SAMPLING_RATE = 16000


def load_audio(path, sr: int = SAMPLING_RATE, start: float | None = None, stop: float | None = None) -> np.ndarray:
    """Decode any audio or video file to a mono float32 signal at ``sr`` Hz.

    ffmpeg is used when available (it handles virtually every media format);
    otherwise the file is decoded with librosa/soundfile.

    Args:
        path: media file path (or URL, when ffmpeg is used).
        sr: target sampling rate.
        start: optional start time, in seconds.
        stop: optional stop time, in seconds.
    """
    if start is not None and start < 0:
        raise ValueError("start must be positive")
    if start is not None and stop is not None and stop <= start:
        raise ValueError("stop must be greater than start")
    if "://" not in str(path) and not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")

    if shutil.which("ffmpeg"):
        return _load_ffmpeg(path, sr, start, stop)
    return _load_librosa(path, sr, start, stop)


def _load_ffmpeg(path, sr, start, stop):
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if stop is not None:
        cmd += ["-t", str(stop - (start or 0))]
    cmd += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}:\n{proc.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _load_librosa(path, sr, start, stop):
    import librosa

    duration = None if stop is None else stop - (start or 0)
    audio, _ = librosa.load(path, sr=sr, mono=True, offset=start or 0.0, duration=duration)
    return audio.astype(np.float32)
