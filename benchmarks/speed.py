#!/usr/bin/env python
"""Measure the processing speed of a system on a directory of audio files.

All the systems of predict.py can be timed on the same audio, whatever the task they
address. The time includes audio decoding and excludes model loading. The files should
be on a local disk, so that reading them does not dominate. For instance, with a local
copy of the InaGVAD test audio:

    python speed.py inavamos /tmp/inaGVAD/test --device cuda --runs 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from predict import SYSTEMS, load_audio

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", choices=list(SYSTEMS))
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--token")
    parser.add_argument("--runs", type=int, default=2, help="number of runs, the fastest is reported")
    args = parser.parse_args()

    files = sorted(p for p in args.audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        raise SystemExit(f"No audio files in {args.audio_dir}")
    duration = sum(len(load_audio(f)) for f in files) / 16000

    system = SYSTEMS[args.system](device=args.device, token=args.token)
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        for f in files:
            system(f)
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(
        f"{args.system}: {len(files)} files, {duration / 3600:.2f} h of audio, runs: "
        + ", ".join(f"{t:.1f}s" for t in times)
        + f" | best {best:.1f}s = {3600 * best / duration:.1f}s per hour = {duration / best:.0f}x real time"
    )


if __name__ == "__main__":
    main()
