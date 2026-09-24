#!/usr/bin/env python
"""Extract the test regions of the music datasets from the feather files used for training.

This documents how ``splits/*_test.csv`` were produced; it is not needed to run the
evaluation. The feather files (internal to INA) contain one row per annotated segment,
with the subset (train/dev/test) it was assigned to. Contiguous test rows of a file are
merged into regions, written as ``file,start,stop`` (seconds).

Usage:
    python extract_splits.py /path/to/datasets/dir
"""

import argparse
from pathlib import Path

import pyarrow.feather as feather

DATASETS = {
    "mirex2015": "Mirex2015.feather",
    "openbmat": "OpenBMAT.feather",
    "seyerlehner": "SeyerlehnerMusicSpeech.feather",
}
COLUMNS = ["subset", "audio_file", "audio_sample_rate", "audio_frames_start", "audio_frames_stop"]


def test_regions(path: Path):
    df = feather.read_table(path, columns=COLUMNS, memory_map=True).to_pandas()
    df = df[df.subset == "test"].sort_values(["audio_file", "audio_frames_start"])
    regions = []
    for audio_file, rows in df.groupby("audio_file", sort=True):
        sample_rate = int(rows.audio_sample_rate.iloc[0])
        current = None
        for start, stop in zip(rows.audio_frames_start, rows.audio_frames_stop):
            if current is not None and start == current[1]:
                current[1] = stop
            else:
                if current is not None:
                    regions.append((audio_file, *current, sample_rate))
                current = [start, stop]
        regions.append((audio_file, *current, sample_rate))
    return [(Path(f).stem, a / sr, b / sr) for f, a, b, sr in regions]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets_dir", type=Path, help="directory containing the feather files")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "splits")
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)
    for name, filename in DATASETS.items():
        regions = test_regions(args.datasets_dir / filename)
        output = args.output_dir / f"{name}_test.csv"
        with open(output, "w") as f:
            f.write("file,start,stop\n")
            for file, start, stop in regions:
                f.write(f"{file},{start:.6f},{stop:.6f}\n")
        hours = sum(b - a for _, a, b in regions) / 3600
        n_files = len({r[0] for r in regions})
        print(f"{output}: {len(regions)} regions in {n_files} files, {hours:.2f} h")


if __name__ == "__main__":
    main()
