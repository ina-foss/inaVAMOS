#!/usr/bin/env python
"""Tune the decision threshold of the music detectors that need one (panns, yamnet).

The threshold maximizes the global music F1 (average of the frame-level F1 of Mirex2015,
OpenBMAT and Seyerlehner) on the *dev* subsets, so that the test subsets stay unseen.
It uses the frame scores saved by predict.py:

    python predict.py panns mirex2015 openbmat seyerlehner --subset dev $DATA
    python tune_thresholds.py panns $DATA

The best threshold is then the default of the system in predict.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from datasets import add_dataset_arguments, load_dataset
from evaluate import MUSIC_DATASETS, music_metrics


def segments(starts, stops, active):
    return [(a, b) for a, b, on in zip(starts, stops, active) if on]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", choices=["panns", "yamnet"])
    parser.add_argument("--predictions-dir", type=Path, default=Path(__file__).parent / "predictions")
    parser.add_argument("--frame-duration", type=float, default=0.01)
    add_dataset_arguments(parser)
    args = parser.parse_args()

    datasets = [load_dataset(name, args, "dev") for name in MUSIC_DATASETS]
    scores = {}
    for dataset in datasets:
        directory = args.predictions_dir / args.system / dataset.name
        scores[dataset.name] = {item.uri: np.load(directory / f"{item.uri}.scores.npz") for item in dataset.items()}

    # Log-spaced below 0.02: AudioSet "Music" scores are often low on broadcast audio.
    thresholds = np.unique(np.round(np.concatenate([np.geomspace(0.001, 0.02, 14), np.arange(0.02, 1.0, 0.02)]), 4))
    results = []
    for threshold in thresholds:
        f1 = []
        for dataset in datasets:
            hypotheses = {
                uri: segments(s["starts"], s["stops"], s["scores"] > threshold) for uri, s in scores[dataset.name].items()
            }
            f1.append(music_metrics(dataset, hypotheses, args.frame_duration)["fmeasure"])
        results.append((float(np.mean(f1)), threshold, f1))

    # Reference point: predicting music everywhere.
    always = [
        music_metrics(d, {item.uri: item.uem for item in d.items()}, args.frame_duration)["fmeasure"] for d in datasets
    ]

    print(f"{args.system}: global dev F1 per threshold")
    for mean, threshold, f1 in results:
        print(f"  {threshold:.4f}: {100 * mean:.1f}  (" + ", ".join(f"{d.name} {100 * v:.1f}" for d, v in zip(datasets, f1)) + ")")
    print(f"  always music: {100 * np.mean(always):.1f}  (" + ", ".join(f"{d.name} {100 * v:.1f}" for d, v in zip(datasets, always)) + ")")
    best = max(results)
    print(f"Best threshold: {best[1]:.4f} (global dev F1 {100 * best[0]:.1f})")


if __name__ == "__main__":
    main()
