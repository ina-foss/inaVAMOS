#!/usr/bin/env python
"""Evaluate the predictions computed by predict.py and print result tables.

- InaGVAD (speech): evaluation code of the InaGVAD repository, on its test set
  (VadEval, 0.3 s collar), as in the InaGVAD paper.
- Mirex2015, OpenBMAT, Seyerlehner (music): frame-level precision, recall and F1 of the
  music class, without collar, pooled over the test regions of each dataset. The global
  F1 is the average of the F1 of the three datasets.

Usage:
    python evaluate.py --openbmat-dir /path/to/OpenBMAT --seyerlehner-dir /path/to/Seyerlehner
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from datasets import add_dataset_arguments, load_dataset

MUSIC_DATASETS = ["mirex2015", "openbmat", "seyerlehner"]
SYSTEM_NAMES = {
    "inavamos": "inaVAMOS",
    "inaspeechsegmenter": "inaSpeechSegmenter",
    "pyannote": "pyannote (segmentation-3.0)",
    "pyannote-legacy": "pyannote (voice-activity-detection, 2.1)",
    "silero": "Silero VAD",
    "panns": "PANNs (CNN14)",
    "yamnet": "YAMNet",
}
# Labels of each system counting as speech or music.
SPEECH_LABELS = {"speech", "male", "female"}
MUSIC_LABELS = {"music"}


def read_predictions(path: Path, labels: set) -> list:
    """Segments of ``path`` whose label (or one of its "+"-separated parts) is in ``labels``."""
    with open(path) as f:
        return [
            (float(row["start"]), float(row["stop"]))
            for row in csv.DictReader(f)
            if labels & set(row["label"].split("+"))
        ]


def complete(dataset, predictions_dir: Path) -> bool:
    missing = [item.uri for item in dataset.items() if not (predictions_dir / f"{item.uri}.csv").exists()]
    if missing and len(missing) < len(dataset.items()):
        print(f"WARNING: {predictions_dir} is incomplete ({len(missing)} files missing), skipped", file=sys.stderr)
    return not missing


def evaluate_inagvad(dataset, predictions_dir: Path) -> dict:
    """Global VAD metrics on the InaGVAD test set, with the InaGVAD evaluation code."""
    sys.path.insert(0, str(dataset.repo_dir))
    from inaGVAD.vad_metrics import VadEval

    # VadEval reads CSV files with start, stop and label columns, speech being labelled
    # "speech", "male" or "female".
    with tempfile.TemporaryDirectory() as tmp:
        for item in dataset.items():
            with open(Path(tmp) / f"{item.uri}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["start", "stop", "label"])
                writer.writerows((start, stop, "speech") for start, stop in read_predictions(predictions_dir / f"{item.uri}.csv", SPEECH_LABELS))
        _, scores = VadEval().evaluation(str(dataset.repo_dir), tmp, "test", "global")
    return {k: float(v) for k, v in scores.items()}


def _inside(times: np.ndarray, segments: list) -> np.ndarray:
    """Whether each time falls inside one of the (sorted, non-overlapping) segments."""
    if not segments:
        return np.zeros(len(times), dtype=bool)
    starts, stops = np.array(segments).T
    index = np.searchsorted(starts, times, side="right") - 1
    return (index >= 0) & (times < stops[np.clip(index, 0, None)])


def music_metrics(dataset, hypotheses: dict, frame_duration: float) -> dict:
    """Frame-level music detection metrics, pooled over the test regions of a dataset.

    ``hypotheses`` maps each file of the dataset to its predicted music segments.
    """
    from datasets import merge_segments

    tp = fp = fn = 0
    for item in dataset.items():
        hypothesis = merge_segments(hypotheses[item.uri])
        for start, stop in item.uem:
            times = start + (np.arange(int((stop - start) / frame_duration)) + 0.5) * frame_duration
            ref, hyp = _inside(times, item.reference), _inside(times, hypothesis)
            tp += int(np.sum(ref & hyp))
            fp += int(np.sum(~ref & hyp))
            fn += int(np.sum(ref & ~hyp))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "fmeasure": f1}


def evaluate_music(dataset, predictions_dir: Path, frame_duration: float) -> dict:
    hypotheses = {item.uri: read_predictions(predictions_dir / f"{item.uri}.csv", MUSIC_LABELS) for item in dataset.items()}
    return music_metrics(dataset, hypotheses, frame_duration)


def fmt(value) -> str:
    return "–" if value is None else f"{100 * value:.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-dir", type=Path, default=Path(__file__).parent / "predictions")
    parser.add_argument("--frame-duration", type=float, default=0.01, help="frame duration of the music metric (s)")
    parser.add_argument("--output", type=Path, help="also save the results as JSON")
    add_dataset_arguments(parser)
    args = parser.parse_args()

    systems = [s for s in SYSTEM_NAMES if (args.predictions_dir / s).is_dir()]
    if not systems:
        raise SystemExit(f"No predictions in {args.predictions_dir}: run predict.py first")
    results = {system: {} for system in systems}

    inagvad = load_dataset("inagvad", args)
    for system in systems:
        predictions = args.predictions_dir / system / inagvad.name
        if predictions.is_dir() and complete(inagvad, predictions):
            results[system][inagvad.name] = evaluate_inagvad(inagvad, predictions)

    for name in MUSIC_DATASETS:
        try:
            dataset = load_dataset(name, args)
        except SystemExit as e:
            print(f"WARNING: {e}", file=sys.stderr)
            continue
        for system in systems:
            predictions = args.predictions_dir / system / dataset.name
            if predictions.is_dir() and complete(dataset, predictions):
                results[system][dataset.name] = evaluate_music(dataset, predictions, args.frame_duration)

    music_names = ["Mirex2015", "OpenBMAT", "Seyerlehner"]
    for system in systems:
        scores = [results[system].get(name, {}).get("fmeasure") for name in music_names]
        if all(s is not None for s in scores):
            results[system]["global_music_f1"] = float(np.mean(scores))

    print("\n### Voice activity detection: InaGVAD test set\n")
    print("| System | Accuracy | Precision | Recall | F1 |")
    print("|--------|----------|-----------|--------|----|")
    for system in systems:
        r = results[system].get("InaGVAD")
        if r:
            print(f"| {SYSTEM_NAMES[system]} | {fmt(r['accuracy'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['fmeasure'])} |")

    print("\n### Music detection: frame-level F1 on the test sets\n")
    print("| System | " + " | ".join(music_names) + " | Global |")
    print("|--------|" + "|".join("-" * (len(n) + 2) for n in music_names) + "|--------|")
    for system in systems:
        scores = [results[system].get(name, {}).get("fmeasure") for name in music_names]
        if any(s is not None for s in scores):
            row = [fmt(s) for s in scores] + [fmt(results[system].get("global_music_f1"))]
            print(f"| {SYSTEM_NAMES[system]} | " + " | ".join(row) + " |")

    print("\nMusic precision / recall:")
    for system in systems:
        for name in music_names:
            r = results[system].get(name)
            if r:
                print(f"  {SYSTEM_NAMES[system]:24s} {name:12s} P={fmt(r['precision'])} R={fmt(r['recall'])}")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
