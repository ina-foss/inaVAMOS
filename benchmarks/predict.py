#!/usr/bin/env python
"""Run a segmentation system on an evaluation dataset and save its predictions.

Predictions are written to ``<output-dir>/<system>/<dataset>/<file>.csv``, with
``label,start,stop`` columns (seconds), one file per audio file. Files already processed
are skipped, so interrupted runs can be resumed.

Each system has its own dependencies and should be run in its own environment
(see README.md):

    python predict.py inavamos inagvad --inagvad-dir /path/to/inaGVAD
    python predict.py inaspeechsegmenter mirex2015
    python predict.py pyannote inagvad --inagvad-dir /path/to/inaGVAD
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from datasets import DATASETS, add_dataset_arguments, load_dataset

SAMPLING_RATE = 16000


def load_audio(path) -> np.ndarray:
    """Decode any media file to 16 kHz mono float32 with ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path)]
    cmd += ["-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLING_RATE), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32).copy()


class System:
    """A segmentation system, returning ``(label, start, stop)`` segments for an audio file."""

    tasks = ("speech", "music")
    packages: tuple = ()

    def __call__(self, path: Path) -> list:
        raise NotImplementedError

    def info(self) -> dict:
        return {"packages": {p: metadata.version(p) for p in self.packages}}


class InaVAMOS(System):
    packages = ("inaVAMOS", "torch", "transformers")

    def __init__(self, device=None, token=None):
        from inavamos import Segmenter

        self.segmenter = Segmenter(device=device, token=token)

    def __call__(self, path):
        # Labels: speech, music, speech+music, other.
        return self.segmenter(str(path))

    def info(self):
        info = super().info()
        info["models"] = {name: spec.repo_id for name, spec in self.segmenter.specs.items()}
        return info


class InaSpeechSegmenter(System):
    """inaSpeechSegmenter with its default configuration, as in the InaGVAD paper."""

    packages = ("inaSpeechSegmenter", "tensorflow")

    def __init__(self, **kwargs):
        from inaSpeechSegmenter import Segmenter

        self.segmenter = Segmenter()

    def __call__(self, path):
        # Labels: male, female, music, noise, noEnergy.
        return [tuple(s) for s in self.segmenter(str(path))]


class Pyannote(System):
    """pyannote.audio >= 3 voice activity detection with pyannote/segmentation-3.0."""

    tasks = ("speech",)
    packages = ("pyannote.audio", "torch")

    def __init__(self, device=None, token=None):
        import torch
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection

        try:
            model = Model.from_pretrained("pyannote/segmentation-3.0", token=token)
        except TypeError:  # pyannote.audio 3.x
            model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
        if model is None:
            raise SystemExit("Cannot load pyannote/segmentation-3.0: accept its conditions on the Hub and provide a token")
        self.pipeline = VoiceActivityDetection(segmentation=model)
        # Hyper-parameters recommended on the model card.
        self.pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
        if device:
            self.pipeline.to(torch.device(device))

    def _waveform(self, path):
        import torch

        return {"waveform": torch.from_numpy(load_audio(path))[None], "sample_rate": SAMPLING_RATE}

    def __call__(self, path):
        output = self.pipeline(self._waveform(path))
        return [("speech", s.start, s.end) for s in output.get_timeline().support()]


class PyannoteLegacy(Pyannote):
    """pyannote.audio 2.x pyannote/voice-activity-detection pipeline, as in the InaGVAD paper."""

    def __init__(self, device=None, token=None):
        import torch
        from pyannote.audio import Pipeline

        try:
            self.pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=token)
        except AttributeError:  # the segmentation model could not be downloaded
            self.pipeline = None
        if self.pipeline is None:
            raise SystemExit(
                "Cannot load pyannote/voice-activity-detection: accept the conditions of "
                "pyannote/voice-activity-detection and pyannote/segmentation on the Hub and provide a token"
            )
        if device:
            # pyannote.audio 2.x pipelines have no .to(): move their inference object.
            inference = self.pipeline._segmentation
            inference.device = torch.device(device)
            inference.model.to(inference.device)


SYSTEMS = {
    "inavamos": InaVAMOS,
    "inaspeechsegmenter": InaSpeechSegmenter,
    "pyannote": Pyannote,
    "pyannote-legacy": PyannoteLegacy,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", choices=list(SYSTEMS))
    parser.add_argument("datasets", nargs="+", choices=list(DATASETS), metavar="dataset", help=f"among {', '.join(DATASETS)}")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "predictions")
    parser.add_argument("--device", help="torch device (inavamos, pyannote)")
    parser.add_argument("--token", help="HuggingFace token (default: HF_TOKEN or `hf auth login`)")
    parser.add_argument("--overwrite", action="store_true", help="recompute existing predictions")
    parser.add_argument("--limit", type=int, help="only process the first N files of each dataset (for testing)")
    add_dataset_arguments(parser)
    args = parser.parse_args()

    system_class = SYSTEMS[args.system]
    datasets = [load_dataset(name, args) for name in args.datasets]
    for dataset in datasets:
        if dataset.task not in system_class.tasks:
            print(f"Skipping {dataset.name}: {args.system} does not detect {dataset.task}")
    datasets = [d for d in datasets if d.task in system_class.tasks]
    if not datasets:
        return

    system = system_class(device=args.device, token=args.token)
    for dataset in datasets:
        output_dir = args.output_dir / args.system / dataset.name
        output_dir.mkdir(parents=True, exist_ok=True)
        items = dataset.items()[: args.limit]
        todo = [item for item in items if args.overwrite or not (output_dir / f"{item.uri}.csv").exists()]
        print(f"{args.system} on {dataset.name}: {len(todo)} files to process ({len(items) - len(todo)} done)")

        elapsed = 0.0
        for i, item in enumerate(todo, 1):
            t0 = time.perf_counter()
            segments = system(dataset.audio(item.uri))
            elapsed += time.perf_counter() - t0
            with open(output_dir / f"{item.uri}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["label", "start", "stop"])
                writer.writerows((label, f"{start:.3f}", f"{stop:.3f}") for label, start, stop in segments)
            print(f"  [{i}/{len(todo)}] {item.uri}: {elapsed:.0f}s elapsed", flush=True)

        if todo:
            info = {"system": args.system, "dataset": dataset.name, "files": len(todo), "processing_time": elapsed}
            info.update(system.info())
            info["device"] = args.device or "default"
            info["platform"] = platform.platform()
            (output_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
