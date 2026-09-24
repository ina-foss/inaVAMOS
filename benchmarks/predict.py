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


def load_audio(path, sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
    """Decode any media file to mono float32 with ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path)]
    cmd += ["-map", "0:a:0", "-ac", "1", "-ar", str(sampling_rate), "-f", "f32le", "-"]
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


class Silero(System):
    """Silero VAD with its default settings."""

    tasks = ("speech",)
    packages = ("silero-vad", "torch")

    def __init__(self, device=None, token=None):
        from silero_vad import load_silero_vad

        # Silero VAD is designed to run on CPU.
        self.model = load_silero_vad()

    def __call__(self, path):
        import torch
        from silero_vad import get_speech_timestamps

        wav = torch.from_numpy(load_audio(path))
        timestamps = get_speech_timestamps(wav, self.model, sampling_rate=SAMPLING_RATE, return_seconds=True)
        return [("speech", t["start"], t["end"]) for t in timestamps]


class FrameMusicDetector(System):
    """AudioSet tagger used as a music detector: frames where the "Music" score exceeds a
    threshold, tuned on the dev subsets of the music datasets (see tune_thresholds.py)."""

    tasks = ("music",)
    threshold: float

    def __init__(self, device=None, token=None, threshold=None):
        if threshold is not None:
            self.threshold = threshold

    def scores(self, path):
        """Music scores of the frames of an audio file: (starts, stops, scores) arrays."""
        raise NotImplementedError

    def __call__(self, path):
        starts, stops, scores = self.scores(path)
        return frames_to_segments(starts, stops, scores > self.threshold, "music")

    def info(self):
        return {**super().info(), "threshold": self.threshold}


def frames_to_segments(starts, stops, active, label):
    """Merge consecutive active frames into (label, start, stop) segments."""
    segments = []
    for start, stop, on in zip(starts, stops, active):
        if not on:
            continue
        if segments and start <= segments[-1][2] + 1e-6:
            segments[-1][2] = float(stop)
        else:
            segments.append([label, float(start), float(stop)])
    return [tuple(s) for s in segments]


class PANNs(FrameMusicDetector):
    """PANNs CNN14 sound event detection model (Cnn14_DecisionLevelMax, AudioSet), "Music" class.

    Audio is processed by 10 s chunks, the duration of the training clips, with 10 ms frames.
    """

    packages = ("panns-inference", "torch")
    # Tuned on the dev subsets with tune_thresholds.py (global dev F1 88.8).
    threshold = 0.01
    SAMPLING_RATE = 32000
    CHUNK = 10 * 32000
    HOP = 320

    def __init__(self, device=None, token=None, threshold=None, batch_size=16):
        super().__init__(threshold=threshold)
        import torch
        from panns_inference import SoundEventDetection, labels

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sed = SoundEventDetection(checkpoint_path=None, device=device)
        self.index = labels.index("Music")
        self.batch_size = batch_size

    def scores(self, path):
        audio = load_audio(path, self.SAMPLING_RATE)
        n_chunks = max(1, -(-len(audio) // self.CHUNK))
        padded = np.zeros(n_chunks * self.CHUNK, dtype=np.float32)
        padded[: len(audio)] = audio
        chunks = padded.reshape(n_chunks, self.CHUNK)
        frames_per_chunk = self.CHUNK // self.HOP
        scores = [
            self.sed.inference(chunks[i : i + self.batch_size])[:, :frames_per_chunk, self.index]
            for i in range(0, n_chunks, self.batch_size)
        ]
        scores = np.concatenate(scores).reshape(-1)[: -(-len(audio) // self.HOP)]
        starts = np.arange(len(scores)) * self.HOP / self.SAMPLING_RATE
        stops = np.minimum(starts + self.HOP / self.SAMPLING_RATE, len(audio) / self.SAMPLING_RATE)
        return starts, stops, scores


class YAMNet(FrameMusicDetector):
    """YAMNet (AudioSet) from TensorFlow Hub, "Music" class.

    YAMNet scores 0.96 s windows every 0.48 s: each score is assigned to the 0.48 s
    centred in its window.
    """

    packages = ("tensorflow", "tensorflow-hub")
    # Tuned on the dev subsets with tune_thresholds.py (global dev F1 86.6).
    threshold = 0.0032
    HOP = 0.48
    WINDOW = 0.96

    def __init__(self, device=None, token=None, threshold=None):
        super().__init__(threshold=threshold)
        import csv as csv_module

        import tensorflow as tf
        import tensorflow_hub as hub

        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        with tf.io.gfile.GFile(self.model.class_map_path().numpy()) as f:
            names = [row["display_name"] for row in csv_module.DictReader(f)]
        self.index = names.index("Music")

    def scores(self, path):
        audio = load_audio(path)
        duration = len(audio) / SAMPLING_RATE
        scores = self.model(audio)[0].numpy()[:, self.index]
        centers = np.arange(len(scores)) * self.HOP + self.WINDOW / 2
        starts = np.clip(centers - self.HOP / 2, 0, duration)
        stops = np.clip(centers + self.HOP / 2, 0, duration)
        starts[0], stops[-1] = 0.0, duration
        return starts, stops, scores


SYSTEMS = {
    "inavamos": InaVAMOS,
    "inaspeechsegmenter": InaSpeechSegmenter,
    "pyannote": Pyannote,
    "pyannote-legacy": PyannoteLegacy,
    "silero": Silero,
    "panns": PANNs,
    "yamnet": YAMNet,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", choices=list(SYSTEMS))
    parser.add_argument("datasets", nargs="+", choices=list(DATASETS), metavar="dataset", help=f"among {', '.join(DATASETS)}")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "predictions")
    parser.add_argument("--device", help="device (inavamos, pyannote, panns; yamnet and inaspeechsegmenter use the GPU if available; silero runs on CPU)")
    parser.add_argument("--token", help="HuggingFace token (default: HF_TOKEN or `hf auth login`)")
    parser.add_argument("--overwrite", action="store_true", help="recompute existing predictions")
    parser.add_argument("--limit", type=int, help="only process the first N files of each dataset (for testing)")
    parser.add_argument(
        "--subset", choices=["test", "dev"], default="test", help="music datasets: test regions, or dev regions for threshold tuning"
    )
    parser.add_argument("--threshold", type=float, help="decision threshold of panns and yamnet (default: tuned on dev)")
    add_dataset_arguments(parser)
    args = parser.parse_args()

    system_class = SYSTEMS[args.system]
    datasets = [load_dataset(name, args, args.subset) for name in args.datasets if not (name == "inagvad" and args.subset != "test")]
    for dataset in datasets:
        if dataset.task not in system_class.tasks:
            print(f"Skipping {dataset.name}: {args.system} does not detect {dataset.task}")
    datasets = [d for d in datasets if d.task in system_class.tasks]
    if not datasets:
        return

    kwargs = {"threshold": args.threshold} if issubclass(system_class, FrameMusicDetector) else {}
    system = system_class(device=args.device, token=args.token, **kwargs)
    for dataset in datasets:
        output_dir = args.output_dir / args.system / dataset.name
        output_dir.mkdir(parents=True, exist_ok=True)
        items = dataset.items()[: args.limit]
        todo = [item for item in items if args.overwrite or not (output_dir / f"{item.uri}.csv").exists()]
        print(f"{args.system} on {dataset.name}: {len(todo)} files to process ({len(items) - len(todo)} done)")

        elapsed = 0.0
        for i, item in enumerate(todo, 1):
            t0 = time.perf_counter()
            if isinstance(system, FrameMusicDetector):
                # Keep the frame scores, to tune the threshold without recomputing them.
                starts, stops, scores = system.scores(dataset.audio(item.uri))
                segments = frames_to_segments(starts, stops, scores > system.threshold, "music")
            else:
                segments = system(dataset.audio(item.uri))
            elapsed += time.perf_counter() - t0
            if isinstance(system, FrameMusicDetector):
                np.savez_compressed(output_dir / f"{item.uri}.scores.npz", starts=starts, stops=stops, scores=scores)
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
