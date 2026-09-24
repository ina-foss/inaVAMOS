"""Evaluation datasets, read from their original distribution format.

Each dataset yields :class:`Item` objects: the regions to evaluate (UEM) and, for music
datasets, the reference music segments. ``Dataset.audio(uri)`` gives the audio file, which
is only needed to compute predictions.

Datasets that can be downloaded automatically are cached in ``$INAVAMOS_BENCHMARK_DATA``
(default: ``~/.cache/inavamos-benchmarks``). The others need a local copy.
"""

from __future__ import annotations

import csv
import os
import subprocess
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

SPLITS_DIR = Path(__file__).parent / "splits"
CACHE_DIR = Path(os.environ.get("INAVAMOS_BENCHMARK_DATA", Path.home() / ".cache" / "inavamos-benchmarks"))

INAGVAD_REPO = "https://github.com/ina-foss/InaGVAD.git"
MIREX2015_URL = "http://mirg.city.ac.uk/datasets/muspeak/muspeak-mirex2015-detection-examples.zip"


@dataclass
class Item:
    uri: str
    # Regions to evaluate, in seconds.
    uem: list = field(default_factory=list)
    # Reference segments where the target class (music) is present, in seconds.
    reference: list = field(default_factory=list)


def merge_segments(segments):
    """Union of possibly overlapping (start, stop) segments."""
    merged = []
    for start, stop in sorted(segments):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [tuple(s) for s in merged]


def read_split(name: str) -> dict:
    """Test regions of a music dataset: {file: [(start, stop), ...]}."""
    regions = {}
    with open(SPLITS_DIR / f"{name}_test.csv") as f:
        for row in csv.DictReader(f):
            regions.setdefault(row["file"], []).append((float(row["start"]), float(row["stop"])))
    return regions


def _download(url: str, path: Path):
    if not path.exists():
        print(f"Downloading {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path.with_suffix(".part"))
        path.with_suffix(".part").rename(path)


def _require_dir(path, name: str, option: str, help_url: str) -> Path:
    if path is None:
        raise SystemExit(f"{name} cannot be downloaded automatically: download it from {help_url} and use {option}")
    path = Path(path)
    if not path.is_dir():
        raise SystemExit(f"{name}: {path} is not a directory")
    return path


def _find_audio(root: Path, stem: str, extensions=(".wav", ".mp3")) -> Path:
    for ext in extensions:
        for candidate in (root / f"{stem}{ext}", root / f"{stem.replace('_', ' ')}{ext}"):
            if candidate.exists():
                return candidate
    matches = [p for ext in extensions for p in root.rglob(f"{stem}{ext}")]
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot find the audio file of {stem} in {root} ({len(matches)} matches)")
    return matches[0]


class Dataset:
    name: str
    task: str  # "speech" or "music"

    def items(self) -> list[Item]:
        raise NotImplementedError

    def audio(self, uri: str) -> Path:
        raise NotImplementedError


class InaGVAD(Dataset):
    """InaGVAD test set (VAD). Audio: https://www.ina.fr/institut-national-audiovisuel/research/dataset-project

    The annotations and the evaluation code come from the InaGVAD repository, cloned
    automatically unless ``repo_dir`` is given.
    """

    name = "InaGVAD"
    task = "speech"

    def __init__(self, audio_dir=None, repo_dir=None):
        self.audio_dir = audio_dir
        if repo_dir is None:
            repo_dir = CACHE_DIR / "InaGVAD"
            if not repo_dir.exists():
                print(f"Cloning {INAGVAD_REPO}")
                subprocess.run(["git", "clone", "-q", "--depth", "1", INAGVAD_REPO, str(repo_dir)], check=True)
        self.repo_dir = Path(repo_dir)

    def items(self):
        with open(self.repo_dir / "annotations" / "filesplit" / "testset.csv") as f:
            uris = [row["fileid"] for row in csv.DictReader(f)]
        return [Item(uri) for uri in uris]

    def audio(self, uri):
        root = _require_dir(
            self.audio_dir, "InaGVAD audio", "--inagvad-dir", "https://www.ina.fr/institut-national-audiovisuel/research/dataset-project"
        )
        return _find_audio(root, uri, (".wav",))


class Mirex2015(Dataset):
    """MIREX 2015 music/speech detection examples (7 files, test regions only)."""

    name = "Mirex2015"
    task = "music"

    def __init__(self, root=None):
        if root is None:
            archive = CACHE_DIR / "muspeak-mirex2015-detection-examples.zip"
            root = CACHE_DIR / "Mirex2015"
            if not root.exists():
                _download(MIREX2015_URL, archive)
                with zipfile.ZipFile(archive) as z:
                    z.extractall(root)
        self.root = Path(root)

    def _annotation(self, stem: str) -> Path:
        # Annotations with a _v2 suffix are corrected versions.
        for candidate in (self.root / f"{stem}_v2.csv", self.root / f"{stem}.csv"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No annotation for {stem} in {self.root}")

    def items(self):
        items = []
        for stem, uem in read_split("mirex2015").items():
            music = []
            with open(self._annotation(stem), encoding="utf-8-sig") as f:
                for row in csv.reader(f):
                    if len(row) >= 3 and row[2].strip() == "m":
                        onset, duration = float(row[0]), float(row[1])
                        music.append((onset, onset + duration))
            items.append(Item(stem, uem, merge_segments(music)))
        return items

    def audio(self, uri):
        return _find_audio(self.root, uri, (".mp3",))


class OpenBMAT(Dataset):
    """OpenBMAT test files. Download (restricted access): https://zenodo.org/records/3381249

    Music is present where at least one of the three annotators marked any music class
    (music, foreground, similar, background, very low background), as in the training data.
    """

    name = "OpenBMAT"
    task = "music"

    def __init__(self, root=None):
        self.root = _require_dir(root, "OpenBMAT", "--openbmat-dir", "https://zenodo.org/records/3381249")

    def items(self):
        annotations = self.root / "annotations" / "tsv" / "MD_mapping"
        annotators = sorted(p for p in annotations.iterdir() if p.is_dir())
        items = []
        for stem, uem in read_split("openbmat").items():
            music = []
            for annotator in annotators:
                with open(annotator / f"{stem}.tsv") as f:
                    for row in csv.reader(f, delimiter="\t"):
                        if len(row) >= 3 and row[2].strip() == "music":
                            music.append((float(row[0]), float(row[1])))
            items.append(Item(stem, uem, merge_segments(music)))
        return items

    def audio(self, uri):
        return self.root / "audio" / f"{uri}.wav"


class Seyerlehner(Dataset):
    """Music/speech dataset of Seyerlehner et al. (DAFx 2007), test regions only.

    Expects the original distribution: ``<name>.mp3`` audio files and ``<name>.wav.ref.label``
    annotations (MATLAB files with 100 Hz frame labels: 0 for speech, 1 for music). The
    directory may also be replaced by the original ``.tgz`` archive.
    """

    name = "Seyerlehner"
    task = "music"

    def __init__(self, root=None):
        if root is not None and str(root).endswith((".tgz", ".tar.gz")):
            archive, root = Path(root), CACHE_DIR / "Seyerlehner"
            if not root.exists():
                with tarfile.open(archive) as t:
                    t.extractall(root, filter="data")
            dirs = [p for p in root.iterdir() if p.is_dir()]
            root = dirs[0] if len(dirs) == 1 else root
        self.root = _require_dir(root, "Seyerlehner", "--seyerlehner-dir", "the authors (K. Seyerlehner, JKU Linz)")

    def _labels(self, stem: str):
        from scipy.io import loadmat

        for name in (stem, stem.replace("_", " ")):
            path = self.root / f"{name}.wav.ref.label"
            if path.exists():
                mat = loadmat(path)
                return int(mat["fs"].ravel()[0]), mat["data"].ravel()
        raise FileNotFoundError(f"No annotation for {stem} in {self.root}")

    def items(self):
        items = []
        for stem, uem in read_split("seyerlehner").items():
            fs, frames = self._labels(stem)
            music, start = [], None
            for i, value in enumerate(list(frames) + [0]):
                if value == 1 and start is None:
                    start = i
                elif value != 1 and start is not None:
                    music.append((start / fs, i / fs))
                    start = None
            items.append(Item(stem, uem, music))
        return items

    def audio(self, uri):
        return _find_audio(self.root, uri, (".mp3", ".wav"))


DATASETS = {"inagvad": InaGVAD, "mirex2015": Mirex2015, "openbmat": OpenBMAT, "seyerlehner": Seyerlehner}


def add_dataset_arguments(parser):
    group = parser.add_argument_group("datasets")
    group.add_argument("--inagvad-dir", type=Path, help="InaGVAD audio files (searched recursively for <fileid>.wav)")
    group.add_argument("--inagvad-repo", type=Path, help="local clone of the InaGVAD repository (default: cloned automatically)")
    group.add_argument("--mirex2015-dir", type=Path, help="extracted Mirex2015 examples (default: downloaded automatically)")
    group.add_argument("--openbmat-dir", type=Path, help="OpenBMAT root directory (containing audio/ and annotations/)")
    group.add_argument("--seyerlehner-dir", type=Path, help="Seyerlehner dataset directory, or its original .tgz archive")


def load_dataset(name: str, args) -> Dataset:
    if name == "inagvad":
        return InaGVAD(args.inagvad_dir, args.inagvad_repo)
    if name == "mirex2015":
        return Mirex2015(args.mirex2015_dir)
    if name == "openbmat":
        return OpenBMAT(args.openbmat_dir)
    if name == "seyerlehner":
        return Seyerlehner(args.seyerlehner_dir)
    raise ValueError(f"Unknown dataset {name!r}. Available: {', '.join(DATASETS)}")
