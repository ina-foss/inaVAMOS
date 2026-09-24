"""Loading of the SSL encoders and downstream classifiers published on the HuggingFace Hub.

The models are loaded directly (without ``trust_remote_code``): the encoder is a
standard ``Data2VecAudioModel`` and the classifiers are small MLPs (:class:`FrameClassifier`)
stored as safetensors, or as ``torch.export`` programs in older revisions of the models.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorSpec:
    """Description of a detector published on the HuggingFace Hub."""

    name: str
    repo_id: str
    # Viterbi self-transition probability used for smoothing.
    transition: float
    # The VAD classifier outputs the probability of *non*-speech.
    invert: bool
    # Input layout of the torch.export classifier (classifier/model.pt2), if used:
    # "frames": (n_frames, n_layers, dim), "batch": (batch, n_layers, n_frames, dim).
    layout: str


DETECTORS = {
    "speech": DetectorSpec(
        name="speech",
        repo_id="ina-foss/ssl-vad-music2vec",
        transition=0.99,
        invert=True,
        layout="frames",
    ),
    "music": DetectorSpec(
        name="music",
        repo_id="ina-foss/ssl-music-detection-music2vec",
        transition=0.95,
        invert=False,
        layout="batch",
    ),
}

DETECTOR_ALIASES = {"vad": "speech", "voice": "speech", "md": "music"}


def resolve_detector(name: str) -> str:
    key = name.lower()
    key = DETECTOR_ALIASES.get(key, key)
    if key not in DETECTORS:
        raise ValueError(f"Unknown detector {name!r}. Available: {', '.join(DETECTORS)}")
    return key


class ModelAccessError(RuntimeError):
    pass


def _access_message(repo_id: str) -> str:
    return (
        f"Cannot access the model '{repo_id}' on the HuggingFace Hub. Check the repository id,\n"
        "and provide a token (HF_TOKEN, `hf auth login` or --token) if the repository is private."
    )


def fetch_file(repo_id: str, filename: str, token=None, revision=None) -> str:
    """Return a local path to ``filename`` from a Hub repository or a local directory."""
    if os.path.isdir(repo_id):
        path = os.path.join(repo_id, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        return hf_hub_download(repo_id, filename, token=token, revision=revision)
    except (GatedRepoError, RepositoryNotFoundError) as e:
        raise ModelAccessError(_access_message(repo_id)) from e


def load_encoder(repo_id: str, token=None, revision=None, device="cpu"):
    """Load the SSL encoder stored in the ``pretrained/`` subfolder of a repository."""
    from transformers import Data2VecAudioConfig, Data2VecAudioModel

    with open(fetch_file(repo_id, "pretrained/config.json", token, revision)) as f:
        config_dict = json.load(f)
    # The published configuration stores a path in `vocab_size`, which recent
    # versions of transformers reject. It is unused by the audio encoder.
    if not isinstance(config_dict.get("vocab_size"), int):
        config_dict.pop("vocab_size", None)
    config = Data2VecAudioConfig(**config_dict)

    weights = fetch_file(repo_id, "pretrained/model.safetensors", token, revision)
    with _quiet_transformers():
        model = Data2VecAudioModel.from_pretrained(os.path.dirname(weights), config=config)
    return model.to(device).eval()


def encoder_fingerprint(repo_id: str, token=None, revision=None) -> str:
    """Identify an encoder by its weights so that identical encoders are only run once.

    Files in the HuggingFace cache are symlinks to blobs named after their sha256.
    """
    weights = fetch_file(repo_id, "pretrained/model.safetensors", token, revision)
    if os.path.islink(weights):
        return os.path.basename(os.path.realpath(weights))
    return os.path.abspath(weights)


@contextlib.contextmanager
def _quiet_transformers():
    from transformers.utils import logging as hf_logging

    enabled = hf_logging.is_progress_bar_enabled()
    verbosity = hf_logging.get_verbosity()
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(verbosity)
        if enabled:
            hf_logging.enable_progress_bar()


@contextlib.contextmanager
def _quiet_torch_export():
    # The classifiers were serialized with an older torch.export format, which
    # makes recent torch versions log a deprecation traceback on load.
    torch_logger = logging.getLogger("torch.export")
    level = torch_logger.level
    torch_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        torch_logger.setLevel(level)


class FrameClassifier(torch.nn.Module):
    """MLP applied to each frame, on the concatenation of the encoder layers.

    Maps features of shape (batch, n_layers, n_frames, dim) to logits of shape (batch, n_frames).
    """

    def __init__(self, input_size: int, hidden_sizes: list, batch_norm_eps: float = 1e-5):
        super().__init__()
        layers = []
        size = input_size
        for hidden_size in hidden_sizes:
            layers += [
                torch.nn.Linear(size, hidden_size),
                torch.nn.BatchNorm1d(hidden_size, eps=batch_norm_eps),
                torch.nn.ReLU(),
            ]
            size = hidden_size
        self.hidden = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, n_frames, _ = features.shape
        x = features.transpose(1, 2).reshape(batch * n_frames, -1)
        return self.out(self.hidden(x)).reshape(batch, n_frames)


class ExportedClassifier(torch.nn.Module):
    """Adapter giving a ``torch.export`` classifier the interface of :class:`FrameClassifier`."""

    def __init__(self, module, layout: str):
        super().__init__()
        self.module = module
        self.layout = layout

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.layout == "frames":
            return torch.stack([self.module(f.transpose(0, 1)) for f in features])
        return self.module(features)[0]


def load_classifier(spec: DetectorSpec, token=None, revision=None, device="cpu") -> torch.nn.Module:
    """Load the classifier of a detector: ``classifier/model.safetensors`` if available,
    otherwise the ``torch.export`` program ``classifier/model.pt2``."""
    from huggingface_hub.errors import EntryNotFoundError

    try:
        config_path = fetch_file(spec.repo_id, "classifier/config.json", token, revision)
        weights_path = fetch_file(spec.repo_id, "classifier/model.safetensors", token, revision)
    except (EntryNotFoundError, FileNotFoundError):
        path = fetch_file(spec.repo_id, "classifier/model.pt2", token, revision)
        with _quiet_torch_export():
            program = torch.export.load(path)
        return ExportedClassifier(program.module(), spec.layout).to(device)

    from safetensors.torch import load_file

    with open(config_path) as f:
        config = json.load(f)
    classifier = FrameClassifier(config["input_size"], config["hidden_sizes"], config.get("batch_norm_eps", 1e-5))
    classifier.load_state_dict(load_file(weights_path))
    return classifier.to(device).eval()
