"""Speech (VAD) and music segmentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import numpy as np
import torch

from . import models
from .audio import SAMPLING_RATE, load_audio

logger = logging.getLogger(__name__)

OTHER_LABEL = "other"


class Segment(NamedTuple):
    """A labelled time interval, in seconds. Compatible with ``(label, start, stop)`` tuples."""

    label: str
    start: float
    stop: float


@dataclass
class Result:
    """Frame-level output of the detectors for one audio signal.

    Attributes:
        duration: duration of the processed audio, in seconds.
        frame_duration: duration of one frame (the encoder stride), in seconds.
        offset: time of the first frame, in seconds (non-zero when processing a portion of a file).
        probabilities: per-detector frame-level probabilities of activity, before smoothing.
        decisions: per-detector frame-level boolean decisions, after Viterbi smoothing.
    """

    duration: float
    frame_duration: float
    offset: float = 0.0
    probabilities: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)

    @property
    def detectors(self) -> list[str]:
        return list(self.decisions)

    def _runs(self, values: np.ndarray):
        """Yield ``(value, start, stop)`` times for each run of identical frame values."""
        n = len(values)
        if n == 0:
            return
        change = np.flatnonzero(values[1:] != values[:-1]) + 1
        bounds = np.concatenate([[0], change, [n]])
        end = self.offset + self.duration
        for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            start = self.offset + a * self.frame_duration
            # The last frame is extended to the end of the audio.
            stop = end if i == len(bounds) - 2 else self.offset + b * self.frame_duration
            yield values[a], round(float(start), 6), round(float(min(stop, end)), 6)

    def segments(self, detector: str, inactive_label: str | None = None) -> list[Segment]:
        """Segments where ``detector`` (e.g. "speech" or "music") is active.

        If ``inactive_label`` is given, segments where the detector is inactive are also
        returned with this label, so that the segments cover the whole signal.
        """
        detector = models.resolve_detector(detector)
        return [
            Segment(detector if v else inactive_label, a, b)
            for v, a, b in self._runs(self.decisions[detector])
            if v or inactive_label is not None
        ]

    def timeline(self) -> list[Segment]:
        """Contiguous segmentation of the whole signal combining all detectors.

        Labels are the names of the active detectors joined by "+" (e.g. "speech",
        "music", "speech+music"), or "other" when no detector is active.
        """
        if not self.decisions:
            return []
        names = self.detectors
        # Encode the combination of active detectors as an integer per frame.
        codes = sum(self.decisions[name].astype(np.int64) << i for i, name in enumerate(names))
        timeline = []
        for code, a, b in self._runs(codes):
            active = [name for i, name in enumerate(names) if code >> i & 1]
            timeline.append(Segment("+".join(active) or OTHER_LABEL, a, b))
        if not timeline and self.duration > 0:
            timeline.append(Segment(OTHER_LABEL, self.offset, self.offset + self.duration))
        return timeline

    def to_dict(self) -> dict:
        return {
            "duration": self.duration,
            "offset": self.offset,
            "detectors": {
                name: [{"start": s.start, "stop": s.stop} for s in self.segments(name)] for name in self.detectors
            },
            "timeline": [s._asdict() for s in self.timeline()],
        }


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Zero mean and unit variance normalization, as done by the models' feature extractor."""
    return (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + 1e-7)


def viterbi_smoothing(prob: np.ndarray, transition: float) -> np.ndarray:
    """Binary Viterbi decoding of frame probabilities with a symmetric transition matrix."""
    import librosa

    if len(prob) == 0:
        return np.zeros(0, dtype=bool)
    matrix = np.array([[transition, 1 - transition], [1 - transition, transition]])
    return librosa.sequence.viterbi_binary(prob[None, :], matrix)[0].astype(bool)


class Segmenter:
    """Speech and music segmentation using SSL-based models from INA.

    Example:
        >>> seg = Segmenter()
        >>> seg("media.mp3")
        [Segment(label='music', start=0.0, stop=22.74), ...]

    Args:
        detectors: detectors to run, among "speech" (voice activity detection) and "music".
        device: torch device. Defaults to CUDA when available.
        window_duration: signals are processed by overlapping windows of this duration
            (seconds), matching the 30 s slices used to train the models, and bounding memory
            usage. ``None`` processes the whole signal at once, like the reference code
            published with the models (memory grows quadratically with duration).
        context_duration: overlap (seconds) on each side of a window whose predictions are
            discarded, in favour of the neighbouring window.
        batch_size: number of windows processed together.
        transitions: override the Viterbi self-transition probability per detector
            (e.g. ``{"music": 0.99}``). Higher values produce fewer, longer segments.
        token: HuggingFace token, only needed for private model repositories.
        revision: model revision (branch, tag or commit) on the HuggingFace Hub.
        repo_ids: override the model repositories (Hub ids or local directories) per detector.
    """

    def __init__(
        self,
        detectors: Sequence[str] = ("speech", "music"),
        device: str | torch.device | None = None,
        window_duration: float | None = 30.0,
        context_duration: float = 2.5,
        batch_size: int = 8,
        transitions: dict | None = None,
        token: str | None = None,
        revision: str | None = None,
        repo_ids: dict | None = None,
    ):
        if isinstance(detectors, str):
            detectors = [detectors]
        names = list(dict.fromkeys(models.resolve_detector(d) for d in detectors))
        if not names:
            raise ValueError("At least one detector is required")
        if context_duration < 0:
            raise ValueError("context_duration must be non-negative")
        if window_duration is not None and window_duration <= 2 * context_duration:
            raise ValueError("window_duration must be greater than twice context_duration")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.window_duration = window_duration
        self.context_duration = context_duration
        self.batch_size = max(1, int(batch_size))

        transitions = {models.resolve_detector(k): v for k, v in (transitions or {}).items()}
        repo_ids = {models.resolve_detector(k): v for k, v in (repo_ids or {}).items()}
        self.specs = {}
        for name in names:
            spec = models.DETECTORS[name]
            self.specs[name] = models.DetectorSpec(
                name=name,
                repo_id=repo_ids.get(name, spec.repo_id),
                transition=transitions.get(name, spec.transition),
                invert=spec.invert,
                first_layer=spec.first_layer,
                layout=spec.layout,
            )

        # Detectors sharing the same encoder weights share the encoder forward pass.
        self.encoders = {}  # fingerprint -> encoder
        self.groups = {}  # fingerprint -> detector names
        self.classifiers = {}
        for name, spec in self.specs.items():
            logger.info("Loading %s detector from %s", name, spec.repo_id)
            key = models.encoder_fingerprint(spec.repo_id, token, revision)
            if key not in self.encoders:
                self.encoders[key] = models.load_encoder(spec.repo_id, token, revision, self.device)
                self.groups[key] = []
            self.groups[key].append(name)
            self.classifiers[name] = models.load_classifier(spec, token, revision, self.device)

    @property
    def detectors(self) -> list[str]:
        return list(self.specs)

    def __call__(self, media, start: float | None = None, stop: float | None = None) -> list[Segment]:
        """Segment a media file (or a 16 kHz signal) and return the combined timeline.

        Returns a list of ``(label, start, stop)`` segments covering the whole signal, with
        labels "speech", "music", "speech+music" or "other".
        """
        return self.process(media, start=start, stop=stop).timeline()

    def process(
        self, media, start: float | None = None, stop: float | None = None, sampling_rate: int = SAMPLING_RATE
    ) -> Result:
        """Run the detectors and return a :class:`Result`.

        Args:
            media: path to any audio/video file readable by ffmpeg, or a mono signal
                (numpy array or torch tensor) sampled at ``sampling_rate``.
            start: optional start time in seconds (files only).
            stop: optional stop time in seconds (files only).
            sampling_rate: sampling rate of ``media`` when a signal is given.
        """
        if isinstance(media, (np.ndarray, torch.Tensor)):
            if start is not None or stop is not None:
                raise ValueError("start/stop are only supported for media files; slice the signal instead")
            audio = media.detach().cpu().numpy() if isinstance(media, torch.Tensor) else media
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim != 1:
                raise ValueError(f"Expected a mono signal (1-D array), got shape {audio.shape}")
            if sampling_rate != SAMPLING_RATE:
                import librosa

                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=SAMPLING_RATE)
        else:
            audio = load_audio(media, start=start, stop=stop)
        return self.process_audio(audio, offset=start or 0.0)

    @torch.inference_mode()
    def process_audio(self, audio: np.ndarray, offset: float = 0.0) -> Result:
        """Run the detectors on a mono float32 signal sampled at 16 kHz."""
        audio = np.asarray(audio, dtype=np.float32)
        frame_duration = None
        probabilities = {}
        for key, names in self.groups.items():
            encoder = self.encoders[key]
            hop = int(np.prod(encoder.config.conv_stride))
            frame_duration = hop / SAMPLING_RATE
            probabilities.update(self._frame_probabilities(encoder, names, audio, hop))

        decisions = {
            name: viterbi_smoothing(probabilities[name], self.specs[name].transition) for name in self.detectors
        }
        return Result(
            duration=len(audio) / SAMPLING_RATE,
            frame_duration=frame_duration,
            offset=offset,
            probabilities=probabilities,
            decisions=decisions,
        )

    def _windows(self, n_samples: int, hop: int):
        """Split a signal into overlapping windows aligned on the encoder stride.

        Returns ``(window_start, window_stop, core_start, core_stop)`` tuples, in samples.
        Frames are only kept for the core of each window, the rest being context.
        """
        window = n_samples if self.window_duration is None else round(self.window_duration * SAMPLING_RATE / hop) * hop
        if n_samples <= window:
            return [(0, n_samples, 0, n_samples)]
        # The CNN receptive field exceeds its stride: at least one hop of context is
        # needed on the right to compute the last frame of a core.
        context = max(hop, round(self.context_duration * SAMPLING_RATE / hop) * hop)
        core = max(hop, window - 2 * context)
        windows = []
        for a in range(0, n_samples, core):
            b = min(a + core, n_samples)
            w0 = max(0, a - context)
            w1 = w0 + window
            if w1 > n_samples:
                # Last window: keep a full-length window by extending it to the left.
                w1 = n_samples
                w0 = (n_samples - window) // hop * hop
            windows.append((w0, w1, a, b))
        return windows

    def _frame_probabilities(self, encoder, names, audio, hop) -> dict:
        n_frames = int(encoder._get_feat_extract_output_lengths(len(audio))) if len(audio) else 0
        if n_frames < 2:
            logger.warning("Audio too short to be processed (%.3fs)", len(audio) / SAMPLING_RATE)
            return {name: np.zeros(0, dtype=np.float32) for name in names}
        probs = {name: np.zeros(n_frames, dtype=np.float32) for name in names}

        # Capture the output of the CNN feature projection during the encoder forward pass.
        projection = []
        hook = encoder.feature_projection.register_forward_hook(lambda module, args, output: projection.append(output[0]))
        try:
            return self._windowed_probabilities(encoder, names, audio, hop, n_frames, probs, projection)
        finally:
            hook.remove()

    def _windowed_probabilities(self, encoder, names, audio, hop, n_frames, probs, projection) -> dict:
        signal = torch.from_numpy(audio)
        windows = self._windows(len(audio), hop)
        # Batch together windows of identical length.
        by_length = {}
        for w in windows:
            by_length.setdefault(w[1] - w[0], []).append(w)
        for group in by_length.values():
            for i in range(0, len(group), self.batch_size):
                batch = group[i : i + self.batch_size]
                inputs = torch.stack([_normalize(signal[w0:w1]) for w0, w1, _, _ in batch]).to(self.device)
                projection.clear()
                hidden = encoder(inputs, output_hidden_states=True).hidden_states
                # (batch, n_layers, n_frames, dim): first layer + transformer layer outputs.
                features = {
                    "encoder_input": torch.stack(hidden, dim=1),
                    "projection": torch.stack([projection[0], *hidden[1:]], dim=1),
                }
                for name in names:
                    batch_probs = self._classify(name, features[self.specs[name].first_layer])
                    for (w0, _, a, b), p in zip(batch, batch_probs):
                        first, last = a // hop, min(b // hop, n_frames)
                        local = (a - w0) // hop
                        probs[name][first:last] = p[local : local + last - first]
        return probs

    def _classify(self, name: str, features: torch.Tensor) -> np.ndarray:
        logits = self.classifiers[name](features)
        probs = torch.sigmoid(logits.float())
        if self.specs[name].invert:
            probs = 1 - probs
        return probs.cpu().numpy()
