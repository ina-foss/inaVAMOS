import numpy as np
import pytest
import torch

from inavamos import segmenter as segmenter_module
from inavamos.models import DETECTORS
from inavamos.segmenter import Result, Segment, Segmenter

HOP = 320
KERNEL = 400


class FakeProjection(torch.nn.Module):
    """Returns twice its input, to tell projection features from encoder inputs."""

    def forward(self, x):
        return 2 * x, x


class FakeEncoder(torch.nn.Module):
    """Frame-wise encoder: each frame is the mean of the samples in its receptive field."""

    class config:
        conv_stride = [5, 2, 2, 2, 2, 2, 2]

    def __init__(self):
        super().__init__()
        self.feature_projection = FakeProjection()

    def _get_feat_extract_output_lengths(self, n):
        return (n - KERNEL) // HOP + 1

    def forward(self, x, output_hidden_states=True):
        frames = x.unfold(1, KERNEL, HOP).mean(-1)  # (batch, n_frames)
        self.feature_projection(frames[..., None].expand(-1, -1, 4))
        hidden = frames[..., None].expand(-1, -1, 4)

        class Output:
            hidden_states = (hidden, hidden)

        return Output()


def fake_segmenter(monkeypatch, window_duration, context_duration=2.5, batch_size=3, detectors=("music",)):
    monkeypatch.setattr(segmenter_module, "_normalize", lambda x: x)
    seg = Segmenter.__new__(Segmenter)
    seg.device = torch.device("cpu")
    seg.window_duration = window_duration
    seg.context_duration = context_duration
    seg.batch_size = batch_size
    seg.specs = {name: DETECTORS[name] for name in detectors}
    seg.encoders = {"enc": FakeEncoder()}
    seg.groups = {"enc": list(detectors)}
    # (batch, n_layers, n_frames, dim) -> (batch, n_frames): the first layer only.
    seg.classifiers = {name: lambda f: f[:, 0, :, 0] for name in detectors}
    return seg


def test_first_layer_features(monkeypatch):
    """The VAD classifier gets the CNN projection, the music classifier the encoder input."""
    audio = np.random.default_rng(0).uniform(-3, 3, 16000 * 40).astype(np.float32)
    result = fake_segmenter(monkeypatch, 30.0, detectors=("speech", "music")).process_audio(audio)
    logits = {
        name: np.log(p / (1 - p)) * (-1 if DETECTORS[name].invert else 1) for name, p in result.probabilities.items()
    }
    np.testing.assert_allclose(logits["speech"], 2 * logits["music"], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("n_samples", [16000 * 100, 16000 * 100 + 123, 16000 * 61, 16000 * 30, 16000 * 12, 16000])
@pytest.mark.parametrize("window_duration,context_duration", [(30.0, 2.5), (10.0, 0.0), (5.0, 2.0)])
def test_windowed_frames_match_whole_signal(monkeypatch, n_samples, window_duration, context_duration):
    rng = np.random.default_rng(0)
    audio = rng.uniform(-10, 10, n_samples).astype(np.float32)
    whole = fake_segmenter(monkeypatch, None).process_audio(audio)
    windowed = fake_segmenter(monkeypatch, window_duration, context_duration).process_audio(audio)
    assert len(windowed.probabilities["music"]) == FakeEncoder()._get_feat_extract_output_lengths(n_samples)
    np.testing.assert_allclose(windowed.probabilities["music"], whole.probabilities["music"], atol=1e-6)


@pytest.mark.parametrize("n_samples", [16000 * 100 + 123, 16000 * 31, 16000 * 30 + 1])
def test_windows(monkeypatch, n_samples):
    seg = fake_segmenter(monkeypatch, 30.0, 2.5)
    windows = seg._windows(n_samples, HOP)
    cores = [(a, b) for _, _, a, b in windows]
    assert cores[0][0] == 0 and cores[-1][1] == n_samples
    assert all(b == a2 for (_, b), (a2, _) in zip(cores[:-1], cores[1:]))
    for w0, w1, a, b in windows:
        assert w0 % HOP == 0 and a % HOP == 0
        assert 0 <= w0 <= a < b <= w1 <= n_samples
        assert 30 * 16000 <= w1 - w0 < 30 * 16000 + HOP


def test_too_short(monkeypatch):
    result = fake_segmenter(monkeypatch, 30.0).process_audio(np.zeros(100, dtype=np.float32))
    assert result.timeline() == [Segment("other", 0.0, 100 / 16000)]


def make_result(speech, music, duration=None, offset=0.0):
    speech, music = np.array(speech, dtype=bool), np.array(music, dtype=bool)
    return Result(
        duration=duration if duration is not None else len(speech) * 0.02,
        frame_duration=0.02,
        offset=offset,
        decisions={"speech": speech, "music": music},
    )


def test_timeline():
    result = make_result([0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0], duration=0.13)
    assert result.timeline() == [
        ("other", 0.0, 0.02),
        ("speech", 0.02, 0.04),
        ("speech+music", 0.04, 0.08),
        ("music", 0.08, 0.1),
        ("other", 0.1, 0.13),  # the last segment is extended to the end of the signal
    ]
    assert result.segments("speech") == [("speech", 0.02, 0.08)]
    assert result.segments("vad") == [("speech", 0.02, 0.08)]
    assert result.segments("music", inactive_label="") == [("", 0.0, 0.04), ("music", 0.04, 0.1), ("", 0.1, 0.13)]


def test_timeline_offset():
    result = make_result([1, 1, 0], [0, 0, 0], offset=10.0)
    assert result.timeline() == [("speech", 10.0, 10.04), ("other", 10.04, 10.06)]


def test_viterbi_smoothing_removes_short_segments():
    prob = np.full(200, 0.1)
    prob[100:102] = 0.9  # 40 ms spike
    prob[150:] = 0.9
    decisions = segmenter_module.viterbi_smoothing(prob, 0.99)
    assert not decisions[:150].any() and decisions[150:].all()
