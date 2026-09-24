"""Tests running the real models. They are skipped when the models cannot be downloaded."""

import numpy as np
import pytest

from inavamos import ModelAccessError, Segmenter

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def segmenter():
    try:
        return Segmenter(device="cpu")
    except ModelAccessError as e:
        pytest.skip(str(e))


def test_shared_encoder(segmenter):
    assert len(segmenter.encoders) == 1


@pytest.mark.parametrize("duration", [3.0, 70.0])
def test_process_signal(segmenter, duration):
    rng = np.random.default_rng(0)
    audio = 0.01 * rng.standard_normal(int(duration * 16000)).astype(np.float32)
    result = segmenter.process(audio)
    n_frames = len(result.probabilities["speech"])
    assert abs(n_frames * result.frame_duration - duration) < 0.05
    for name in ("speech", "music"):
        probs = result.probabilities[name]
        assert probs.shape == (n_frames,) and np.all((probs >= 0) & (probs <= 1))
    timeline = result.timeline()
    assert timeline[0].start == 0.0 and timeline[-1].stop == pytest.approx(duration)
    assert all(a.stop == b.start for a, b in zip(timeline[:-1], timeline[1:]))


def test_resampling(segmenter):
    audio = np.zeros(8000 * 2, dtype=np.float32)
    result = segmenter.process(audio, sampling_rate=8000)
    assert result.duration == pytest.approx(2.0)


MUSANMIX_URL = "https://github.com/ina-foss/inaSpeechSegmenter/raw/master/media/musanmix.mp3"


@pytest.fixture(scope="module")
def musanmix(pytestconfig):
    """74 s mix of music, speech and noise from MUSAN, used by inaSpeechSegmenter (cached)."""
    import urllib.request

    path = pytestconfig.cache.mkdir("inavamos") / "musanmix.mp3"
    if not path.exists():
        try:
            urllib.request.urlretrieve(MUSANMIX_URL, path)
        except OSError as e:
            pytest.skip(f"Cannot download {MUSANMIX_URL}: {e}")
    return path


@pytest.mark.parametrize("name", ["speech", "music"])
def test_same_output_as_huggingface_model(segmenter, musanmix, name):
    """Whole-file processing gives the segments of the reference code published on the model cards."""
    import librosa
    from transformers import AutoModel

    audio, sr = librosa.load(musanmix, sr=16000)
    reference = AutoModel.from_pretrained(segmenter.specs[name].repo_id, trust_remote_code=True)
    expected = [(s["start"], s["stop"]) for s in reference(audio=audio, sampling_rate=sr) if s["label"]]

    window_duration = segmenter.window_duration
    segmenter.window_duration = None
    try:
        actual = [(s.start, s.stop) for s in segmenter.process(audio).segments(name)]
    finally:
        segmenter.window_duration = window_duration

    assert len(actual) == len(expected) > 0
    # The reference code computes times as frame_index * n_frames / duration, and ends the
    # last segment one frame early: boundaries may differ by about one frame (20 ms).
    np.testing.assert_allclose(actual, expected, atol=0.03)
