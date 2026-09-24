import json

import numpy as np
import pytest

from inavamos.export import FORMATS, export
from inavamos.segmenter import Result


@pytest.fixture
def result():
    return Result(
        duration=0.13,
        frame_duration=0.02,
        decisions={
            "speech": np.array([0, 1, 1, 1, 0, 0], dtype=bool),
            "music": np.array([0, 0, 0, 0, 0, 0], dtype=bool),
        },
    )


def test_csv(result):
    assert export(result, "csv") == "label,start,stop\nother,0.000,0.020\nspeech,0.020,0.080\nother,0.080,0.130\n"
    assert export(result, "tsv").splitlines()[1] == "other\t0.000\t0.020"


def test_json(result):
    data = json.loads(export(result, "json", file="a.wav"))
    assert data["file"] == "a.wav"
    assert data["detectors"] == {"speech": [{"start": 0.02, "stop": 0.08}], "music": []}
    assert data["timeline"][1] == {"label": "speech", "start": 0.02, "stop": 0.08}


def test_textgrid(result):
    text = export(result, "textgrid")
    assert "size = 3" in text
    assert 'name = "speech"' in text and 'name = "music"' in text and 'name = "timeline"' in text
    # The music tier has no activity: a single empty interval covering the signal.
    music_tier = text.split('name = "music"')[1].split("item [")[0]
    assert "intervals: size = 1" in music_tier


def test_audacity(result):
    assert export(result, "audacity").splitlines()[1] == "0.020000\t0.080000\tspeech"


def test_all_formats(result):
    for fmt in FORMATS:
        assert export(result, fmt)
