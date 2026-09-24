"""inaVAMOS: Voice And Music Open Segmenter. Speech (voice activity) and music detection based on SSL models from INA."""

from .audio import load_audio
from .models import DETECTORS, ModelAccessError
from .segmenter import Result, Segment, Segmenter

__version__ = "0.1.0"

__all__ = ["Segmenter", "Segment", "Result", "load_audio", "DETECTORS", "ModelAccessError", "__version__"]
