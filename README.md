# inaVAMOS: Voice And Music Open Segmenter

Speech (voice activity) and music detection for audio and video files, based on
self-supervised (SSL) models from [INA](https://www.ina.fr).

`inaVAMOS` splits an audio stream into homogeneous segments labelled
`speech`, `music`, `speech+music` or `other`. It wraps two models published on the
HuggingFace Hub:

| Detector | Model | Task |
|----------|-------|------|
| `speech` | [ina-foss/ssl-vad-music2vec](https://huggingface.co/ina-foss/ssl-vad-music2vec) | Voice activity detection (VAD) |
| `music`  | [ina-foss/ssl-music-detection-music2vec](https://huggingface.co/ina-foss/ssl-music-detection-music2vec) | Music detection |

Both models use the CNN and the first transformer layer of the
[music2vec](https://huggingface.co/m-a-p/music2vec-v1) SSL encoder, followed by an MLP
classifier and Viterbi smoothing. They share the same encoder, which is computed only
once when both detectors are used.

### Output labels

The two detectors make independent decisions every 20 ms, which are combined into
a single timeline:

| speech | music | label |
|--------|-------|-------|
| yes | no  | `speech` |
| no  | yes | `music` |
| yes | yes | `speech+music` |
| no  | no  | `other` |

`other` covers everything that is neither speech nor music: silence, background
noise, applause, sound effects... When a single detector is run, `other` means "not
detected" by that detector: with `-d speech`, it is non-speech and can include
music; with `-d music`, it is non-music and can include speech.

The segments of each detector are also available separately (see
[Python usage](#python-usage), and the `json` and `textgrid` output formats).

## Installation

```bash
pip install inaVAMOS
```

The development version can be installed with
`pip install git+https://github.com/ina-foss/inaVAMOS.git`.

[ffmpeg](https://ffmpeg.org) is recommended to read any audio or video format
(`apt install ffmpeg`, `brew install ffmpeg` or `conda install ffmpeg`). Without it,
files are read with [librosa](https://librosa.org) (wav, flac, ogg, mp3…).

To use a GPU, install the [PyTorch build](https://pytorch.org/get-started/locally/)
matching your CUDA version before installing `inaVAMOS`.

The models (about 80 MB) are downloaded from the HuggingFace Hub on first use and
cached. Afterwards, `HF_HUB_OFFLINE=1` can be set to work offline.

## Command line usage

```bash
# Print the segmentation of a file as CSV
ina-vamos -i media.mp3

# Process several files and write one result per file in a directory
ina-vamos -i *.mp4 -o results/ -f textgrid

# Voice activity detection only, on a GPU, on the first 10 minutes
ina-vamos -i media.wav -d speech --device cuda --stop 600
```

Output example (CSV):

```
label,start,stop
music,0.000,23.900
other,23.900,32.060
...
speech,63.380,65.300
other,65.300,65.980
speech,65.980,68.200
```

Main options (see `ina-vamos --help`):

| Option | Description |
|--------|-------------|
| `-i` | input files, in any format readable by ffmpeg |
| `-o` | output directory (default: standard output). Existing results are skipped unless `--overwrite` is given, so interrupted batches can be resumed |
| `-f` | output format: `csv`, `tsv`, `json`, `textgrid` (Praat) or `audacity` (label track) |
| `-d` | detectors to run: `speech` and/or `music` (default: both) |
| `--device` | `cpu`, `cuda`, `cuda:1`… (default: GPU if available) |
| `--start`, `--stop` | process only a portion of the files (seconds) |

`python -m inavamos` is equivalent to `ina-vamos`.

## Python usage

```python
from inavamos import Segmenter

seg = Segmenter()  # or Segmenter(detectors=["speech"]), Segmenter(device="cuda")...

# Combined timeline: a list of (label, start, stop) tuples covering the whole file
for label, start, stop in seg("media.mp3"):
    print(label, start, stop)
```

`Segmenter.process` returns a `Result` object with more detailed outputs:

```python
result = seg.process("media.mp3", start=60, stop=120)

result.segments("speech")    # [Segment(label='speech', start=63.38, stop=65.3), ...]
result.segments("music")     # segments where music is detected
result.timeline()            # combined timeline, as returned by seg("media.mp3")
result.probabilities["music"]  # frame-level probabilities (numpy array, one frame every 20 ms)
result.decisions["music"]      # frame-level boolean decisions after Viterbi smoothing
result.frame_duration          # 0.02
```

Signals can also be given directly, as 1-D numpy arrays or torch tensors:

```python
import librosa

audio, sr = librosa.load("media.wav", sr=16000)
result = seg.process(audio, sampling_rate=sr)
```

Results can be exported with `inavamos.export.export(result, "textgrid")`.

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `detectors` | `("speech", "music")` | detectors to run |
| `device` | GPU if available | torch device |
| `window_duration` | `30.0` | audio is processed by overlapping windows of this duration (seconds), matching the slices used to train the models. `None` processes a signal at once, like the code published on the model cards; memory then grows quadratically with the duration |
| `context_duration` | `2.5` | overlap on each side of the windows (seconds) |
| `batch_size` | `8` | number of windows processed together |
| `transitions` | `{"speech": 0.99, "music": 0.95}` | Viterbi self-transition probabilities. Higher values produce fewer, longer segments |
| `token` | `None` | HuggingFace token, only needed for private repositories |
| `revision` | `None` | model revision on the Hub (branch, tag or commit), to pin a version |
| `repo_ids` | `None` | alternative model repositories or local directories, e.g. `{"speech": "/models/ssl-vad-music2vec"}` |

## Performance

About 30× faster than real time on a laptop CPU (an hour of audio in about 2 minutes),
and much faster on a GPU.

## Development

```bash
pip install -e ".[test]"
pytest                    # all tests (downloads the models on first run)
pytest -m "not models"    # unit tests only, without the models
```

The `models` tests check, among others, that the output is identical to the
reference code published on the model cards.

To release a new version: update `__version__` in `src/inavamos/__init__.py`, then
publish a GitHub release tagged `v<version>` (e.g. `v0.2.0`). The
[publish workflow](https://github.com/ina-foss/inaVAMOS/blob/main/.github/workflows/publish.yml) runs the tests, builds the package
and uploads it to PyPI.

## License and citation

inaVAMOS and the models it uses are distributed under the Pantagruel Research-only
License ([French version](https://github.com/ina-foss/inaVAMOS/blob/main/LICENSE.md), which prevails, and
[unofficial English translation](https://github.com/ina-foss/inaVAMOS/blob/main/LICENSE.en.md)). It restricts their use to
non-commercial research and development activities, by research organisations and
heritage institutions (libraries, museums, archives, audiovisual heritage). For any
other use, contact the Pantagruel Consortium at pantagruel-licence@univ-grenoble-alpes.fr.

If you use this tool or the models, please cite:

```bibtex
@inproceedings{pelloin2026lrec,
  author    = "Pelloin, Valentin and Bekkali, Lina and Dehak, Reda and Doukhan, David",
  year      = "2026",
  title     = "Data Selection Effects on Self-Supervised Learning of Audio Representations for French Audiovisual Broadcasts",
  booktitle = "Fifteenth International Conference on Language Resources and Evaluation (LREC 2026)",
  address   = "Palma, Mallorca, Spain",
  publisher = "European Language Resources Association",
}
```
