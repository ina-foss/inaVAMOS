# Benchmarks

Scripts to evaluate inaVAMOS, [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter)
and [pyannote](https://github.com/pyannote/pyannote-audio) on voice activity detection
(VAD) and music detection.

| Task | Dataset | Test data | Metric |
|------|---------|-----------|--------|
| VAD | [InaGVAD](https://github.com/ina-foss/InaGVAD) | test set: 217 files, 3h37 | InaGVAD evaluation code: accuracy, precision, recall, F1 with a 0.3 s collar |
| Music | [Mirex2015](https://www.music-ir.org/mirex/wiki/2015:Music/Speech_Classification_and_Detection) | 5 regions of 4 files, 0.64 h | frame-level F1 of the music class, no collar |
| Music | [OpenBMAT](https://zenodo.org/records/3381249) | 247 files, 4.12 h | idem |
| Music | [Seyerlehner](https://www.cp.jku.at/research/papers/Seyerlehner_etal_DAFx_2007.pdf) | 7 regions of 5 files, 1.00 h | idem |

The music detection model of inaVAMOS was trained on the training subsets of Mirex2015,
OpenBMAT and Seyerlehner: only their test subsets are evaluated. These subsets are
listed in [`splits/`](splits) (file, start and stop of each test region, in seconds).
They were extracted from the data used for training with
[`extract_splits.py`](extract_splits.py).

Music metrics are computed on 10 ms frames, pooled over all the test regions of a
dataset. The *global* music F1 is the average of the F1 of the three datasets.

## Datasets

| Dataset | Download | Option |
|---------|----------|--------|
| InaGVAD | Audio on request on [INA's website](https://www.ina.fr/institut-national-audiovisuel/research/dataset-project). Annotations and evaluation code: cloned automatically from GitHub | `--inagvad-dir` (audio, searched recursively for `<fileid>.wav`), `--inagvad-repo` (optional local clone) |
| Mirex2015 | Downloaded automatically ([zip](http://mirg.city.ac.uk/datasets/muspeak/muspeak-mirex2015-detection-examples.zip), 329 MB) | `--mirex2015-dir` (optional, extracted zip) |
| OpenBMAT | Restricted access on [Zenodo](https://zenodo.org/records/3381249) | `--openbmat-dir` (directory containing `audio/` and `annotations/`) |
| Seyerlehner | Not publicly available: contact the authors | `--seyerlehner-dir` (original directory of `.mp3` and `.wav.ref.label` files, or the original `.tgz` archive) |

Datasets downloaded automatically are cached in `~/.cache/inavamos-benchmarks`
(or `$INAVAMOS_BENCHMARK_DATA`).

References are built from the original annotations:

- **Mirex2015**: `m` segments of the annotation files (`_v2` versions when available).
- **OpenBMAT**: music is present where at least one of the three annotators marked any
  music class (music, foreground, similar, background or very low background music), as
  in the training data of inaVAMOS.
- **Seyerlehner**: frames labelled 1 (music) in the `.wav.ref.label` files.

## Environments

The systems have incompatible dependencies: each one needs its own environment. For
instance with [uv](https://docs.astral.sh/uv/):

```bash
cd benchmarks
uv venv -p 3.10 .envs/inavamos           && uv pip install -p .envs/inavamos -r requirements/inavamos.txt
uv venv -p 3.11 .envs/inaspeechsegmenter && uv pip install -p .envs/inaspeechsegmenter -r requirements/inaspeechsegmenter.txt
uv venv -p 3.12 .envs/pyannote           && uv pip install -p .envs/pyannote -r requirements/pyannote.txt
uv venv -p 3.10 .envs/pyannote-legacy    && uv pip install -p .envs/pyannote-legacy -r requirements/pyannote-legacy.txt --index-strategy unsafe-best-match
uv venv -p 3.12 .envs/eval               && uv pip install -p .envs/eval -r requirements/eval.txt
```

To use a GPU, install the CUDA builds of the frameworks: torch from the
[PyTorch index](https://pytorch.org/get-started/locally/) matching your CUDA version
(for `pyannote-legacy`, replace the `cpu` index of its requirements by `cu117`), and
`tensorflow[and-cuda]` for inaSpeechSegmenter. Then pass `--device cuda` to `predict.py`
for inaVAMOS and pyannote (TensorFlow uses the GPU automatically).

ffmpeg must be installed. The pyannote models are gated: accept their conditions on the
HuggingFace Hub ([segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) for
`pyannote`; [voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection)
and [segmentation](https://huggingface.co/pyannote/segmentation) for `pyannote-legacy`),
and authenticate with `hf auth login` or `--token`.

## Running

Compute the predictions of each system (`predict.py SYSTEM DATASET...`). They are saved
in `predictions/<system>/<dataset>/`, and interrupted runs can be resumed:

```bash
DATA="--inagvad-dir /data/inaGVAD --openbmat-dir /data/OpenBMAT --seyerlehner-dir /data/Seyerlehner"

.envs/inavamos/bin/python predict.py inavamos inagvad mirex2015 openbmat seyerlehner $DATA
.envs/inaspeechsegmenter/bin/python predict.py inaspeechsegmenter inagvad mirex2015 openbmat seyerlehner $DATA
.envs/pyannote/bin/python predict.py pyannote inagvad $DATA
.envs/pyannote-legacy/bin/python predict.py pyannote-legacy inagvad $DATA
```

pyannote only detects speech: it is only evaluated on InaGVAD. Systems are used with their
default settings; pyannote (segmentation-3.0) uses the VAD hyper-parameters of its model card.

Then evaluate all the predictions and print the result tables:

```bash
.envs/eval/bin/python evaluate.py $DATA --output results.json
```

Speech is `speech`, `male` or `female` segments, and music is `music` segments. Labels
combining several classes, like `speech+music` for inaVAMOS, count for each class.
inaSpeechSegmenter predicts exclusive classes: speech over music is labelled as speech
only, which lowers its music recall on datasets with background music (OpenBMAT,
Seyerlehner).
