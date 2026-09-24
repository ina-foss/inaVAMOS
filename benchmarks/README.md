# Benchmarks

Scripts to evaluate inaVAMOS and other freely available systems on voice activity
detection (VAD) and music detection:

| System | Task | Setting |
|--------|------|---------|
| inaVAMOS | VAD, music | default settings |
| [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter) | VAD, music | default settings |
| pyannote ([segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)) | VAD | VAD hyper-parameters of the model card |
| pyannote 2.1 ([voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection)) | VAD | default settings, as in the InaGVAD paper |
| [Silero VAD](https://github.com/snakers4/silero-vad) | VAD | default settings |
| [PANNs](https://github.com/qiuqiangkong/panns_inference) (CNN14, AudioSet) | music | "Music" class, threshold tuned on dev |
| [YAMNet](https://www.kaggle.com/models/google/yamnet) (AudioSet) | music | "Music" class, threshold tuned on dev |

| Task | Dataset | Test data | Metric |
|------|---------|-----------|--------|
| VAD | [InaGVAD](https://github.com/ina-foss/InaGVAD) | test set: 217 files, 3h37 | InaGVAD evaluation code: accuracy, precision, recall, F1 with a 0.3 s collar |
| Music | [Mirex2015](https://www.music-ir.org/mirex/wiki/2015:Music/Speech_Classification_and_Detection) | 5 regions of 4 files, 0.64 h | frame-level F1 of the music class, no collar |
| Music | [OpenBMAT](https://zenodo.org/records/3381249) | 247 files, 4.12 h | idem |
| Music | [Seyerlehner](https://www.cp.jku.at/research/papers/Seyerlehner_etal_DAFx_2007.pdf) | 7 regions of 5 files, 1.00 h | idem |

The music detection model of inaVAMOS was trained on the training subsets of Mirex2015,
OpenBMAT and Seyerlehner: only their test subsets are evaluated. These subsets are
listed in [`splits/`](splits) (file, start and stop of each test region, in seconds),
together with the dev subsets used to tune thresholds. They were extracted from the data
used for training with [`extract_splits.py`](extract_splits.py).

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
uv venv -p 3.12 .envs/silero             && uv pip install -p .envs/silero -r requirements/silero.txt
uv venv -p 3.12 .envs/panns              && uv pip install -p .envs/panns -r requirements/panns.txt
uv venv -p 3.11 .envs/yamnet             && uv pip install -p .envs/yamnet -r requirements/yamnet.txt
uv venv -p 3.12 .envs/eval               && uv pip install -p .envs/eval -r requirements/eval.txt
```

To use a GPU, install the CUDA builds of the frameworks: torch from the
[PyTorch index](https://pytorch.org/get-started/locally/) matching your CUDA version
(for `pyannote-legacy`, replace the `cpu` index of its requirements by `cu117`), and
`tensorflow[and-cuda]` for inaSpeechSegmenter and YAMNet. Then pass `--device cuda` to
`predict.py` for inaVAMOS, pyannote and PANNs (TensorFlow uses the GPU automatically).
Silero VAD is designed for CPU and always runs on CPU.

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
.envs/silero/bin/python predict.py silero inagvad $DATA
.envs/panns/bin/python predict.py panns mirex2015 openbmat seyerlehner $DATA
.envs/yamnet/bin/python predict.py yamnet mirex2015 openbmat seyerlehner $DATA
```

pyannote and Silero only detect speech: they are only evaluated on InaGVAD. PANNs and
YAMNet only detect music: they are only evaluated on the music datasets. Each
`info.json` records the package versions and the processing time.

### Thresholds of PANNs and YAMNet

PANNs and YAMNet are audio taggers trained on AudioSet: a frame is music when the score
of the AudioSet "Music" class exceeds a threshold. This threshold was tuned once to
maximize the global music F1 on the **dev** subsets of the three music datasets, and is
the default of `predict.py` (0.01 for PANNs, 0.0032 for YAMNet). To tune it again:

```bash
.envs/panns/bin/python predict.py panns mirex2015 openbmat seyerlehner --subset dev $DATA
.envs/eval/bin/python tune_thresholds.py panns $DATA
```

PANNs processes audio by 10 s chunks (the duration of its training clips) with 10 ms
frames. YAMNet scores 0.96 s windows every 0.48 s; each score is assigned to the central
0.48 s of its window.

### Speed

[`speed.py`](speed.py) times a system on a directory of audio files (audio decoding
included, model loading excluded), whatever the task it addresses. The speed table of
the main README was measured on a local copy of the InaGVAD test audio, running the
systems one after the other and keeping the best of two runs:

```bash
.envs/inavamos/bin/python speed.py inavamos /data/inaGVAD/test --device cuda
```

Then evaluate all the predictions and print the result tables:

```bash
.envs/eval/bin/python evaluate.py $DATA --output results.json
```

Speech is `speech`, `male` or `female` segments, and music is `music` segments. Labels
combining several classes, like `speech+music` for inaVAMOS, count for each class.
inaSpeechSegmenter predicts exclusive classes: speech over music is labelled as speech
only, which lowers its music recall on datasets with background music (OpenBMAT,
Seyerlehner).
