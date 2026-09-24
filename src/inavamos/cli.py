"""Command line interface: ``ina-vamos -i media.mp3 -o output_dir``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .export import FORMATS, export
from .models import DETECTORS, ModelAccessError
from .segmenter import Segmenter

logger = logging.getLogger("inavamos")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ina-vamos",
        description=(
            "inaVAMOS (Voice And Music Open Segmenter): speech (voice activity) and music "
            "detection in audio and video files, using SSL-based models from INA (https://huggingface.co/ina-foss)."
        ),
    )
    parser.add_argument("-i", "--input", nargs="+", required=True, help="media files to process (any format read by ffmpeg)")
    parser.add_argument(
        "-o", "--output-dir", type=Path, help="directory where results are written (default: print to standard output)"
    )
    parser.add_argument("-f", "--format", choices=list(FORMATS), default="csv", help="output format (default: csv)")
    parser.add_argument(
        "-d",
        "--detectors",
        nargs="+",
        default=list(DETECTORS),
        metavar="{speech,music}",
        help="detectors to run (default: speech music). 'vad' is an alias of 'speech'",
    )
    parser.add_argument("--device", help="torch device, e.g. cpu, cuda, cuda:1 (default: cuda if available)")
    parser.add_argument("--batch-size", type=int, default=8, help="number of windows processed together (default: 8)")
    parser.add_argument(
        "--window-duration",
        type=float,
        default=30.0,
        help="duration in seconds of the analysis windows (default: 30). 0 processes files at once",
    )
    parser.add_argument(
        "--context-duration",
        type=float,
        default=2.5,
        help="overlap in seconds on each side of the windows (default: 2.5)",
    )
    parser.add_argument("--start", type=float, help="start time in seconds of the portion to process")
    parser.add_argument("--stop", type=float, help="stop time in seconds of the portion to process")
    parser.add_argument("--token", help="HuggingFace token, only needed for private model repositories")
    parser.add_argument("--revision", help="revision of the models on the HuggingFace Hub")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing output files instead of skipping them")
    parser.add_argument("-v", "--verbose", action="store_true", help="print progress information")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO if args.verbose else logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    inputs = [Path(p) for p in args.input]
    outputs = {}
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in inputs:
            output = args.output_dir / (path.stem + FORMATS[args.format])
            if output in outputs.values():
                logger.error("Several inputs would be written to %s", output)
                return 2
            outputs[path] = output
        if not args.overwrite:
            todo = [p for p in inputs if not outputs[p].exists()]
            if len(todo) < len(inputs):
                logger.warning("Skipping %d files already processed (use --overwrite)", len(inputs) - len(todo))
            inputs = todo
    if not inputs:
        return 0

    try:
        segmenter = Segmenter(
            detectors=args.detectors,
            device=args.device,
            window_duration=args.window_duration or None,
            context_duration=args.context_duration,
            batch_size=args.batch_size,
            token=args.token,
            revision=args.revision,
        )
    except (ModelAccessError, ValueError) as e:
        logger.error("%s", e)
        return 2

    repo_ids = {name: spec.repo_id for name, spec in segmenter.specs.items()}
    failures = 0
    for i, path in enumerate(inputs, 1):
        t0 = time.perf_counter()
        try:
            result = segmenter.process(path, start=args.start, stop=args.stop)
        except Exception as e:  # keep processing the other files
            logger.error("%s: %s", path, e)
            failures += 1
            continue
        elapsed = time.perf_counter() - t0
        logger.info(
            "[%d/%d] %s: %.1fs of audio in %.1fs (%.0fx real time)",
            i,
            len(inputs),
            path,
            result.duration,
            elapsed,
            result.duration / max(elapsed, 1e-6),
        )
        content = export(result, args.format, file=str(path), models=repo_ids)
        if args.output_dir:
            outputs[path].write_text(content)
        else:
            if len(inputs) > 1:
                print(f"# {path}")
            sys.stdout.write(content)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
