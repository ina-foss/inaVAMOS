"""Export of segmentation results to common file formats."""

from __future__ import annotations

import csv
import io
import json

from .segmenter import Result

FORMATS = {
    "csv": ".csv",
    "tsv": ".tsv",
    "json": ".json",
    "textgrid": ".TextGrid",
    "audacity": ".txt",
}


def to_csv(result: Result, delimiter: str = ",") -> str:
    """Combined timeline as a table with ``label,start,stop`` columns."""
    out = io.StringIO()
    writer = csv.writer(out, delimiter=delimiter, lineterminator="\n")
    writer.writerow(["label", "start", "stop"])
    for label, start, stop in result.timeline():
        writer.writerow([label, f"{start:.3f}", f"{stop:.3f}"])
    return out.getvalue()


def to_json(result: Result, **metadata) -> str:
    """Per-detector segments and combined timeline as JSON."""
    return json.dumps({**metadata, **result.to_dict()}, indent=2)


def to_audacity(result: Result) -> str:
    """Combined timeline as an Audacity label track (File > Import > Labels)."""
    return "".join(f"{start:.6f}\t{stop:.6f}\t{label}\n" for label, start, stop in result.timeline())


def _quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def to_textgrid(result: Result) -> str:
    """Praat TextGrid with one tier per detector and a tier for the combined timeline."""
    xmin, xmax = result.offset, result.offset + result.duration
    tiers = [(name, result.segments(name, inactive_label="")) for name in result.detectors]
    tiers.append(("timeline", result.timeline()))

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {xmin}",
        f"xmax = {xmax}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for i, (name, segments) in enumerate(tiers, 1):
        if not segments:
            segments = [("", xmin, xmax)]
        lines += [
            f"    item [{i}]:",
            '        class = "IntervalTier"',
            f"        name = {_quote(name)}",
            f"        xmin = {xmin}",
            f"        xmax = {xmax}",
            f"        intervals: size = {len(segments)}",
        ]
        for j, (label, start, stop) in enumerate(segments, 1):
            lines += [
                f"        intervals [{j}]:",
                f"            xmin = {start}",
                f"            xmax = {stop}",
                f"            text = {_quote(label)}",
            ]
    return "\n".join(lines) + "\n"


def export(result: Result, fmt: str, **metadata) -> str:
    """Serialize ``result`` to one of :data:`FORMATS`."""
    fmt = fmt.lower()
    if fmt == "csv":
        return to_csv(result)
    if fmt == "tsv":
        return to_csv(result, delimiter="\t")
    if fmt == "json":
        return to_json(result, **metadata)
    if fmt == "textgrid":
        return to_textgrid(result)
    if fmt == "audacity":
        return to_audacity(result)
    raise ValueError(f"Unknown format {fmt!r}. Available: {', '.join(FORMATS)}")
