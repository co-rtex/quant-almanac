"""View-model helpers: formatting filters and chart geometry for console templates.

Chart geometry is computed here (not in templates) so markup stays declarative and
the page needs no inline scripts or styles, which keeps a strict CSP possible.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

STATUS_LABELS = {
    "pass": ("✓", "Pass"),
    "warn": ("!", "Warning"),
    "fail": ("✕", "Fail"),
    "unavailable": ("–", "Unavailable"),
}


def status_icon(status: str | None) -> str:
    """Return the glyph paired with a status color (never color alone)."""

    return STATUS_LABELS.get(status or "unavailable", STATUS_LABELS["unavailable"])[0]


def status_label(status: str | None) -> str:
    """Return the human label for a status."""

    return STATUS_LABELS.get(status or "unavailable", STATUS_LABELS["unavailable"])[1]


def status_class(status: str | None) -> str:
    """Return the CSS modifier for a status."""

    return status if status in STATUS_LABELS else "unavailable"


def fmt_num(value: object, digits: int = 3) -> str:
    """Format a number, rendering missing values as an em dash."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def fmt_pct(value: object, digits: int = 1) -> str:
    """Format a ratio as a signed percentage."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value * 100:+.{digits}f}%"


def short_id(value: object, length: int = 8) -> str:
    """Abbreviate an identifier or digest."""

    return str(value)[:length] if value else "—"


def pretty_json(value: object) -> str:
    """Render evidence payloads as stable, indented JSON."""

    return json.dumps(value, indent=2, sort_keys=True, default=str)


def calibration_chart(bins: Sequence[Mapping[str, Any]], size: int = 300) -> dict[str, Any]:
    """Return SVG geometry for a reliability diagram (mean prediction vs observed rate)."""

    pad_left, pad_bottom, pad_top, pad_right = 44, 36, 12, 12
    plot = size - pad_left - pad_right
    plot_h = size - pad_top - pad_bottom

    def x(value: float) -> float:
        return round(pad_left + value * plot, 2)

    def y(value: float) -> float:
        return round(pad_top + (1.0 - value) * plot_h, 2)

    points = [
        {
            "cx": x(float(row["mean_prediction"])),
            "cy": y(float(row["observed_rate"])),
            "prediction": float(row["mean_prediction"]),
            "observed": float(row["observed_rate"]),
            "count": int(float(row["count"])),
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
        }
        for row in bins
        if "mean_prediction" in row and "observed_rate" in row
    ]
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    return {
        "width": size,
        "height": size,
        "points": points,
        "path": " ".join(
            f"{'M' if index == 0 else 'L'}{point['cx']},{point['cy']}"
            for index, point in enumerate(points)
        ),
        "diagonal": {"x1": x(0), "y1": y(0), "x2": x(1), "y2": y(1)},
        "x_ticks": [{"x": x(tick), "y": y(0), "label": f"{tick:g}"} for tick in ticks],
        "y_ticks": [{"x": x(0), "y": y(tick), "x2": x(1), "label": f"{tick:g}"} for tick in ticks],
        "plot": {"left": pad_left, "top": pad_top, "right": x(1), "bottom": y(0)},
    }


def importance_chart(
    importance: Sequence[Mapping[str, Any]],
    limit: int = 10,
    width: int = 560,
) -> dict[str, Any]:
    """Return SVG geometry for a horizontal mean-|SHAP| bar chart."""

    rows = [row for row in importance[:limit] if "mean_abs_shap" in row]
    label_width, value_width, row_height, bar_height = 190, 70, 26, 14
    bar_space = width - label_width - value_width
    maximum = max((float(row["mean_abs_shap"]) for row in rows), default=0.0) or 1.0
    bars = []
    for index, row in enumerate(rows):
        value = float(row["mean_abs_shap"])
        bar_width = max(2.0, round(value / maximum * bar_space, 2))
        top = index * row_height
        bars.append(
            {
                "feature": str(row.get("feature", "")),
                "value": value,
                "y": top + (row_height - bar_height) / 2,
                "text_y": top + row_height / 2 + 4,
                "x": label_width,
                "width": bar_width,
                "height": bar_height,
                "value_x": label_width + bar_width + 6,
            }
        )
    return {
        "width": width,
        "height": max(row_height, len(rows) * row_height),
        "bars": bars,
        "label_x": label_width - 8,
    }


def split_timeline(training_config: Mapping[str, Any] | None, width: int = 720) -> dict[str, Any]:
    """Return SVG geometry for train | embargo | validation | embargo | test sessions."""

    split = training_config.get("split") if training_config else None
    if not isinstance(split, Mapping) or "train_dates" not in split:
        return {"available": False}
    embargo = int(split.get("embargo_days", 0))
    segments_spec = [
        ("train", "Train", int(split["train_dates"])),
        ("embargo", "Embargo", embargo),
        ("validation", "Validation", int(split["validation_dates"])),
        ("embargo", "Embargo", embargo),
        ("test", "Test", int(split["test_dates"])),
    ]
    total = sum(count for _, _, count in segments_spec) or 1
    gap = 2.0
    usable = width - gap * (len(segments_spec) - 1)
    cursor = 0.0
    segments = []
    for kind, label, count in segments_spec:
        segment_width = max(3.0, usable * count / total) if count else 0.0
        segments.append(
            {
                "kind": kind,
                "label": label,
                "sessions": count,
                "x": round(cursor, 2),
                "width": round(segment_width, 2),
                "label_x": round(cursor + segment_width / 2, 2),
                "show_label": segment_width >= 70,
            }
        )
        cursor += segment_width + (gap if segment_width else 0.0)
    legend: list[dict[str, Any]] = []
    for segment in segments:
        if all(item["kind"] != segment["kind"] for item in legend):
            legend.append(segment)
    return {
        "available": True,
        "width": width,
        "height": 44,
        "segments": segments,
        "legend": legend,
    }
