"""Render one CV+NLP result MP4 per processed video.

Each reel walks the exact sampled frames used by the pipeline and overlays:

* YOLO26 boxes with label/confidence (or the OpenCV fallback boxes),
* manifest/auto zones, keyframes, anomalies, text regions and action states,
* the timestamp-aligned transcript/log line, parsed event, grounding verdict
  and top retrieval hit from ``report.json``.

These are analysis reels, not full-resolution re-encodes. They make every
processed video directly viewable instead of leaving only a JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.media_probe import load_manifest, probe_media  # noqa: E402
from core.yolo_detector import DetectorConfig, YOLODetector  # noqa: E402

OUT_W, OUT_H = 320, 240
FPS = 8
HOLD = 4
TITLE_FRAMES = 12
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _wrap(text: str, width: int, limit: int) -> list[str]:
    """Return at most ``limit`` wrapped lines."""
    lines = textwrap.wrap(str(text), width) or [""]
    return lines[:limit]


def _near(records: list[dict[str, Any]], timestamp: float, tolerance: float, key: str = "timestamp_sec") -> dict[str, Any] | None:
    """Return the closest record inside ``tolerance`` seconds."""
    best: dict[str, Any] | None = None
    distance = float("inf")
    for record in records:
        if key not in record:
            continue
        delta = abs(float(record[key]) - timestamp)
        if delta < distance:
            best, distance = record, delta
    return best if distance <= tolerance else None


def _state_at(states: list[dict[str, Any]], timestamp: float) -> dict[str, Any] | None:
    """Return the action state covering ``timestamp``."""
    for state in states:
        if float(state.get("start_sec", 0)) <= timestamp <= float(state.get("end_sec", 0)):
            return state
    return None


def _title_card(report: dict[str, Any]) -> np.ndarray:
    """Build a self-describing opening frame."""
    frame = np.full((OUT_H, OUT_W, 3), (24, 30, 39), dtype=np.uint8)
    cv2.putText(frame, "CV + NLP RESULT REEL", (12, 32), FONT, 0.62, (245, 249, 252), 2)
    lines = [
        report["video_id"],
        f"{report['industry']}",
        f"source   : {report['media']['source']}",
        f"detector : {report['detector']['mode']} / {report['detector']['model']}",
        f"features : {report['telemetry']['active_feature_count']} active, "
        f"{report['telemetry']['skipped_feature_count']} skipped",
        f"objects  : {report['detector']['detections']} across "
        f"{report['detector']['frames_analyzed']} sampled frames",
        "",
        "amber=YOLO box  cyan=zone  green=text region",
        "footer=aligned text + event + grounding",
    ]
    y = 62
    for line in lines:
        for piece in _wrap(line, 46, 2):
            cv2.putText(frame, piece, (12, y), FONT, 0.36, (188, 205, 220), 1)
            y += 15
    return frame


def _overlay(
    frame: np.ndarray,
    index: int,
    timestamp: float,
    context: Any,
    result: Any,
    report: dict[str, Any],
    tolerance: float,
) -> np.ndarray:
    """Draw CV and NLP evidence on one resized frame."""
    out = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    sx, sy = OUT_W / max(context.width, 1), OUT_H / max(context.height, 1)
    frame_detections = [item for item in result.detections if item.get("frame_index") == index]

    # Header.
    cv2.rectangle(out, (0, 0), (OUT_W, 27), (18, 24, 32), -1)
    cv2.putText(
        out,
        f"{report['video_id'][:22]} | {report['media']['source'][:10]} | "
        f"{result.mode.upper()} objs={len(frame_detections)}",
        (5, 18), FONT, 0.38, (245, 249, 252), 1,
    )
    cv2.putText(out, f"{timestamp:05.1f}s", (OUT_W - 48, 18), FONT, 0.38, (255, 209, 102), 1)

    # Zones (cyan when a report zone is active near this frame).
    zones_by_id = {str(zone.get("zone_id")): zone for zone in report.get("zones", [])}
    for roi in context.rois:
        zone_id = str(roi.get("zone_id", ""))
        x1, y1, x2, y2 = (float(value) for value in roi.get("coords", [0, 0, 1, 1])[:4])
        if max(x1, y1, x2, y2) <= 1.0:
            x1, x2, y1, y2 = x1 * OUT_W, x2 * OUT_W, y1 * OUT_H, y2 * OUT_H
        else:
            x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
        active = any(
            abs(float(stamp) - timestamp) <= tolerance
            for stamp in zones_by_id.get(zone_id, {}).get("active_timestamps", [])
        )
        color = (255, 214, 82) if active else (90, 130, 150)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, cv2.LINE_AA)
        cv2.putText(out, zone_id[:16], (int(x1) + 3, int(y1) + 11), FONT, 0.32, color, 1)

    # Text regions from the report (green) when present near this frame.
    text_regions = [
        item for item in report.get("text_regions", [])
        if abs(float(item.get("timestamp_sec", 0)) - timestamp) <= tolerance
    ]
    for region in text_regions[:10]:
        x, y, w, h = (float(value) for value in region.get("bbox", [0, 0, 0, 0]))
        cv2.rectangle(
            out, (int(x * sx), int(y * sy)), (int((x + w) * sx), int((y + h) * sy)),
            (124, 255, 178), 1, cv2.LINE_AA,
        )

    # YOLO / fallback boxes.
    for item in frame_detections:
        x, y, w, h = (float(value) for value in item.get("bbox", [0, 0, 0, 0]))
        x1, y1 = int(x * sx), int(y * sy)
        x2, y2 = int((x + w) * sx), int((y + h) * sy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 209, 102), 2, cv2.LINE_AA)
        label = f"{item.get('label', 'object')} {float(item.get('confidence', 0)):.2f}"
        if item.get("zone_id"):
            label += f" @{item['zone_id'][:9]}"
        cv2.putText(out, label[:28], (x1, max(y1 - 3, 30)), FONT, 0.31, (18, 24, 32), 3)
        cv2.putText(out, label[:28], (x1, max(y1 - 3, 30)), FONT, 0.31, (255, 235, 150), 1)

    # CV feature callouts.
    keyframe = _near(report.get("keyframes", []), timestamp, tolerance)
    if keyframe:
        cv2.putText(
            out,
            f"KEYFRAME motion={float(keyframe.get('motion_score', 0)):.1f} "
            f"objs={keyframe.get('object_count', 0)}",
            (5, 43), FONT, 0.34, (255, 170, 120), 1,
        )
    anomaly = _near(report.get("anomalies", []), timestamp, tolerance)
    if anomaly:
        cv2.putText(
            out,
            f"ANOMALY {anomaly.get('kind', '')} score={float(anomaly.get('score', 0)):.1f}",
            (5, 58), FONT, 0.34, (120, 190, 255), 1,
        )
    state = _state_at(report.get("action_states", []), timestamp)
    if state:
        label = f"STATE {state.get('state', '')}"
        (text_w, _), _ = cv2.getTextSize(label, FONT, 0.34, 1)
        cv2.rectangle(out, (OUT_W - text_w - 12, 30), (OUT_W - 4, 48), (40, 90, 70), -1)
        cv2.putText(out, label, (OUT_W - text_w - 8, 43), FONT, 0.34, (235, 255, 240), 1)

    # NLP evidence caption.
    segment = _near(report.get("aligned_segments", []), timestamp, max(tolerance, 1.2), key="start_sec")
    event = _near(report.get("parsed_events", []), timestamp, tolerance)
    grounding = None
    if event:
        grounding = next(
            (item for item in report.get("grounding_checks", []) if item.get("event_id") == event.get("event_id")),
            None,
        )
    hit = _near(report.get("retrieval_results", []), timestamp, max(tolerance, 1.2), key="start_sec")
    lines: list[str] = []
    if segment:
        lines.append(f"TEXT {segment['text'][:86]}")
    if event:
        lines.append(f"EVENT {event['event_id']} zone={event.get('zone', '')} :: {event.get('claim', '')[:70]}")
    if grounding:
        evidence = ",".join(grounding.get("supporting_evidence", [])[:3])
        lines.append(f"GROUND {grounding.get('status', '')} [{evidence}]")
    if hit:
        lines.append(f"RETR @{float(hit['start_sec']):.1f}s score={float(hit.get('score', 0)):.2f}")
    if not lines:
        lines.append("NLP: no aligned text at this frame (gate/skip recorded in report)")
    wrapped: list[str] = []
    for line in lines[:3]:
        wrapped.extend(_wrap(line, 47, 2))
    wrapped = wrapped[:5]
    band_h = 14 + 12 * len(wrapped)
    cv2.rectangle(out, (0, OUT_H - band_h), (OUT_W, OUT_H), (16, 21, 28), -1)
    for row, line in enumerate(wrapped):
        cv2.putText(out, line[:52], (5, OUT_H - band_h + 12 + row * 12), FONT, 0.30, (222, 235, 245), 1)
    return out


def render_one(
    *,
    report: dict[str, Any],
    manifest: dict[str, Any],
    videos_dir: Path,
    logs_dir: Path,
    real_dir: Path | None,
    output_dir: Path,
) -> Path:
    """Render and verify one result reel."""
    from core.media_probe import find_video_entry

    entry = find_video_entry(manifest, report["video_id"])
    context = probe_media(
        entry, videos_dir=videos_dir, logs_dir=logs_dir,
        real_dir=real_dir,
    )
    if not context.sampled_frames:
        raise RuntimeError(f"no frames for {report['video_id']}")
    detector = YOLODetector(DetectorConfig(model=report["detector"]["model"]))
    detection = detector.detect(
        context.sampled_frames,
        context.timestamps,
        width=context.width,
        height=context.height,
        rois=context.rois,
        scene=context.scene,
        auto_zones=context.auto_zones,
    )
    interval = (
        (context.timestamps[-1] - context.timestamps[0]) / max(len(context.timestamps) - 1, 1)
        if len(context.timestamps) > 1 else 1.0
    )
    tolerance = max(0.75, interval * 0.75)

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{report['video_id']}_cv_nlp.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {target}")
    try:
        for _ in range(TITLE_FRAMES):
            writer.write(_title_card(report))
        for index, (frame, timestamp) in enumerate(zip(context.sampled_frames, context.timestamps)):
            overlay = _overlay(frame, index, timestamp, context, detection, report, tolerance)
            for _ in range(HOLD):
                writer.write(overlay)
    finally:
        writer.release()
    assert target.is_file() and target.stat().st_size > 1000, f"empty result video: {target}"
    print(f"rendered {target} ({len(context.sampled_frames)} frames, {detection.mode})")
    return target


def contact_sheet(videos: list[Path], target: Path) -> None:
    """Build a thumbnail sheet from the rendered reels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 5
    rows = max(1, int(np.ceil(len(videos) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(14.5, 2.9 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, path in zip(axes, videos):
        capture = cv2.VideoCapture(str(path))
        capture.set(cv2.CAP_PROP_POS_MSEC, 1900)
        ok, frame = capture.read()
        capture.release()
        if ok:
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(path.stem.replace("_cv_nlp", "").replace("_", " "), fontsize=8.5, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Processed CV + NLP result videos (one reel per input video)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=170, bbox_inches="tight")
    plt.close(fig)
    assert target.stat().st_size > 1000


def _render_set(
    manifest_path: Path,
    reports_dir: Path,
    videos_dir: Path,
    logs_dir: Path,
    real_dir: Path | None,
    output_dir: Path,
    limit: int | None,
) -> list[Path]:
    manifest = load_manifest(manifest_path)
    reports = sorted(reports_dir.glob("*/report.json"))
    if limit is not None:
        reports = reports[:limit]
    written = []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        written.append(
            render_one(
                report=report, manifest=manifest, videos_dir=videos_dir,
                logs_dir=logs_dir, real_dir=real_dir, output_dir=output_dir,
            )
        )
    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--reports-dir")
    parser.add_argument("--videos-dir", default=str(ROOT / "data" / "videos"))
    parser.add_argument("--logs-dir", default=str(ROOT / "data" / "logs"))
    parser.add_argument("--real-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--contact-sheet", default=str(ROOT / "docs" / "figures" / "result_video_contact_sheet.png"))
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render the two default sets, or one custom set when flags are given."""
    args = build_parser().parse_args(argv)
    if args.manifest or args.reports_dir or args.output_dir:
        if not args.manifest or not args.reports_dir or not args.output_dir:
            print("error: --manifest, --reports-dir and --output-dir must be used together")
            return 1
        written = _render_set(
            Path(args.manifest), Path(args.reports_dir), Path(args.videos_dir),
            Path(args.logs_dir), Path(args.real_dir) if args.real_dir else None,
            Path(args.output_dir), args.limit,
        )
        videos = sorted(Path(args.output_dir).glob("*.mp4"))
    else:
        written = []
        written += _render_set(
            ROOT / "config" / "videos_manifest.json", ROOT / "outputs",
            ROOT / "data" / "videos", ROOT / "data" / "logs", None,
            ROOT / "docs" / "videos" / "generated", None,
        )
        written += _render_set(
            ROOT / "config" / "real_videos_manifest.json", ROOT / "outputs" / "real",
            ROOT / "data" / "videos", ROOT / "data" / "real" / "logs",
            ROOT / "data" / "real" / "videos", ROOT / "docs" / "videos" / "real", None,
        )
        videos = sorted((ROOT / "docs" / "videos").glob("*/*.mp4"))
    if not written:
        print("error: no reports found to render")
        return 1
    contact_sheet(videos, Path(args.contact_sheet))
    print(f"wrote {len(written)} result videos and contact sheet {Path(args.contact_sheet).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
