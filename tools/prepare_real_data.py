"""Prepare authentic detection-derived event logs for the real-video track.

Runs local YOLO26 weights over real footage and writes a timestamped JSON log
per clip. The logs are machine-generated from the same frames the CV pipeline
analyses, so the NLP track consumes real detections instead of invented text.
No network access is used unless ``--download`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.media_probe import load_manifest  # noqa: E402
from core.real_media_ingest import probe_real_stream, read_real_frames  # noqa: E402
from core.yolo_detector import DetectorConfig, build_detector  # noqa: E402

VIDEO_BASE = "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"


def _download(url: str, target: Path) -> None:
    """Download one asset when explicitly requested."""
    from urllib.request import urlretrieve

    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print(f"downloading {url} -> {target}")
        urlretrieve(url, str(target))


def aggregate_events(detections: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Group real detections into timestamped, zone-aware event records."""
    grouped: dict[tuple[float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in detections:
        key = (
            round(float(item.get("timestamp_sec", 0.0)), 3),
            str(item.get("zone_id", "")),
            str(item.get("label", "")),
        )
        grouped[key].append(item)
    events: list[dict[str, Any]] = []
    for (timestamp, zone, label), items in sorted(grouped.items()):
        confidences = [float(item.get("confidence", 0.0)) for item in items]
        events.append(
            {
                "timestamp_sec": timestamp,
                "label": label,
                "count": len(items),
                "confidence": round(max(confidences), 4),
                "zone": zone,
                "source": source,
            }
        )
    return events


def prepare(
    manifest: Path,
    real_dir: Path,
    logs_dir: Path,
    model: str,
    sample_count: int,
    force: bool,
    download: bool,
) -> list[dict[str, Any]]:
    """Generate one detection-derived JSON log per real video."""
    config = load_manifest(manifest)
    logs_dir.mkdir(parents=True, exist_ok=True)
    detector = build_detector(DetectorConfig(model=model))
    summary: list[dict[str, Any]] = []
    for entry in config.get("videos", []):
        video_id = str(entry["video_id"])
        filename = str(entry["file"])
        video = real_dir / filename
        if download:
            _download(VIDEO_BASE + filename, video)
            _download(WEIGHTS_URL, ROOT / "weights" / model)
        if not video.is_file():
            raise FileNotFoundError(f"missing real video: {video} (pass --download)")
        probe = probe_real_stream(video)
        if probe.status != "ok":
            raise RuntimeError(f"{video_id}: {probe.reason}")
        frames, timestamps = read_real_frames(
            video, probe, sample_count, target_size=(320, 240)
        )
        result = detector.detect(
            frames,
            timestamps,
            width=320,
            height=240,
            rois=entry.get("rois", []),
            scene=str(entry.get("content_profile", {}).get("scene", "")),
        )
        events = aggregate_events(result.detections, f"{result.model}:real_frames")
        payload = {
            "video_id": video_id,
            "generated_by": "tools/prepare_real_data.py",
            "source_dataset": entry.get("source_dataset", ""),
            "detector_mode": result.mode,
            "model": result.model,
            "sampled_frames": len(frames),
            "events": events,
        }
        log_path = logs_dir / f"{video.stem}.json"
        if log_path.exists() and not force:
            print(f"{video_id}: keeping existing {log_path.name}")
        else:
            log_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"{video_id}: wrote {len(events)} real detection events -> {log_path.name}")
        summary.append(
            {
                "video_id": video_id,
                "events": len(events),
                "detections": len(result.detections),
                "detector_mode": result.mode,
                "log": str(log_path),
            }
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "config" / "real_videos_manifest.json"))
    parser.add_argument("--real-dir", default=str(ROOT / "data" / "real" / "videos"))
    parser.add_argument("--logs-dir", default=str(ROOT / "data" / "real" / "logs"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--force", action="store_true", help="Rewrite existing logs.")
    parser.add_argument("--download", action="store_true", help="Fetch source videos and weights.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        prepare(
            Path(args.manifest),
            Path(args.real_dir),
            Path(args.logs_dir),
            args.model,
            args.sample_count,
            args.force,
            args.download,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
