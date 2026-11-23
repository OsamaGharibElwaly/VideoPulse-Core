"""Create a VideoPulse manifest from any folder of video files.

This is the bridge from a downloaded dataset (UCF101, Kinetics, SoccerNet,
stock clips, your own footage) to the existing real-data pipeline. It writes a
manifest that `tools/prepare_real_data.py` and `run_pipeline.py --manifest`
already understand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.real_media_ingest import REAL_VIDEO_EXTENSIONS  # noqa: E402

DEFAULT_TAGS = ["real", "dataset_clip", "object_motion"]


def build_manifest(
    videos_dir: Path,
    output: Path,
    *,
    limit: int | None,
    industry: str,
    prefix: str,
) -> int:
    """Write one manifest entry per discovered video; return the count."""
    files = sorted(
        path for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in REAL_VIDEO_EXTENSIONS
    )
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError(f"no supported videos found under {videos_dir}")
    entries = []
    for index, path in enumerate(files, start=1):
        video_id = f"{prefix}_{index:04d}"
        entries.append(
            {
                "video_id": video_id,
                "short_id": f"{prefix}_{index}",
                "file": str(path.relative_to(videos_dir)).replace("\\", "/"),
                "industry": industry,
                "description": f"Dataset clip {path.name}",
                "source_dataset": str(videos_dir),
                "visual_tags": DEFAULT_TAGS,
                "rois": [
                    {"zone_id": "left_area", "kind": "box", "coords": [0.0, 0.0, 0.5, 1.0]},
                    {"zone_id": "right_area", "kind": "box", "coords": [0.5, 0.0, 1.0, 1.0]},
                ],
                "content_profile": {
                    "scene": "dataset",
                    "baseline_mode": "dynamic_scene",
                    "text_mode": "unknown",
                    "groundable": True,
                    "state_vocabulary": ["QUIET", "ACTIVE", "BURST"],
                },
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    import json

    output.write_text(
        json.dumps(
            {"version": "1.0.0", "project": "VideoPulse bulk dataset track", "videos": entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(entries)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", required=True, help="Folder containing downloaded videos.")
    parser.add_argument("--output", default=str(ROOT / "config" / "bulk_videos_manifest.json"))
    parser.add_argument("--limit", type=int, help="Only use the first N videos (e.g. 300).")
    parser.add_argument("--industry", default="Bulk video dataset")
    parser.add_argument("--prefix", default="bulk")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        count = build_manifest(
            Path(args.videos_dir), Path(args.output),
            limit=args.limit, industry=args.industry, prefix=args.prefix,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"wrote {count} manifest entries -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
