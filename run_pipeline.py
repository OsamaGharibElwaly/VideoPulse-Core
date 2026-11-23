"""VideoPulse Core CLI: gated CV + NLP analysis over 6 local videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.fusion_and_eval import process_video, run_all_videos  # noqa: E402
from core.media_probe import find_video_entry, load_manifest  # noqa: E402
from tools.generate_sample_assets import generate_assets  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run the local VideoPulse Core pipeline. Paths are resolved from the repository root."
    )
    parser.add_argument("--all", action="store_true", help="Process all 6 configured videos.")
    parser.add_argument("--video-id", metavar="ID", help="Process one full id or short alias.")
    parser.add_argument("--query", default="", metavar="TEXT", help="Query indexed moments.")
    parser.add_argument("--generate-samples", action="store_true", help="Create local sample assets first.")
    parser.add_argument(
        "--real-dir",
        metavar="DIR",
        help="Folder containing real videos and optional .srt/.vtt/.json/.csv/.log side-cars.",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        help="Ultralytics weights name for YOLO26 (default: yolo26n.pt). No implicit downloads.",
    )
    parser.add_argument(
        "--manifest",
        metavar="PATH",
        help="Manifest JSON (default: config/videos_manifest.json). Use config/real_videos_manifest.json for real footage.",
    )
    parser.add_argument(
        "--logs-dir",
        metavar="DIR",
        help="Directory holding sidecar transcripts/logs (default: data/logs).",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Report root (default: outputs). Use outputs/real for real footage.",
    )
    return parser


def _print_summary(report: dict[str, Any], query: str) -> None:
    """Print a compact human-readable report summary."""
    telemetry = report["telemetry"]
    print(f"[{report['video_id']}] {report['industry']}")
    print(
        f"  active={telemetry['active_feature_count']} "
        f"skipped={telemetry['skipped_feature_count']} "
        f"grounding={telemetry['grounding_match_rate']:.2f} "
        f"total_ms={telemetry['total_ms']:.1f} "
        f"frame_skip={telemetry['frames_skipped_ratio']:.2f} "
        f"detector={telemetry.get('detector_mode', 'unknown')}"
    )
    print(f"  media={report['media']['source']} detector_reason={telemetry.get('detector_reason', '')}")
    if report["skipped_features"]:
        skipped = ", ".join(
            f"{item['feature']}({item['reason']})" for item in report["skipped_features"]
        )
        print(f"  skipped: {skipped}")
    if query.strip() and report["retrieval_results"]:
        top = report["retrieval_results"][0]
        print(f"  top hit @{top['start_sec']:.1f}s score={top['score']:.2f}: {top['text'][:70]}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)

    def resolve(path: str | None, default: Path) -> Path:
        candidate = Path(path) if path else default
        return candidate if candidate.is_absolute() else ROOT / candidate

    config_path = resolve(args.manifest, ROOT / "config" / "videos_manifest.json")
    videos_dir = ROOT / "data" / "videos"
    logs_dir = resolve(args.logs_dir, ROOT / "data" / "logs")
    output_dir = resolve(args.output_dir, ROOT / "outputs")
    real_dir = Path(args.real_dir) if args.real_dir else None
    if real_dir is not None and not real_dir.is_absolute():
        real_dir = ROOT / real_dir
    try:
        config = load_manifest(config_path)
        if args.generate_samples:
            generate_assets(config, videos_dir=videos_dir, logs_dir=logs_dir)
            print(f"Sample assets ready in {videos_dir} and {logs_dir}.")
            if not args.all and not args.video_id:
                return 0
        if args.video_id:
            find_video_entry(config, args.video_id)
            reports = [
                process_video(
                    args.video_id,
                    config=config,
                    videos_dir=videos_dir,
                    logs_dir=logs_dir,
                    output_dir=output_dir,
                    query=args.query,
                    real_dir=real_dir,
                    model=args.model,
                )
            ]
        else:
            reports = run_all_videos(
                config,
                videos_dir=videos_dir,
                logs_dir=logs_dir,
                output_dir=output_dir,
                query=args.query,
                real_dir=real_dir,
                model=args.model,
            )
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        print(f"error: {exc}")
        if args.video_id:
            ids = ", ".join(str(v.get("video_id")) for v in config.get("videos", []))
            print(f"available ids: {ids}")
        return 1
    for report in reports:
        _print_summary(report, args.query)
    print(f"Reports written to {output_dir}/<video_id>/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
