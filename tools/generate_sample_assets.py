"""Generate tiny local sample clips and text logs for all 6 video slots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.media_probe import load_manifest  # noqa: E402
from core.synthetic_media import render_frame  # noqa: E402


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content.rstrip() + "\n")


def _write_clip(entry: dict[str, Any], videos_dir: Path) -> Path:
    """Render one short mp4 clip with OpenCV's built-in mp4v codec."""
    width = int(entry.get("width", 320))
    height = int(entry.get("height", 240))
    fps = float(entry.get("fps", 12))
    frame_count = max(int(fps * float(entry.get("duration_sec", 12.0))), 12)
    path = videos_dir / str(entry.get("file", f"{entry['video_id']}.mp4"))
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open video writer for {path}")
    try:
        scene = str(entry.get("content_profile", {}).get("scene", "static"))
        for index in range(frame_count):
            writer.write(
                render_frame(
                    scene,
                    index,
                    frame_count,
                    width,
                    height,
                    seed=str(entry["video_id"]),
                )
            )
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Video writer produced an empty file: {path}")
    return path


def generate_assets(
    config: dict[str, Any],
    *,
    videos_dir: Path,
    logs_dir: Path,
) -> list[dict[str, str]]:
    """Generate clips and text side-cars for every manifest video."""
    generated: list[dict[str, str]] = []
    for entry in config.get("videos", []):
        clip = _write_clip(entry, videos_dir)
        texts: list[str] = [str(clip)]
        for file_key, inline_key in (
            ("transcript_file", "sample_transcript"),
            ("log_file", "sample_log"),
        ):
            filename = entry.get(file_key)
            content = entry.get(inline_key)
            if filename and content:
                text_path = logs_dir / str(filename)
                _write_text(text_path, str(content))
                texts.append(str(text_path))
        generated.append(
            {"video_id": str(entry["video_id"]), "assets": ", ".join(texts)}
        )
    return generated


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with friendly errors (no raw tracebacks)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "videos_manifest.json"))
    parser.add_argument("--videos-dir", default=str(ROOT / "data" / "videos"))
    parser.add_argument("--logs-dir", default=str(ROOT / "data" / "logs"))
    args = parser.parse_args(argv)
    try:
        config = load_manifest(Path(args.config))
        generated = generate_assets(
            config,
            videos_dir=Path(args.videos_dir),
            logs_dir=Path(args.logs_dir),
        )
    except (OSError, RuntimeError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"Generated {len(generated)} sample clips and text side-cars:")
    for item in generated:
        print(f"  - {item['video_id']}: {item['assets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
