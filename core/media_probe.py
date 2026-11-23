"""Pre-run media probe.

Inspects a real ``.mp4`` when present, otherwise produces a deterministic
synthetic stand-in. The probe never raises on missing media: it records what
was observed so the feature router can gate honestly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.real_media_ingest import ingest_real
from core.synthetic_media import render_frame

SAMPLE_FRAMES = 24
MIN_MOTION_ENERGY = 0.30
MIN_TEXT_DENSITY = 0.03


@dataclass
class MediaContext:
    """Everything the router and pipelines need to know about one video."""

    video_id: str
    industry: str = ""
    entry: dict[str, Any] = field(default_factory=dict)
    media_available: bool = False
    source: str = "synthetic_fallback"
    probe_error: str = ""
    media_path: Path = field(default_factory=Path)
    fps: float = 12.0
    width: int = 320
    height: int = 240
    frame_count: int = 0
    duration_sec: float = 0.0
    motion_energy: float = 0.0
    edge_density: float = 0.0
    text_region_density: float = 0.0
    roi_presence: bool = False
    transcript_available: bool = False
    log_available: bool = False
    transcript_source: str = "none"
    log_source: str = "none"
    transcript_path: Path | None = None
    log_path: Path | None = None
    transcript_text: str = ""
    log_text: str = ""
    profile: dict[str, Any] = field(default_factory=dict)
    visual_tags: list[str] = field(default_factory=list)
    rois: list[dict[str, Any]] = field(default_factory=list)
    sampled_frames: list[np.ndarray] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    # Phase 2: real-ingestion and detector-derived context.
    use_real: bool = False
    ingest_status: str = "not_requested"
    sidecar_sources: list[str] = field(default_factory=list)
    sidecar_records: list[dict[str, Any]] = field(default_factory=list)
    has_audio_stream: bool | None = None
    auto_zones: bool = False
    detected_zone_ids: list[str] = field(default_factory=list)

    @property
    def scene(self) -> str:
        """Return the authored scene profile name."""
        return str(self.profile.get("scene", "static"))

    @property
    def text_source_available(self) -> bool:
        """Return whether any transcript or log text can be aligned."""
        return bool(self.transcript_text.strip() or self.log_text.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary (frames are intentionally omitted)."""
        return {
            "video_id": self.video_id,
            "industry": self.industry,
            "media_available": self.media_available,
            "source": self.source,
            "probe_error": self.probe_error,
            "fps": round(self.fps, 3),
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "duration_sec": round(self.duration_sec, 3),
            "motion_energy": round(self.motion_energy, 4),
            "edge_density": round(self.edge_density, 4),
            "text_region_density": round(self.text_region_density, 4),
            "roi_presence": self.roi_presence,
            "transcript_available": self.transcript_available,
            "log_available": self.log_available,
            "transcript_source": self.transcript_source,
            "log_source": self.log_source,
            "use_real": self.use_real,
            "ingest_status": self.ingest_status,
            "sidecar_sources": list(self.sidecar_sources),
            "has_audio_stream": self.has_audio_stream,
        }


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the video manifest as UTF-8 JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_video_entry(config: dict[str, Any], video_id: str) -> dict[str, Any]:
    """Resolve a full video id or short alias to a manifest entry."""
    for entry in config.get("videos", []):
        if video_id in (entry.get("video_id"), entry.get("short_id")):
            return entry
    raise KeyError(f"Unknown video id: {video_id}")


def probe_media(
    entry: dict[str, Any],
    *,
    videos_dir: Path,
    logs_dir: Path,
    sample_count: int = SAMPLE_FRAMES,
    real_dir: Path | None = None,
) -> MediaContext:
    """Probe one manifest entry and return a content-aware context.

    With ``real_dir`` set, a real video and its side-cars are preferred. Any
    ingestion problem becomes an explicit status code and the Phase 1 sample
    / synthetic path takes over, so missing media is never fatal. Real videos
    never inherit the manifest's sample transcript text.
    """
    defaults = {"fps": 12, "duration_sec": 12.0, "width": 320, "height": 240}
    video_id = str(entry["video_id"])
    profile = dict(entry.get("content_profile", {}))
    fps = float(entry.get("fps", defaults["fps"]))
    width = int(entry.get("width", defaults["width"]))
    height = int(entry.get("height", defaults["height"]))
    duration = float(entry.get("duration_sec", defaults["duration_sec"]))
    media_path = Path(videos_dir) / str(entry.get("file", f"{video_id}.mp4"))

    real_ingest = None
    if real_dir is not None:
        real_ingest = ingest_real(
            entry, Path(real_dir), Path(logs_dir), sample_count, target_size=(width, height)
        )
    real_ok = real_ingest is not None and real_ingest.status == "ok"

    if real_ok:
        sidecars = real_ingest.sidecars  # type: ignore[union-attr]
        transcript_text = sidecars.transcript_text
        log_text = sidecars.log_text
        transcript_path = sidecars.transcript_path
        log_path = sidecars.log_path
        transcript_source = sidecars.transcript_source
        log_source = sidecars.log_source
    else:
        transcript_path, transcript_text, transcript_source = _resolve_text(
            entry, "transcript_file", "sample_transcript", logs_dir
        )
        log_path, log_text, log_source = _resolve_text(
            entry, "log_file", "sample_log", logs_dir
        )

    context = MediaContext(
        video_id=video_id,
        industry=str(entry.get("industry", "")),
        entry=dict(entry),
        media_path=media_path,
        fps=fps,
        width=width,
        height=height,
        duration_sec=duration,
        profile=profile,
        visual_tags=[str(tag) for tag in entry.get("visual_tags", [])],
        rois=[dict(roi) for roi in entry.get("rois", [])],
        roi_presence=bool(entry.get("rois")),
        transcript_available=bool(transcript_text.strip()),
        log_available=bool(log_text.strip()),
        transcript_source=transcript_source,
        log_source=log_source,
        transcript_path=transcript_path,
        log_path=log_path,
        transcript_text=transcript_text,
        log_text=log_text,
        use_real=real_ok,
        ingest_status=(
            "ok"
            if real_ok
            else (real_ingest.status if real_ingest else "REAL_VIDEO_NOT_FOUND")
        ),
        sidecar_sources=[
            source
            for source in (
                transcript_source if real_ok else "none",
                log_source if real_ok else "none",
            )
            if source != "none"
        ],
        sidecar_records=list(real_ingest.sidecars.records) if real_ok else [],
        has_audio_stream=real_ingest.stream.has_audio_stream if real_ok else None,
        auto_zones=bool(profile.get("auto_zones", False)),
    )

    if real_ok:
        frames = list(real_ingest.frames)  # type: ignore[union-attr]
        timestamps = list(real_ingest.timestamps)  # type: ignore[union-attr]
        measured_frame_count = real_ingest.stream.frame_count  # type: ignore[union-attr]
        context.media_available = True
        context.source = "real_file"
        context.media_path = real_ingest.paths.video_path  # type: ignore[union-attr]
        context.fps = real_ingest.stream.fps or fps  # type: ignore[union-attr]
        context.width = real_ingest.stream.width or width  # type: ignore[union-attr]
        context.height = real_ingest.stream.height or height  # type: ignore[union-attr]
        context.probe_error = real_ingest.sidecars.reason  # type: ignore[union-attr]
    else:
        observed = _observe_video(media_path, context, sample_count)
        if real_ingest is not None:
            context.probe_error = real_ingest.reason
        if observed is None:
            frames, timestamps, measured_frame_count = _synthetic_observations(context, sample_count)
            context.media_available = False
            context.source = "synthetic_fallback"
            if not media_path.exists() and not context.probe_error:
                context.probe_error = f"media file not found: {media_path}"
        else:
            frames, timestamps, measured_frame_count = observed
            context.media_available = True
            context.source = "file"
    context.sampled_frames = frames
    context.timestamps = timestamps
    context.frame_count = measured_frame_count or max(int(fps * duration), 1)
    context.duration_sec = (
        round(context.frame_count / fps, 3) if measured_frame_count else duration
    )
    if frames:
        context.motion_energy = _motion_energy(frames)
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        context.edge_density = float(
            np.mean([float(np.mean(cv2.Canny(gray, 60, 160) > 0)) for gray in gray_frames])
        )
        context.text_region_density = _text_region_density(gray_frames)
    return context


def _resolve_text(
    entry: dict[str, Any],
    file_key: str,
    inline_key: str,
    logs_dir: Path,
) -> tuple[Path | None, str, str]:
    """Prefer a real file; fall back to inline sample text in the manifest."""
    filename = entry.get(file_key)
    path = Path(logs_dir) / str(filename) if filename else None
    if path is not None and path.exists():
        try:
            return path, path.read_text(encoding="utf-8"), "file"
        except (OSError, UnicodeError):
            pass
    inline = entry.get(inline_key)
    if inline and str(inline).strip():
        return path, str(inline), "inline"
    return path, "", "none"


def _observe_video(
    path: Path, context: MediaContext, sample_count: int
) -> tuple[list[np.ndarray], list[float], int] | None:
    """Sample frames from a readable file, or return None for fallback."""
    if not path.exists():
        return None
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or context.fps
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, reported // max(sample_count, 1)) if reported else 1
        frames: list[np.ndarray] = []
        timestamps: list[float] = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                frames.append(cv2.resize(frame, (context.width, context.height)))
                timestamps.append(index / fps)
            index += 1
        if not frames:
            return None
        context.fps = fps
        return frames, timestamps, index
    except cv2.error as exc:  # pragma: no cover - defensive for codec issues
        context.probe_error = f"could not decode {path}: {exc}"
        return None
    finally:
        capture.release()


def _synthetic_observations(
    context: MediaContext, sample_count: int
) -> tuple[list[np.ndarray], list[float], int]:
    """Render deterministic stand-in frames for a not-yet-generated clip."""
    total = max(int(context.fps * context.duration_sec), sample_count)
    step = max(1, total // max(sample_count, 1))
    positions = list(range(0, total, step))[:sample_count]
    frames = [
        render_frame(
            context.scene,
            i,
            total,
            context.width,
            context.height,
            seed=context.video_id,
        )
        for i in positions
    ]
    return frames, [i / context.fps for i in positions], total


def _motion_energy(frames: list[np.ndarray]) -> float:
    """Return mean absolute difference between consecutive grayscale frames."""
    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    diffs = [
        float(np.mean(cv2.absdiff(grays[i], grays[i - 1])))
        for i in range(1, len(grays))
    ]
    return float(np.mean(diffs)) if diffs else 0.0


def _text_region_density(gray_frames: list[np.ndarray]) -> float:
    """Count text-like connected components normalised per 10k pixels."""
    if not gray_frames:
        return 0.0
    total = 0
    for gray in gray_frames:
        top = gray[: int(gray.shape[0] * 0.8), :]
        edges = cv2.Canny(top, 80, 180)
        dilated = cv2.dilate(edges, np.ones((5, 9), dtype=np.uint8))
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 9 and 4 <= h <= 40 and w * h >= 28:
                total += 1
    scale = max((gray_frames[0].shape[0] * gray_frames[0].shape[1]) / 10_000, 1.0)
    return total / (len(gray_frames) * scale)
