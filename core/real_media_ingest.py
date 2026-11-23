"""Real video and sidecar ingestion for VideoPulse Core Phase 2.

Handles common video containers and text/JSON/CSV event side-cars without any
heavy dependencies. Every failure is reported as an explicit status code so
callers can gate features instead of crashing.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

REAL_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".webm"}
)
SIDECAR_EXTENSIONS: frozenset[str] = frozenset(
    {".srt", ".vtt", ".txt", ".log", ".json", ".csv"}
)
TRANSCRIPT_EXTENSIONS: frozenset[str] = frozenset({".srt", ".vtt"})

_CLOCK_RE = re.compile(r"(\d+):(\d{2}):(\d{2})(?:[,.](\d+))?")
_SHORT_CLOCK_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?:[,.](\d+))?")
_OFFSET_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*s?\b", re.IGNORECASE)
_TIME_FIELD_NAMES = ("timestamp_sec", "timestamp", "time", "start_sec", "start", "offset", "ts", "t")
_TEXT_FIELD_NAMES = ("text", "message", "event", "claim", "description", "action", "label")


@dataclass(frozen=True)
class StreamProbe:
    """Metadata observed from a real video stream."""

    status: str
    reason: str = ""
    path: Path = field(default_factory=Path)
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    duration_sec: float = 0.0
    has_audio_stream: bool | None = None


@dataclass(frozen=True)
class RealPaths:
    """Resolved paths for one video and its optional side-cars."""

    video_path: Path
    transcript_path: Path | None = None
    log_path: Path | None = None
    transcript_source: str = "none"
    log_source: str = "none"


@dataclass
class SidecarBundle:
    """Canonicalised transcript/log text plus their original records."""

    transcript_text: str = ""
    log_text: str = ""
    transcript_source: str = "none"
    log_source: str = "none"
    transcript_path: Path | None = None
    log_path: Path | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    reason: str = ""


@dataclass
class RealIngest:
    """Result of ingesting one real video slot."""

    paths: RealPaths
    stream: StreamProbe
    sidecars: SidecarBundle
    frames: list[np.ndarray] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    status: str = "ok"
    reason: str = ""


def parse_timestamp(value: Any) -> float | None:
    """Parse ``HH:MM:SS,mmm``, ``MM:SS`` or a numeric epoch-relative offset.

    Numeric values (e.g. ``12.5`` or ``+12.5s``) are seconds from the start of
    the video. Returns ``None`` when no timestamp can be recognised.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace("-->", " ")
    if not text:
        return None
    clock = _CLOCK_RE.search(text)
    if clock:
        hours, minutes, seconds = (int(clock.group(i)) for i in (1, 2, 3))
        digits = clock.group(4)
        fraction = float(f"0.{digits}") if digits else 0.0
        return hours * 3600 + minutes * 60 + seconds + fraction
    short = _SHORT_CLOCK_RE.search(text)
    if short:
        digits = short.group(3)
        return int(short.group(1)) * 60 + int(short.group(2)) + (
            float(f"0.{digits}") if digits else 0.0
        )
    offset = _OFFSET_RE.search(text)
    if offset:
        return float(offset.group(1))
    return None


def _format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS,mmm`` for SRT-style canonical text."""
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _candidate_stems(entry: dict[str, Any]) -> set[str]:
    """Build the set of video filename stems an manifest entry may refer to."""
    stems: set[str] = set()
    for key in ("real_file", "file", "video_id", "short_id"):
        value = entry.get(key)
        if not value:
            continue
        path = Path(str(value))
        stems.add(path.stem.lower())
        stems.add(path.name.lower())
    return stems


def resolve_real_paths(entry: dict[str, Any], real_dir: Path) -> RealPaths | None:
    """Locate a real video and its side-cars inside ``real_dir``.

    Returns ``None`` when no matching video file exists; callers then fall
    back to the Phase 1 sample path instead of failing.
    """
    base = Path(real_dir)
    if not base.is_dir():
        return None
    stems = _candidate_stems(entry)
    videos = [
        path
        for path in base.rglob("*")
        if path.is_file()
        and path.suffix.lower() in REAL_VIDEO_EXTENSIONS
        and (path.stem.lower() in stems or path.name.lower() in stems)
    ]
    if not videos:
        return None
    video = sorted(videos, key=lambda path: (path.stem.lower() != str(entry.get("video_id")).lower(), str(path)))[0]

    sidecars = _discover_sidecars(entry, base, video.stem)
    return RealPaths(
        video_path=video,
        transcript_path=sidecars.get("transcript"),
        log_path=sidecars.get("log"),
        transcript_source=f"file:{sidecars['transcript'].name}" if sidecars.get("transcript") else "none",
        log_source=f"file:{sidecars['log'].name}" if sidecars.get("log") else "none",
    )


def _discover_sidecars(
    entry: dict[str, Any], real_dir: Path, video_stem: str
) -> dict[str, Path]:
    """Discover transcript/log side-cars by explicit field or matching stem."""
    found: dict[str, Path] = {}
    explicit = {
        "transcript": entry.get("real_transcript") or entry.get("transcript_file"),
        "log": entry.get("real_log") or entry.get("log_file"),
    }
    for kind, value in explicit.items():
        if not value:
            continue
        candidate = Path(real_dir) / str(value)
        if candidate.is_file():
            found[kind] = candidate
    for path in sorted(real_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SIDECAR_EXTENSIONS:
            continue
        if path.stem.lower() != video_stem.lower():
            continue
        kind = (
            "transcript"
            if path.suffix.lower() in TRANSCRIPT_EXTENSIONS
            or any(word in path.stem.lower() for word in ("transcript", "caption", "subtitle"))
            else "log"
        )
        found.setdefault(kind, path)
    return found


def probe_real_stream(path: Path) -> StreamProbe:
    """Probe an existing video file, returning an explicit status on failure."""
    video = Path(path)
    if not video.exists():
        return StreamProbe("missing", "REAL_VIDEO_MISSING", video)
    if video.suffix.lower() not in REAL_VIDEO_EXTENSIONS:
        return StreamProbe("unsupported", "UNSUPPORTED_VIDEO_FORMAT", video)
    try:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            capture.release()
            return StreamProbe("unreadable", "REAL_VIDEO_UNREADABLE", video)
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        audio: bool | None = None
        try:  # OpenCV exposes stream counts on recent builds only.
            total_audio = int(capture.get(cv2.CAP_PROP_AUDIO_TOTAL_STREAMS))
            audio = total_audio > 0
        except (AttributeError, cv2.error):
            audio = None
        ok, _ = capture.read()
        capture.release()
        if not ok:
            return StreamProbe("unreadable", "REAL_VIDEO_NO_DECODABLE_FRAME", video)
        duration = frame_count / fps if fps > 0 else 0.0
        return StreamProbe(
            status="ok",
            path=video,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_sec=round(duration, 3),
            has_audio_stream=audio,
        )
    except cv2.error as exc:  # pragma: no cover - codec-dependent guard
        return StreamProbe("unreadable", f"REAL_VIDEO_DECODE_ERROR: {exc}", video)


def read_real_frames(
    path: Path,
    probe: StreamProbe,
    sample_count: int,
    target_size: tuple[int, int] | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    """Sample frames evenly from a real stream, skipping undecodable ones."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return [], []
    step = max(1, probe.frame_count // max(sample_count, 1)) if probe.frame_count else 1
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index % step == 0:
                if target_size and (frame.shape[1], frame.shape[0]) != target_size:
                    frame = cv2.resize(frame, target_size)
                frames.append(frame)
                timestamps.append(index / probe.fps if probe.fps > 0 else 0.0)
            index += 1
    except cv2.error:  # pragma: no cover - truncated/corrupt stream guard
        pass
    finally:
        capture.release()
    return frames, timestamps


def _record_from_text(
    timestamp: float | None, text: str, source: str, *, end: float | None = None
) -> dict[str, Any]:
    """Create a normalised record with a derived timestamp when necessary."""
    return {
        "timestamp_sec": float(timestamp) if timestamp is not None else 0.0,
        "end_sec": float(end) if end is not None else None,
        "time_source": "explicit" if timestamp is not None else "derived",
        "text": text.strip(),
        "source": source,
    }


def _records_from_cues(text: str, source: str) -> list[dict[str, Any]]:
    """Parse SRT/VTT cues into timestamped records."""
    records: list[dict[str, Any]] = []
    current_start: float | None = None
    current_end: float | None = None
    lines: list[str] = []
    for raw_line in text.splitlines() + [""]:
        line = raw_line.strip()
        if "-->" in line:
            if current_start is not None and lines:
                records.append(_record_from_text(current_start, " ".join(lines), source, end=current_end))
            parts = line.split("-->", 1)
            current_start = parse_timestamp(parts[0])
            current_end = parse_timestamp(parts[1])
            # SRT/VTT may put the caption on the timing line itself.
            trailing = re.sub(
                r"^\s*\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d+)?", "", parts[1]
            ).strip()
            lines = [trailing] if trailing else []
        elif line and not line.isdigit() and not line.upper().startswith("WEBVTT"):
            lines.append(line)
        elif not line and current_start is not None and lines:
            records.append(_record_from_text(current_start, " ".join(lines), source, end=current_end))
            current_start, current_end, lines = None, None, []
    return records


def _records_from_lines(text: str, source: str) -> list[dict[str, Any]]:
    """Parse plain-text or .log lines, supporting missing timestamps."""
    records: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        timestamp = parse_timestamp(line)
        payload = line
        if timestamp is not None:
            clock = _CLOCK_RE.search(payload) or _SHORT_CLOCK_RE.search(payload)
            if clock:
                payload = payload[clock.end() :]
            payload = payload.lstrip(" ,:-")
        else:
            timestamp = float(line_no * 2)  # derived, explicitly flagged
        records.append(_record_from_text(timestamp, payload or line, source))
    return records


def _payload_from_mapping(mapping: dict[str, Any], timestamp: float) -> tuple[float, str]:
    """Convert a JSON/CSV record into a canonical ``key=value`` log line."""
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    when = None
    for key in _TIME_FIELD_NAMES:
        if key in lowered and lowered[key] not in (None, ""):
            when = parse_timestamp(lowered[key])
            if when is not None:
                break
    if when is None:
        when = timestamp
    skip = set(_TIME_FIELD_NAMES) | {"end", "end_sec", "duration"}
    tokens = []
    for key, value in mapping.items():
        if str(key).lower() in skip or value is None:
            continue
        cleaned = str(value).strip().replace("=", "-")
        if cleaned:
            tokens.append(f"{str(key).strip()}={cleaned}")
    if not tokens:
        for key in _TEXT_FIELD_NAMES:
            if key in lowered and lowered[key]:
                tokens.append(str(lowered[key]))
                break
    return when, " ".join(tokens)


def _records_from_json(text: str, source: str) -> list[dict[str, Any]]:
    """Parse a JSON array or an object with an events/records list."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _records_from_lines(text, source)
    if isinstance(payload, dict):
        payload = next(
            (payload[key] for key in ("events", "records", "segments", "logs") if isinstance(payload.get(key), list)),
            [payload],
        )
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload if isinstance(payload, list) else []):
        if isinstance(item, dict):
            when, line = _payload_from_mapping(item, float(index * 2))
            records.append(_record_from_text(when, line, source))
        else:
            records.append(_record_from_text(float(index * 2), str(item), source))
    return records


def _records_from_csv(text: str, source: str) -> list[dict[str, Any]]:
    """Parse CSV rows with a header into timestamped key=value lines."""
    try:
        rows = list(csv.DictReader(text.splitlines()))
    except csv.Error:
        return _records_from_lines(text, source)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        when, line = _payload_from_mapping(dict(row), float(index * 2))
        records.append(_record_from_text(when, line, source))
    return records


def parse_sidecar(text: str, suffix: str, source: str) -> list[dict[str, Any]]:
    """Dispatch sidecar parsing based on its extension."""
    extension = suffix.lower()
    if extension in TRANSCRIPT_EXTENSIONS:
        return _records_from_cues(text, source)
    if extension == ".json":
        return _records_from_json(text, source)
    if extension == ".csv":
        return _records_from_csv(text, source)
    return _records_from_lines(text, source)


def canonicalise(records: Iterable[dict[str, Any]]) -> tuple[str, str]:
    """Return canonical SRT-style transcript text and log-style text."""
    transcript_lines: list[str] = []
    log_lines: list[str] = []
    ordered = sorted(records, key=lambda record: record["timestamp_sec"])
    for record in ordered:
        start = record["timestamp_sec"]
        payload = record.get("text", "").strip()
        if not payload:
            continue
        end = record.get("end_sec")
        if record.get("source", "").endswith((".srt", ".vtt")) or end is not None:
            finish = float(end) if end is not None else start + 2.0
            transcript_lines.append(
                f"{_format_timestamp(start)} --> {_format_timestamp(finish)}\n{payload}\n"
            )
        else:
            log_lines.append(f"{_format_timestamp(start)} {payload}")
    return "\n".join(transcript_lines), "\n".join(log_lines)


def load_sidecars(paths: RealPaths, logs_dir: Path | None = None) -> SidecarBundle:
    """Read and canonicalise all available sidecar files for one video."""
    bundle = SidecarBundle()
    targets: list[tuple[str, str, Path | None]] = [
        ("transcript", "transcript", paths.transcript_path),
        ("log", "log", paths.log_path),
    ]
    if logs_dir:
        preferences = {
            "transcript": (".srt", ".vtt"),
            "log": (".json", ".csv", ".log", ".txt"),
        }
        for index, (field_name, _suffix, path) in enumerate(targets):
            if path is not None:
                continue
            for suffix in preferences[field_name]:
                candidate = Path(logs_dir) / f"{paths.video_path.stem}{suffix}"
                if candidate.is_file():
                    targets[index] = (field_name, suffix, candidate)
                    break
    for kind, _suffix, path in targets:
        if path is None:
            continue
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            bundle.status = "sidecar_unreadable"
            bundle.reason = f"SIDECAR_UNREADABLE: {path.name}: {exc}"
            continue
        records = parse_sidecar(text, Path(path).suffix, kind)
        if not records:
            continue
        bundle.records.extend(records)
        if kind == "transcript":
            bundle.transcript_text, _ = canonicalise(records)
            bundle.transcript_source = f"file:{Path(path).name}"
            bundle.transcript_path = Path(path)
        else:
            _, bundle.log_text = canonicalise(records)
            bundle.log_source = f"file:{Path(path).name}"
            bundle.log_path = Path(path)
    if not bundle.records and bundle.status == "ok":
        bundle.status = "no_sidecars"
        bundle.reason = "NO_TRANSCRIPT_OR_LOG"
    return bundle


def ingest_real(
    entry: dict[str, Any],
    real_dir: Path,
    logs_dir: Path,
    sample_count: int,
    target_size: tuple[int, int] | None = None,
) -> RealIngest | None:
    """Ingest one real-video slot; return ``None`` when no real file exists."""
    paths = resolve_real_paths(entry, Path(real_dir))
    if paths is None:
        return None
    stream = probe_real_stream(paths.video_path)
    if stream.status != "ok":
        return RealIngest(paths, stream, SidecarBundle(status=stream.status, reason=stream.reason), status=stream.status, reason=stream.reason)
    frames, timestamps = read_real_frames(paths.video_path, stream, sample_count, target_size)
    if not frames:
        reason = "REAL_VIDEO_NO_FRAMES"
        return RealIngest(paths, stream, SidecarBundle(status="no_frames", reason=reason), status="no_frames", reason=reason)
    sidecars = load_sidecars(paths, logs_dir)
    return RealIngest(paths, stream, sidecars, frames, timestamps)
