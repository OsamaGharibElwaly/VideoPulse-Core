"""Deterministic, lightweight synthetic frame rendering.

Shared by the sample-asset generator and by the media probe when a real
``.mp4`` is not yet present. No models, no downloads, no randomness that is
not seeded from the video id.
"""

from __future__ import annotations

import zlib

import cv2
import numpy as np

# BGR colours used across the synthetic scenes.
_BG = {
    "slides": (238, 238, 238),
    "demo": (84, 74, 62),
    "conveyor": (36, 34, 30),
    "assembly": (58, 54, 48),
    "aisle": (206, 206, 200),
    "facility": (188, 190, 192),
}

_SLIDE_TITLES = [
    "VIDEOPULSE LECTURE",
    "KEY TERM GRADIENT",
    "OPTICAL FLOW",
    "KEYFRAME SUMMARY",
]
_SLIDE_BULLETS = [
    "definition of slide transition",
    "visual state change",
    "compare diagram on slide",
    "high change frames",
]


def _rng(seed: str) -> np.random.Generator:
    """Create a reproducible generator from a string seed."""
    return np.random.default_rng(zlib.crc32(seed.encode("utf-8")))


def render_frame(
    scene: str,
    frame_index: int,
    total_frames: int,
    width: int,
    height: int,
    *,
    seed: str = "videopulse",
) -> np.ndarray:
    """Render one BGR frame for a content profile scene.

    Args:
        scene: One of the ``content_profile.scene`` values in the manifest.
        frame_index: Zero-based frame position.
        total_frames: Total frame count, used to derive normalised time.
        width: Frame width in pixels.
        height: Frame height in pixels.
        seed: Stable seed making the output reproducible per video.

    Returns:
        A ``uint8`` BGR frame of shape ``(height, width, 3)``.
    """
    frame = np.full((height, width, 3), _BG.get(scene, (128, 128, 128)), dtype=np.uint8)
    renderer = _RENDERERS.get(scene, _render_static)
    renderer(frame, frame_index, max(total_frames, 1), seed)
    return frame


def _render_slides(frame: np.ndarray, i: int, total: int, seed: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (8, 8), (w - 8, h - 34), (255, 255, 255), -1)
    cv2.rectangle(frame, (8, 8), (w - 8, h - 34), (90, 90, 90), 2)
    slide = min(i // max(total // 4, 1), len(_SLIDE_TITLES) - 1)
    cv2.putText(frame, _SLIDE_TITLES[slide], (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2, cv2.LINE_AA)
    for line_no, text in enumerate(_SLIDE_BULLETS):
        y = 70 + line_no * 26
        cv2.circle(frame, (26, y - 4), 3, (30, 30, 30), -1)
        cv2.putText(frame, text, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (35, 35, 35), 1, cv2.LINE_AA)
    # Rising contrast box plus a moving cursor gives measurable moderate motion.
    cv2.rectangle(frame, (24, 180), (24 + int(180 * i / total), 192), (210, 220, 235), -1)
    cursor_x = 30 + int((w - 70) * i / total)
    cv2.circle(frame, (cursor_x, 200), 5, (0, 0, 220), -1)


def _render_demo(frame: np.ndarray, i: int, total: int, seed: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (int(w * 0.25), int(h * 0.25)), (int(w * 0.8), int(h * 0.85)), (110, 100, 88), -1)
    cv2.rectangle(frame, (int(w * 0.25), int(h * 0.25)), (int(w * 0.8), int(h * 0.85)), (40, 40, 40), 2)
    cv2.putText(frame, "TOOL", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, "STEP", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
    phase = 2.0 * np.pi * i / max(total / 3.0, 1.0)
    tool_x = int(w * 0.5 + w * 0.2 * np.sin(phase))
    tool_y = int(h * 0.55 + h * 0.18 * np.cos(phase * 0.7))
    cv2.rectangle(frame, (tool_x - 18, tool_y - 10), (tool_x + 18, tool_y + 10), (60, 220, 240), -1)
    cv2.circle(frame, (tool_x, tool_y), 7, (245, 245, 245), -1)
    cv2.rectangle(frame, (int(w * 0.36), int(h * 0.62)), (int(w * 0.46), int(h * 0.72)), (190, 190, 195), -1)


def _render_conveyor(frame: np.ndarray, i: int, total: int, seed: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, int(h * 0.35)), (w, int(h * 0.75)), (52, 50, 46), -1)
    for y in range(int(h * 0.35), int(h * 0.75), 14):
        cv2.line(frame, (0, y), (w, y), (66, 64, 60), 1)
    progress = i / total
    defect_window = (0.23 <= progress <= 0.29) or (0.55 <= progress <= 0.61)
    for item in range(4):
        x = int(((i * 4 + item * (w // 3)) % (w + 60)) - 30)
        y = int(h * 0.55 + 3 * np.sin(i * 0.4 + item))
        if defect_window and item == 0:
            cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y + 20), (30, 30, 235), -1)
        else:
            cv2.circle(frame, (x, y), 13, (205, 205, 210), -1)


def _render_assembly(frame: np.ndarray, i: int, total: int, seed: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (int(w * 0.2), int(h * 0.3)), (int(w * 0.8), int(h * 0.8)), (90, 84, 74), -1)
    cv2.rectangle(frame, (int(w * 0.42), int(h * 0.5)), (int(w * 0.58), int(h * 0.62)), (170, 170, 175), -1)
    phase = (i % max(total // 6, 1)) / max(total // 6, 1)
    if phase < 0.6:
        offset = int(60 * np.sin(phase * np.pi / 0.6))
        cv2.ellipse(frame, (int(w * 0.5) - 40 + offset, int(h * 0.48)), (22, 13), 0, 0, 360, (196, 168, 140), -1)
        cv2.ellipse(frame, (int(w * 0.5) + 40 - offset, int(h * 0.52)), (22, 13), 0, 0, 360, (196, 168, 140), -1)


def _render_aisle(frame: np.ndarray, i: int, total: int, seed: str) -> None:
    h, w = frame.shape[:2]
    for shelf in range(3):
        x0 = 10 + shelf * (w // 3)
        x1 = x0 + (w // 3) - 16
        cv2.rectangle(frame, (x0, int(h * 0.18)), (x1, int(h * 0.78)), (168, 170, 165), -1)
        for row in range(4):
            y = int(h * 0.24) + row * 30
            cv2.line(frame, (x0, y), (x1, y), (120, 122, 118), 2)
            # Tiny marks stand in for unreadable price tags / signage.
            cv2.rectangle(frame, (x0 + 6, y + 4), (x0 + 14, y + 8), (90, 92, 88), -1)
    busy = total * 0.33 <= i <= total * 0.55
    count = 6 if busy else 3
    rng = _rng(f"{seed}:{i}")
    for person in range(count):
        x = int((i * (2 + person % 3) + person * 73) % w)
        y = int(h * (0.55 + 0.12 * ((person * 37) % 5) / 5.0))
        colour = (60, 70, 150) if person % 2 else (150, 70, 60)
        cv2.ellipse(frame, (x, y), (9, 18), 0, 0, 360, colour, -1)
        cv2.circle(frame, (x, y - 22), 6, (200, 170, 140), -1)
    _ = rng  # rng kept for future variants without changing current output


def _render_facility(frame: np.ndarray, i: int, total: int, seed: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (int(w * 0.35), int(h * 0.15)), (int(w * 0.65), int(h * 0.85)), (200, 202, 204), -1)
    cv2.rectangle(frame, (int(w * 0.65), int(h * 0.2)), (int(w * 0.95), int(h * 0.8)), (170, 172, 176), -1)
    cv2.rectangle(frame, (int(w * 0.35), int(h * 0.15)), (int(w * 0.65), int(h * 0.85)), (110, 112, 114), 2)
    # Very small sensor noise keeps the "static" camera honest without faking motion.
    noise = _rng(f"{seed}:facility:{i}").integers(-2, 3, size=frame.shape[:2], dtype=np.int16)
    frame[:] = np.clip(frame.astype(np.int16) + noise[:, :, None], 0, 255).astype(np.uint8)
    events = ((int(total * 0.33), 0.35, 0.65), (int(total * 0.75), 0.65, 0.95))
    for centre, x0, x1 in events:
        if abs(i - centre) <= max(total // 24, 2):
            progress = (i - centre + max(total // 24, 2)) / max(2 * max(total // 24, 2), 1)
            x = int(w * (x0 + (x1 - x0) * progress))
            cv2.ellipse(frame, (x, int(h * 0.6)), (10, 20), 0, 0, 360, (45, 45, 55), -1)
    if abs(i - int(total * 0.5)) <= 1:
        cv2.rectangle(frame, (0, 0), (w, h), (235, 240, 250), 1)


def _render_static(frame: np.ndarray, i: int, total: int, seed: str) -> None:
    noise = _rng(f"{seed}:{i}").integers(-1, 2, size=frame.shape[:2], dtype=np.int16)
    frame[:] = np.clip(frame.astype(np.int16) + noise[:, :, None], 0, 255).astype(np.uint8)


_RENDERERS = {
    "slides": _render_slides,
    "demo": _render_demo,
    "conveyor": _render_conveyor,
    "assembly": _render_assembly,
    "aisle": _render_aisle,
    "facility": _render_facility,
}


def render_clip(
    scene: str,
    frame_count: int,
    width: int,
    height: int,
    *,
    seed: str = "videopulse",
) -> list[np.ndarray]:
    """Render a whole clip as a list of frames."""
    return [
        render_frame(scene, i, frame_count, width, height, seed=seed)
        for i in range(frame_count)
    ]
