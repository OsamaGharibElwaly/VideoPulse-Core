"""Verify that the README's local evidence assets actually resolve."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


class TestReadmeAssets(unittest.TestCase):
    """Small guard against broken paths after reports are regenerated."""

    def test_all_local_readme_links_exist(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        checked = 0
        for raw in LINK_RE.findall(readme):
            target = raw.strip().split()[0]
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = (ROOT / target).resolve()
            self.assertTrue(path.is_file(), f"README link is missing: {target}")
            checked += 1
        self.assertGreater(checked, 10, "expected the embedded evidence links")

    def test_embedded_assets_are_readable_images(self) -> None:
        folders = [ROOT / "docs" / "figures", ROOT / "docs" / "screenshots"]
        images = [path for folder in folders for path in folder.glob("*.png")]
        self.assertEqual(len(images), 12, "unexpected number of generated assets")
        for path in images:
            self.assertGreater(path.stat().st_size, 1000, f"empty asset: {path}")
            image = cv2.imread(str(path))
            self.assertIsNotNone(image, f"unreadable asset: {path}")
            self.assertGreaterEqual(min(image.shape[:2]), 100)

    def test_result_videos_decode(self) -> None:
        videos = sorted((ROOT / "docs" / "videos").glob("*/*_cv_nlp.mp4"))
        self.assertEqual(len(videos), 10, "expected one result reel per input video")
        for path in videos:
            self.assertGreater(path.stat().st_size, 1000, f"empty result video: {path}")
            capture = cv2.VideoCapture(str(path))
            self.assertTrue(capture.isOpened(), f"cannot open: {path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.assertGreater(frame_count, 50, f"too short: {path}")
            capture.set(cv2.CAP_PROP_POS_MSEC, 1900)
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok, f"cannot decode frame: {path}")
            self.assertEqual(frame.shape[:2], (240, 320))


if __name__ == "__main__":
    unittest.main()
