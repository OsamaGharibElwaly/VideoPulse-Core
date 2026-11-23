"""Phase 2 tests: real ingestion, YOLO26 integration, gating and grounding.

No network or weights download is used. A stub YOLO result exercises the parse
path; the default configuration exercises the offline OpenCV fallback.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_router import CHECKS, FEATURE_ORDER, build_feature_plan  # noqa: E402
from core.fusion_and_eval import process_video  # noqa: E402
from core.media_probe import find_video_entry, load_manifest, probe_media  # noqa: E402
from core.nlp_pipeline import (  # noqa: E402
    entity_graph_linking,
    grounding_consistency_check,
    multimodal_bm25_tfidf_retrieval,
)
from core.real_media_ingest import (  # noqa: E402
    canonicalise,
    ingest_real,
    load_sidecars,
    parse_sidecar,
    parse_timestamp,
    probe_real_stream,
    resolve_real_paths,
)
from core.yolo_detector import DetectorConfig, YOLODetector, build_detector, resolve_weights  # noqa: E402
from run_pipeline import main as cli_main  # noqa: E402
from tools.generate_sample_assets import generate_assets  # noqa: E402

NAMES = {0: "person", 1: "part", 2: "car"}

VIDEO_IDS = (
    "edu_01_slide_lecture",
    "edu_02_hands_on_demo",
    "mfg_01_conveyor_inspection",
    "mfg_02_assembly_station",
    "ret_01_aisle_traffic",
    "ret_02_facility_monitoring",
)


class StubBoxes:
    """Minimal stand-in for ultralytics' Boxes result object."""

    def __init__(self, xyxy: list[list[float]], conf: list[float], cls: list[int], ids: list[int] | None = None) -> None:
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.conf = np.asarray(conf, dtype=float)
        self.cls = np.asarray(cls, dtype=int)
        self.id = None if ids is None else np.asarray(ids, dtype=int)


class StubResult:
    """Minimal stand-in for one ultralytics result."""

    def __init__(self, boxes: StubBoxes, names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


class StubYOLOModel:
    """Deterministic YOLO stub; one confident 'person' with a track id."""

    def __init__(self, *, fail_calls: int = 0, track: bool = False) -> None:
        self.names = NAMES
        self.calls = 0
        self.fail_calls = fail_calls
        self.track_enabled = track

    def _result(self) -> list[StubResult]:
        boxes = StubBoxes([[40.0, 40.0, 120.0, 120.0]], [0.92], [0], [7])
        return [StubResult(boxes, NAMES)]

    def predict(self, frame: Any, **kwargs: Any) -> list[StubResult]:
        self.calls += 1
        if self.calls <= self.fail_calls:
            raise RuntimeError("stub inference failure")
        return self._result()

    def track(self, frame: Any, **kwargs: Any) -> list[StubResult]:
        self.calls += 1
        return self._result()


def stub_detector(**kwargs: Any) -> YOLODetector:
    """Build a detector backed by the stub YOLO model."""
    return YOLODetector(DetectorConfig(), model=StubYOLOModel(**kwargs))


def sample_frames(count: int = 4) -> list[np.ndarray]:
    """Return tiny deterministic frames with a moving bright object."""
    frames = []
    for index in range(count):
        frame = np.full((60, 80, 3), 30, dtype=np.uint8)
        x = 4 + index * 12
        cv2.rectangle(frame, (x, 12), (x + 16, 28), (240, 240, 240), -1)
        frames.append(frame)
    return frames


class Phase2Base(unittest.TestCase):
    """Create real-format clips plus side-cars once for the module."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_manifest(ROOT / "config" / "videos_manifest.json")
        cls.temp = tempfile.TemporaryDirectory()
        base = Path(cls.temp.name)
        cls.real_dir = base / "real"
        cls.other_dir = base / "other"
        cls.output_dir = base / "outputs"
        generate_assets(cls.config, videos_dir=cls.real_dir, logs_dir=cls.real_dir)

        # A directory holding only a video: no transcript, no log.
        cls.no_sidecar_dir = base / "no_sidecars"
        cls.no_sidecar_dir.mkdir(parents=True)
        shutil.copy(
            cls.real_dir / "edu_01_slide_lecture.mp4",
            cls.no_sidecar_dir / "edu_01_slide_lecture.mp4",
        )

        # A corrupt clip with the right name, plus intact Phase 1 samples.
        cls.corrupt_dir = base / "corrupt"
        cls.sample_videos = base / "phase1_videos"
        cls.sample_logs = base / "phase1_logs"
        cls.corrupt_dir.mkdir(parents=True)
        cls.sample_videos.mkdir(parents=True)
        cls.sample_logs.mkdir(parents=True)
        (cls.corrupt_dir / "mfg_01_conveyor_inspection.mp4").write_text(
            "this is not a video", encoding="utf-8"
        )
        shutil.copy(
            cls.real_dir / "mfg_01_conveyor_inspection.mp4",
            cls.sample_videos / "mfg_01_conveyor_inspection.mp4",
        )
        shutil.copy(
            cls.real_dir / "mfg_01_conveyor_inspection.log",
            cls.sample_logs / "mfg_01_conveyor_inspection.log",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()


class TestRealMediaIngestion(Phase2Base):
    """Real streams, side-cars and honest status codes."""

    def test_parse_timestamp_formats(self) -> None:
        self.assertEqual(parse_timestamp("00:01:03,250"), 63.25)
        self.assertEqual(parse_timestamp("00:07.500 -->"), 7.5)
        self.assertAlmostEqual(float(parse_timestamp("+12.5s")), 12.5)
        self.assertEqual(parse_timestamp(4.0), 4.0)
        self.assertIsNone(parse_timestamp("no time here"))

    def test_resolve_real_paths_covers_all_six_slots(self) -> None:
        for video_id in VIDEO_IDS:
            entry = find_video_entry(self.config, video_id)
            paths = resolve_real_paths(entry, self.real_dir)
            with self.subTest(video=video_id):
                self.assertIsNotNone(paths)
                assert paths is not None
                self.assertTrue(paths.video_path.is_file())
                self.assertTrue(paths.transcript_path or paths.log_path)
        # Other supported container names resolve by the same stem.
        (self.other_dir / "edu_01_slide_lecture.mkv").parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            self.real_dir / "edu_01_slide_lecture.mp4",
            self.other_dir / "edu_01_slide_lecture.mkv",
        )
        mkv = resolve_real_paths(find_video_entry(self.config, "edu_01"), self.other_dir)
        self.assertIsNotNone(mkv)
        assert mkv is not None
        self.assertEqual(mkv.video_path.suffix, ".mkv")

    def test_sidecar_parsers_cover_srt_vtt_json_csv_log(self) -> None:
        srt = "1\n00:00:01,000 --> 00:00:03,000\nDEFECT on belt\n"
        transcript, _ = canonicalise(parse_sidecar(srt, ".srt", "transcript"))
        self.assertIn("DEFECT on belt", transcript)
        vtt = "WEBVTT\n\n00:00:02.000 --> 00:00:04.000\nAlert in bay\n"
        transcript, _ = canonicalise(parse_sidecar(vtt, ".vtt", "transcript"))
        self.assertIn("Alert in bay", transcript)
        payload = json.dumps({"events": [{"time": "00:03", "status": "DEFECT", "part": "P1"}]})
        _, logs = canonicalise(parse_sidecar(payload, ".json", "log"))
        self.assertIn("status=DEFECT", logs)
        csv_text = "time,status,zone\n00:05,ALERT,doorway\n"
        _, logs = canonicalise(parse_sidecar(csv_text, ".csv", "log"))
        self.assertIn("zone=doorway", logs)
        _, logs = canonicalise(parse_sidecar("00:07 status=NORMAL\n", ".log", "log"))
        self.assertIn("status=NORMAL", logs)

    def test_probe_stream_status_codes(self) -> None:
        missing = probe_real_stream(self.real_dir / "absent.mp4")
        self.assertEqual(missing.status, "missing")
        self.assertEqual(missing.reason, "REAL_VIDEO_MISSING")
        wrong = self.real_dir / "notes.txt"
        wrong.write_text("hello", encoding="utf-8")
        unsupported = probe_real_stream(wrong)
        self.assertEqual(unsupported.reason, "UNSUPPORTED_VIDEO_FORMAT")
        corrupt = probe_real_stream(self.corrupt_dir / "mfg_01_conveyor_inspection.mp4")
        self.assertEqual(corrupt.status, "unreadable")
        good = probe_real_stream(self.real_dir / "mfg_01_conveyor_inspection.mp4")
        self.assertEqual(good.status, "ok")
        self.assertGreater(good.fps, 0.0)
        self.assertGreater(good.frame_count, 0)
        self.assertAlmostEqual(good.duration_sec, 12.0, delta=2.0)

    def test_ingest_real_reads_frames_and_sidecars(self) -> None:
        entry = find_video_entry(self.config, "mfg_01_conveyor_inspection")
        result = ingest_real(entry, self.real_dir, self.real_dir, sample_count=8)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.frames)
        self.assertTrue(result.timestamps)
        self.assertEqual(result.timestamps, sorted(result.timestamps))
        self.assertTrue(result.sidecars.log_text)
        self.assertIn("DEFECT", result.sidecars.log_text)

    def test_inline_srt_cues_from_real_sidecar_are_kept(self) -> None:
        entry = find_video_entry(self.config, "edu_01_slide_lecture")
        paths = resolve_real_paths(entry, self.real_dir)
        assert paths is not None
        bundle = load_sidecars(paths, self.real_dir)
        self.assertEqual(bundle.status, "ok")
        self.assertEqual(len(bundle.records), 5)
        self.assertIn("slide_transition", bundle.transcript_text)
        self.assertTrue(bundle.transcript_source.startswith("file:"))

    def test_no_sidecar_is_reported_not_invented(self) -> None:
        entry = find_video_entry(self.config, "edu_01_slide_lecture")
        result = ingest_real(entry, self.no_sidecar_dir, self.no_sidecar_dir, sample_count=4)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.sidecars.status, "no_sidecars")
        self.assertEqual(result.sidecars.reason, "NO_TRANSCRIPT_OR_LOG")
        self.assertEqual(result.sidecars.transcript_text, "")
        self.assertEqual(result.sidecars.log_text, "")

    def test_probe_media_prefers_real_and_drops_inline_sample(self) -> None:
        entry = find_video_entry(self.config, "edu_01_slide_lecture")
        context = probe_media(
            entry, videos_dir=self.sample_videos, logs_dir=self.sample_logs, real_dir=self.no_sidecar_dir
        )
        self.assertTrue(context.use_real)
        self.assertEqual(context.source, "real_file")
        self.assertEqual(context.ingest_status, "ok")
        self.assertFalse(context.transcript_available)
        self.assertEqual(context.transcript_source, "none")
        self.assertNotIn("We define", context.transcript_text)


class TestYOLOIntegration(Phase2Base):
    """Stubbed YOLO parsing and strict offline fallback."""

    def test_stub_yolo_detections_are_normalised(self) -> None:
        detector = stub_detector()
        result = detector.detect(
            sample_frames(),
            [0.0, 1.0, 2.0, 3.0],
            width=320,
            height=240,
            rois=[{"zone_id": "top_left", "coords": [0, 0, 0.5, 0.5]}],
        )
        self.assertEqual(result.mode, "yolo")
        self.assertEqual(result.reason, "injected_test_model")
        self.assertEqual(len(result.detections), 4)
        first = result.detections[0]
        self.assertEqual(first["label"], "person")
        self.assertEqual(first["track_id"], 7)
        self.assertEqual(first["zone_id"], "top_left")
        self.assertAlmostEqual(float(first["confidence"]), 0.92)
        self.assertEqual(first["bbox"], [40.0, 40.0, 80.0, 80.0])
        self.assertEqual(result.frames_analyzed, 4)

    def test_tracking_mode_uses_track(self) -> None:
        model = StubYOLOModel(track=True)
        detector = YOLODetector(DetectorConfig(use_tracking=True), model=model)
        detector.detect(sample_frames(2), [0.0, 1.0], width=80, height=60)
        self.assertEqual(model.calls, 2)

    def test_one_failed_frame_does_not_kill_detection(self) -> None:
        detector = stub_detector(fail_calls=1)
        result = detector.detect(sample_frames(3), [0.0, 1.0, 2.0], width=80, height=60)
        self.assertEqual(result.mode, "yolo")
        self.assertEqual(len(result.detections), 2)

    def test_offline_fallback_never_downloads_weights(self) -> None:
        detector = build_detector(DetectorConfig(model="missing_weights_xyz.pt"))
        self.assertEqual(detector.mode, "opencv_fallback")
        self.assertEqual(detector.reason, "YOLO_WEIGHTS_UNAVAILABLE_OFFLINE")
        result = detector.detect(sample_frames(4), [0.0, 1.0, 2.0, 3.0], width=80, height=60)
        self.assertEqual(result.mode, "opencv_fallback")
        self.assertTrue(result.detections)
        self.assertTrue(all(item["source"] == "opencv_fallback" for item in result.detections))

    def test_resolve_weights_finds_local_file(self) -> None:
        weights_dir = self.real_dir.parent / "weights"
        weights_dir.mkdir(exist_ok=True)
        local = weights_dir / "local_model.pt"
        local.write_bytes(b"stub")
        self.assertEqual(resolve_weights("local_model.pt", weights_dir), local)
        self.assertIsNone(resolve_weights("not_here.pt", weights_dir))

    def test_auto_zones_created_from_detections(self) -> None:
        result = stub_detector().detect(
            sample_frames(1), [0.0], width=320, height=240, rois=[], auto_zones=True
        )
        self.assertEqual(len(result.zones), 6)
        self.assertTrue(result.zones[0]["zone_id"].startswith("auto_r"))
        self.assertTrue(result.detections[0]["zone_id"].startswith("auto_r"))


class TestGatingAndGrounding(Phase2Base):
    """Content-aware skips, object-aware retrieval, graph and grounding."""

    def test_all_ten_can_run_checks_with_real_context(self) -> None:
        entry = find_video_entry(self.config, "ret_02_facility_monitoring")
        context = probe_media(entry, videos_dir=self.sample_videos, logs_dir=self.real_dir, real_dir=self.real_dir)
        self.assertEqual(set(CHECKS), set(FEATURE_ORDER))
        for feature in FEATURE_ORDER:
            check, category = CHECKS[feature]
            outcome = check(context)
            with self.subTest(feature=feature):
                self.assertIsInstance(outcome, tuple)
                self.assertEqual(len(outcome), 2)
                self.assertIsInstance(outcome[0], bool)
                self.assertIsInstance(outcome[1], str)
                self.assertTrue(outcome[1])
                self.assertIn(category, {"cv", "nlp"})

    def test_missing_sidecar_skips_transcript_alignment(self) -> None:
        entry = find_video_entry(self.config, "edu_01_slide_lecture")
        context = probe_media(entry, videos_dir=self.sample_videos, logs_dir=self.sample_logs, real_dir=self.no_sidecar_dir)
        context.detected_zone_ids = ["auto_r0c0"]
        skipped = {item["feature"]: item["reason"] for item in build_feature_plan(context).skipped}
        self.assertEqual(skipped.get("transcript_log_alignment"), "NO_AUDIO_TRANSCRIPT")
        self.assertIn("entity_graph_linking", build_feature_plan(context).active)

    def test_detected_zones_enable_roi_feature(self) -> None:
        config = json.loads(json.dumps(self.config))
        for entry in config["videos"]:
            if entry["video_id"] == "edu_01_slide_lecture":
                entry["content_profile"]["auto_zones"] = True
        report = process_video(
            "edu_01",
            config=config,
            videos_dir=self.sample_videos,
            logs_dir=self.sample_logs,
            real_dir=self.no_sidecar_dir,
            output_dir=self.output_dir,
            detector=stub_detector(),
            query="person",
        )
        self.assertIn("roi_zone_tracking", report["executed_features"])
        self.assertTrue(report["zones"])
        self.assertTrue(all(zone["zone_id"].startswith("auto_r") for zone in report["zones"]))

    def test_yolo_labels_enrich_retrieval_and_graph(self) -> None:
        context = probe_media(
            find_video_entry(self.config, "ret_02_facility_monitoring"),
            videos_dir=self.sample_videos,
            logs_dir=self.real_dir,
            real_dir=self.real_dir,
        )
        detections = [
            {
                "label": "person",
                "confidence": 0.9,
                "timestamp_sec": 4.0,
                "zone_id": "doorway",
            }
        ]
        results, _ = multimodal_bm25_tfidf_retrieval(
            context, [], {"object_detections": detections}, "person doorway"
        )
        self.assertTrue(results)
        self.assertIn("person", results[0]["text"])
        events = [
            {
                "event_id": "log_L1",
                "timestamp_sec": 4.0,
                "claim": "person near doorway",
                "raw_text": "person near doorway",
                "entity": "",
                "action": "",
                "status": "ALERT",
                "zone": "doorway",
                "defect": "",
            }
        ]
        edges, _ = entity_graph_linking(context, [], events, detections)
        relations = {edge["relation"] for edge in edges}
        self.assertIn("DETECTED_IN", relations)
        self.assertIn("SUPPORTED_BY_OBJECT", relations)
        self.assertTrue(any(edge["target"] == "object:person" for edge in edges))

    def test_grounding_uses_object_evidence_and_rejects_bad_zone(self) -> None:
        context = probe_media(
            find_video_entry(self.config, "ret_02_facility_monitoring"),
            videos_dir=self.sample_videos,
            logs_dir=self.real_dir,
            real_dir=self.real_dir,
        )
        detections = [
            {"label": "person", "confidence": 0.9, "timestamp_sec": 4.0, "zone_id": "doorway"}
        ]
        cv_results = {
            "keyframes": [{"timestamp_sec": 4.0}],
            "object_detections": detections,
        }
        events = [
            {
                "event_id": "ok",
                "timestamp_sec": 4.0,
                "claim": "person at doorway",
                "zone": "doorway",
            },
            {
                "event_id": "late",
                "timestamp_sec": 99.0,
                "claim": "person at doorway",
                "zone": "doorway",
            },
        ]
        checks, stats = grounding_consistency_check(context, events, cv_results)
        by_id = {check["event_id"]: check for check in checks}
        self.assertEqual(by_id["ok"]["status"], "GROUNDED")
        self.assertIn("object:person", by_id["ok"]["supporting_evidence"])
        self.assertEqual(by_id["late"]["status"], "UNGROUNDED_CLAIM")
        self.assertAlmostEqual(float(stats["grounding_match_rate"]), 0.5, places=3)

        # With only the zoned detection available, a different claim zone
        # must not be credited with that object evidence.
        wrong_zone = [
            {
                "event_id": "wrong_zone",
                "timestamp_sec": 4.0,
                "claim": "person at doorway",
                "zone": "restricted_bay",
            }
        ]
        zoned_checks, _ = grounding_consistency_check(
            context, wrong_zone, {"object_detections": detections}
        )
        self.assertEqual(zoned_checks[0]["status"], "UNGROUNDED_CLAIM")


class TestEndToEndPhase2(Phase2Base):
    """Reports, telemetry and CLI verification."""

    def test_process_video_real_with_stub_yolo(self) -> None:
        report = process_video(
            "mfg_01",
            config=self.config,
            videos_dir=self.sample_videos,
            logs_dir=self.real_dir,
            real_dir=self.real_dir,
            output_dir=self.output_dir,
            detector=stub_detector(),
            query="defect person",
        )
        self.assertEqual(report["media"]["source"], "real_file")
        self.assertTrue(report["media"]["use_real"])
        self.assertEqual(report["detector"]["mode"], "yolo")
        telemetry = report["telemetry"]
        self.assertEqual(telemetry["detector_mode"], "yolo")
        self.assertEqual(telemetry["detector_reason"], "injected_test_model")
        self.assertGreater(telemetry["object_detections"], 0)
        self.assertIn("motion_keyframe_extraction", report["executed_features"])
        path = self.output_dir / report["video_id"] / "report.json"
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["video_id"], report["video_id"])

    def test_corrupt_real_video_falls_back_to_phase1(self) -> None:
        report = process_video(
            "mfg_01",
            config=self.config,
            videos_dir=self.sample_videos,
            logs_dir=self.sample_logs,
            real_dir=self.corrupt_dir,
            output_dir=self.output_dir,
            model="__missing_weights__.pt",
            query="defect",
        )
        self.assertEqual(report["media"]["source"], "file")
        self.assertEqual(report["media"]["ingest_status"], "unreadable")
        self.assertFalse(report["media"]["use_real"])
        self.assertEqual(report["detector"]["mode"], "opencv_fallback")
        self.assertTrue(report["executed_features"])

    def test_cli_accepts_real_dir_and_model(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli_main(
                [
                    "--video-id",
                    "ret_02",
                    "--real-dir",
                    str(self.real_dir),
                    "--model",
                    "yolo26n.pt",
                    "--query",
                    "alert",
                ]
            )
        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("detector=", output)
        self.assertIn("ret_02_facility_monitoring", output)


if __name__ == "__main__":
    unittest.main()
