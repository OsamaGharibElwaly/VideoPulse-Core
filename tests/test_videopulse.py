"""Deterministic unit and integration tests for VideoPulse Core."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_router import CHECKS, FEATURE_ORDER, build_feature_plan  # noqa: E402
from core.fusion_and_eval import process_video  # noqa: E402
from core.media_probe import find_video_entry, load_manifest, probe_media  # noqa: E402
from core.nlp_pipeline import grounding_consistency_check  # noqa: E402
from tools.generate_sample_assets import generate_assets  # noqa: E402

QUERY = "defect alert slide step"

EXPECTED_GATING: dict[str, tuple[set[str], dict[str, str]]] = {
    "edu_01_slide_lecture": (
        {
            "motion_keyframe_extraction",
            "visual_anomaly_scoring",
            "on_screen_text_detection",
            "action_state_indexing",
            "transcript_log_alignment",
            "multimodal_bm25_tfidf_retrieval",
            "rule_schema_event_parsing",
            "entity_graph_linking",
            "grounding_consistency_check",
        },
        {"roi_zone_tracking": "NO_SPATIAL_ZONES"},
    ),
    "edu_02_hands_on_demo": (
        {
            "motion_keyframe_extraction",
            "roi_zone_tracking",
            "on_screen_text_detection",
            "action_state_indexing",
            "transcript_log_alignment",
            "multimodal_bm25_tfidf_retrieval",
            "rule_schema_event_parsing",
            "entity_graph_linking",
        },
        {
            "visual_anomaly_scoring": "NO_STABLE_BASELINE",
            "grounding_consistency_check": "INSUFFICIENT_VISUAL_LABELS",
        },
    ),
    "mfg_01_conveyor_inspection": (
        {
            "motion_keyframe_extraction",
            "roi_zone_tracking",
            "visual_anomaly_scoring",
            "action_state_indexing",
            "transcript_log_alignment",
            "multimodal_bm25_tfidf_retrieval",
            "rule_schema_event_parsing",
            "entity_graph_linking",
            "grounding_consistency_check",
        },
        {"on_screen_text_detection": "NO_READABLE_TEXT"},
    ),
    "mfg_02_assembly_station": (
        {
            "motion_keyframe_extraction",
            "roi_zone_tracking",
            "action_state_indexing",
            "transcript_log_alignment",
            "multimodal_bm25_tfidf_retrieval",
            "rule_schema_event_parsing",
            "entity_graph_linking",
            "grounding_consistency_check",
        },
        {
            "visual_anomaly_scoring": "HIGH_MOTION_VARIANCE",
            "on_screen_text_detection": "NO_TEXT_REGION",
        },
    ),
    "ret_01_aisle_traffic": (
        {
            "motion_keyframe_extraction",
            "roi_zone_tracking",
            "action_state_indexing",
            "transcript_log_alignment",
            "multimodal_bm25_tfidf_retrieval",
            "rule_schema_event_parsing",
            "entity_graph_linking",
            "grounding_consistency_check",
        },
        {
            "visual_anomaly_scoring": "DYNAMIC_SCENE_BASELINE",
            "on_screen_text_detection": "NO_LEGIBLE_SIGNAGE",
        },
    ),
    "ret_02_facility_monitoring": (
        {
            "motion_keyframe_extraction",
            "roi_zone_tracking",
            "visual_anomaly_scoring",
            "action_state_indexing",
            "transcript_log_alignment",
            "multimodal_bm25_tfidf_retrieval",
            "rule_schema_event_parsing",
            "entity_graph_linking",
            "grounding_consistency_check",
        },
        {"on_screen_text_detection": "NO_TEXT_REGION"},
    ),
}

REPORT_FIELDS = (
    "video_id",
    "industry",
    "duration_sec",
    "processed_fps",
    "executed_features",
    "skipped_features",
    "keyframes",
    "zones",
    "anomalies",
    "text_regions",
    "action_states",
    "aligned_segments",
    "parsed_events",
    "graph_edges",
    "grounding_checks",
    "retrieval_results",
    "telemetry",
)


class VideoPulseTestBase(unittest.TestCase):
    """Generate local assets once and process all six videos once."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_manifest(ROOT / "config" / "videos_manifest.json")
        cls.temp = tempfile.TemporaryDirectory()
        base = Path(cls.temp.name)
        cls.videos_dir = base / "videos"
        cls.logs_dir = base / "logs"
        cls.output_dir = base / "outputs"
        generate_assets(cls.config, videos_dir=cls.videos_dir, logs_dir=cls.logs_dir)
        cls.reports: dict[str, dict[str, Any]] = {
            str(entry["video_id"]): process_video(
                str(entry["video_id"]),
                config=cls.config,
                videos_dir=cls.videos_dir,
                logs_dir=cls.logs_dir,
                output_dir=cls.output_dir,
                query=QUERY,
            )
            for entry in cls.config["videos"]
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()


class TestManifestAndGating(VideoPulseTestBase):
    """Configuration, preconditions and skip-code behaviour."""

    def test_manifest_covers_six_videos_three_industries(self) -> None:
        videos = self.config["videos"]
        self.assertEqual(len(videos), 6)
        self.assertEqual(len({video["industry"] for video in videos}), 3)
        self.assertEqual(len({video["short_id"] for video in videos[:1]}), 1)
        self.assertEqual(
            [video["short_id"] for video in videos],
            ["edu_01", "edu_02", "mfg_01", "mfg_02", "ret_01", "ret_02"],
        )

    def test_all_ten_features_have_can_run_checks(self) -> None:
        self.assertEqual(len(FEATURE_ORDER), 10)
        self.assertEqual(set(CHECKS), set(FEATURE_ORDER))
        entry = find_video_entry(self.config, "mfg_01_conveyor_inspection")
        context = probe_media(entry, videos_dir=self.videos_dir, logs_dir=self.logs_dir)
        for feature in FEATURE_ORDER:
            check, category = CHECKS[feature]
            result = check(context)
            self.assertIsInstance(result, tuple, feature)
            self.assertEqual(len(result), 2, feature)
            self.assertIsInstance(result[0], bool, feature)
            self.assertIsInstance(result[1], str, feature)
            self.assertTrue(result[1], feature)
            self.assertIn(category, {"cv", "nlp"})

    def test_gating_matches_prd_for_all_six_videos(self) -> None:
        for video_id, (expected_active, expected_skipped) in EXPECTED_GATING.items():
            entry = find_video_entry(self.config, video_id)
            context = probe_media(entry, videos_dir=self.videos_dir, logs_dir=self.logs_dir)
            plan = build_feature_plan(context)
            with self.subTest(video=video_id):
                self.assertEqual(set(plan.active), expected_active)
                actual_skipped = {item["feature"]: item["reason"] for item in plan.skipped}
                self.assertEqual(actual_skipped, expected_skipped)
                self.assertTrue(all(item["status"] == "SKIPPED" for item in plan.skipped))

    def test_missing_media_is_reported_not_crashed(self) -> None:
        entry = find_video_entry(self.config, "edu_01_slide_lecture")
        context = probe_media(
            entry,
            videos_dir=Path(self.temp.name) / "does_not_exist",
            logs_dir=self.logs_dir,
        )
        self.assertFalse(context.media_available)
        self.assertEqual(context.source, "synthetic_fallback")
        self.assertIn("not found", context.probe_error)
        self.assertGreater(context.motion_energy, 0.0)
        self.assertEqual(len(build_feature_plan(context).active + build_feature_plan(context).skipped), 10)


class TestFeatureOutputs(VideoPulseTestBase):
    """Every active feature must produce honest, usable output."""

    def test_cv_features_produce_outputs(self) -> None:
        expectations = {
            "edu_01_slide_lecture": ("keyframes", "anomalies", "text_regions", "action_states"),
            "ret_01_aisle_traffic": ("zones", "keyframes", "action_states"),
            "mfg_01_conveyor_inspection": ("anomalies", "zones"),
            "ret_02_facility_monitoring": ("anomalies", "zones"),
        }
        for video_id, fields in expectations.items():
            report = self.reports[video_id]
            for field in fields:
                with self.subTest(video=video_id, field=field):
                    self.assertTrue(report[field], f"{video_id}.{field} is empty")
        self.assertEqual(len(self.reports["ret_01_aisle_traffic"]["zones"]), 3)

    def test_nlp_features_produce_outputs(self) -> None:
        for video_id in EXPECTED_GATING:
            report = self.reports[video_id]
            with self.subTest(video=video_id):
                self.assertTrue(report["aligned_segments"])
                self.assertTrue(report["parsed_events"])
                self.assertTrue(report["graph_edges"])
        mfg_report = self.reports["mfg_01_conveyor_inspection"]
        self.assertTrue(mfg_report["grounding_checks"])
        self.assertIn("DEFECT", str(mfg_report["parsed_events"][1]["status"]))

    def test_action_states_use_profile_vocabulary(self) -> None:
        for video_id, expected in (
            ("mfg_01_conveyor_inspection", {"IDLE", "MOVING", "BURST"}),
            ("ret_01_aisle_traffic", {"QUIET", "FLOW", "CONGESTION"}),
        ):
            states = {state["state"] for state in self.reports[video_id]["action_states"]}
            with self.subTest(video=video_id):
                self.assertTrue(states)
                self.assertTrue(states.issubset(expected))

    def test_srt_segments_are_not_merged(self) -> None:
        report = self.reports["edu_01_slide_lecture"]
        self.assertEqual(len(report["aligned_segments"]), 5)
        self.assertEqual(len(report["parsed_events"]), 5)


class TestGrounding(VideoPulseTestBase):
    """Cross-modal grounding tags and mismatch detection."""

    def test_grounding_detects_supported_and_unsupported_claims(self) -> None:
        cv_results = {
            "keyframes": [
                {"timestamp_sec": 1.0},
                {"timestamp_sec": 1.5},
            ],
            "anomalies": [{"timestamp_sec": 1.2}],
            "zones": [],
            "text_regions": [],
            "action_states": [],
        }
        events = [
            {
                "event_id": "supported",
                "timestamp_sec": 1.0,
                "claim": "defect observed at zone",
                "zone": "",
            },
            {
                "event_id": "impossible",
                "timestamp_sec": 99.0,
                "claim": "defect observed after the video ended",
                "zone": "",
            },
        ]
        context = find_video_entry(self.config, "mfg_01_conveyor_inspection")
        probe = probe_media(context, videos_dir=self.videos_dir, logs_dir=self.logs_dir)
        checks, stats = grounding_consistency_check(probe, events, cv_results)
        statuses = {check["event_id"]: check["status"] for check in checks}
        self.assertEqual(statuses["supported"], "GROUNDED")
        self.assertEqual(statuses["impossible"], "UNGROUNDED_CLAIM")
        self.assertAlmostEqual(stats["grounding_match_rate"], 0.5)

    def test_grounding_partial_when_only_one_signal_present(self) -> None:
        cv_results = {
            "keyframes": [{"timestamp_sec": 4.0}],
            "zones": [],
            "anomalies": [],
            "text_regions": [],
            "action_states": [],
        }
        events = [
            {
                "event_id": "partial",
                "timestamp_sec": 4.0,
                "claim": "alert",
                "zone": "",
            }
        ]
        probe = probe_media(
            find_video_entry(self.config, "ret_02_facility_monitoring"),
            videos_dir=self.videos_dir,
            logs_dir=self.logs_dir,
        )
        checks, stats = grounding_consistency_check(probe, events, cv_results)
        self.assertEqual(checks[0]["status"], "PARTIAL_GROUNDING")
        self.assertAlmostEqual(stats["grounding_match_rate"], 0.5)

    def test_zone_mismatch_does_not_count_as_grounding(self) -> None:
        cv_results = {
            "keyframes": [],
            "zones": [{"zone_id": "doorway", "active_timestamps": [4.0], "peak_timestamp_sec": 4.0}],
            "anomalies": [],
            "text_regions": [],
            "action_states": [],
        }
        events = [
            {
                "event_id": "wrong_zone",
                "timestamp_sec": 4.0,
                "claim": "alert in restricted bay",
                "zone": "restricted_bay",
            }
        ]
        probe = probe_media(
            find_video_entry(self.config, "ret_02_facility_monitoring"),
            videos_dir=self.videos_dir,
            logs_dir=self.logs_dir,
        )
        checks, _ = grounding_consistency_check(probe, events, cv_results)
        self.assertEqual(checks[0]["status"], "UNGROUNDED_CLAIM")


class TestEndToEnd(VideoPulseTestBase):
    """Full reports, telemetry, retrieval and persisted JSON."""

    def test_reports_have_schema_and_expected_gating(self) -> None:
        for video_id, (expected_active, expected_skipped) in EXPECTED_GATING.items():
            report = self.reports[video_id]
            with self.subTest(video=video_id):
                for field in REPORT_FIELDS:
                    self.assertIn(field, report)
                self.assertEqual(set(report["executed_features"]), expected_active)
                actual_skipped = {
                    item["feature"]: item["reason"] for item in report["skipped_features"]
                }
                self.assertEqual(actual_skipped, expected_skipped)

    def test_reports_are_written_as_utf8_json(self) -> None:
        for video_id in EXPECTED_GATING:
            path = self.output_dir / video_id / "report.json"
            with self.subTest(video=video_id):
                self.assertTrue(path.exists())
                loaded = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(loaded["video_id"], video_id)

    def test_telemetry_is_present_and_bounded(self) -> None:
        for report in self.reports.values():
            telemetry = report["telemetry"]
            with self.subTest(video=report["video_id"]):
                self.assertEqual(len(telemetry["per_feature_ms"]), 10)
                self.assertTrue(all(value >= 0 for value in telemetry["per_feature_ms"].values()))
                self.assertGreater(telemetry["total_ms"], 0)
                self.assertGreaterEqual(telemetry["probe_ms"], 0)
                self.assertGreaterEqual(telemetry["frames_skipped_ratio"], 0.0)
                self.assertLessEqual(telemetry["frames_skipped_ratio"], 1.0)
                self.assertGreaterEqual(telemetry["grounding_match_rate"], 0.0)
                self.assertLessEqual(telemetry["grounding_match_rate"], 1.0)

    def test_retrieval_ranks_query_terms(self) -> None:
        report = self.reports["mfg_01_conveyor_inspection"]
        results = report["retrieval_results"]
        self.assertTrue(results)
        scores = [item["score"] for item in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIn("defect", results[0]["text"].lower())
        self.assertGreater(results[0]["score"], 0.0)
        self.assertGreater(results[0]["tfidf_score"], 0.0)
        self.assertEqual(self.reports["mfg_01_conveyor_inspection"]["retrieval_stats"]["scorer"], "BM25+TFIDF")

    def test_short_id_alias_resolves(self) -> None:
        report = process_video(
            "mfg_02",
            config=self.config,
            videos_dir=self.videos_dir,
            logs_dir=self.logs_dir,
            output_dir=self.output_dir,
            query=QUERY,
        )
        self.assertEqual(report["video_id"], "mfg_02_assembly_station")


if __name__ == "__main__":
    unittest.main()
