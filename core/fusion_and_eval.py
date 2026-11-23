"""Fusion, grounding telemetry, latency accounting and report writing."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core import cv_pipeline, nlp_pipeline
from core.feature_router import CV_FEATURES, build_feature_plan
from core.media_probe import MediaContext, find_video_entry, probe_media
from core.yolo_detector import DetectorConfig, build_detector

REPORT_FIELD = {
    "motion_keyframe_extraction": "keyframes",
    "roi_zone_tracking": "zones",
    "visual_anomaly_scoring": "anomalies",
    "on_screen_text_detection": "text_regions",
    "action_state_indexing": "action_states",
    "transcript_log_alignment": "aligned_segments",
    "multimodal_bm25_tfidf_retrieval": "retrieval_results",
    "rule_schema_event_parsing": "parsed_events",
    "entity_graph_linking": "graph_edges",
    "grounding_consistency_check": "grounding_checks",
}


def _empty_report(context: MediaContext) -> dict[str, Any]:
    """Create the unified report skeleton."""
    return {
        "video_id": context.video_id,
        "industry": context.industry,
        "description": str(context.entry.get("description", "")),
        "duration_sec": round(context.duration_sec, 3),
        "processed_fps": round(context.fps, 3),
        "media": context.to_dict(),
        "executed_features": [],
        "skipped_features": [],
        "keyframes": [],
        "zones": [],
        "anomalies": [],
        "text_regions": [],
        "action_states": [],
        "aligned_segments": [],
        "parsed_events": [],
        "graph_edges": [],
        "grounding_checks": [],
        "retrieval_results": [],
        "detector": {},
        "telemetry": {},
    }


def _elapsed_ms(start: float) -> float:
    """Return milliseconds since a ``perf_counter`` start value."""
    return round((time.perf_counter() - start) * 1000.0, 3)


def process_video(
    video_id: str,
    *,
    config: dict[str, Any],
    videos_dir: Path,
    logs_dir: Path,
    output_dir: Path,
    query: str = "",
    real_dir: Path | None = None,
    model: str | None = None,
    detector: Any | None = None,
) -> dict[str, Any]:
    """Run the gated CV+NLP pipeline for one video and persist its report.

    ``real_dir`` prefers a real video plus side-cars; ``model`` selects the
    Ultralytics weights name. An injected ``detector`` lets tests supply a
    stub YOLO model without downloading anything.
    """
    entry = find_video_entry(config, video_id)
    run_start = time.perf_counter()
    probe_start = time.perf_counter()
    context = probe_media(entry, videos_dir=videos_dir, logs_dir=logs_dir, real_dir=real_dir)
    probe_ms = _elapsed_ms(probe_start)

    selected_model = (
        model
        or entry.get("yolo_model")
        or config.get("defaults", {}).get("model")
        or "yolo26n.pt"
    )
    detector = detector or build_detector(DetectorConfig(model=str(selected_model)))
    detection = detector.detect(
        context.sampled_frames,
        context.timestamps,
        width=context.width,
        height=context.height,
        rois=context.rois,
        scene=context.scene,
        auto_zones=context.auto_zones,
    )
    context.detected_zone_ids = [str(zone.get("zone_id", "")) for zone in detection.zones]

    report = _empty_report(context)
    report["detector"] = detection.to_dict()
    plan = build_feature_plan(context)
    report["skipped_features"] = [dict(item) for item in plan.skipped]
    per_feature_ms = {name: 0.0 for name in REPORT_FIELD}
    cv_results: dict[str, list[dict[str, Any]]] = {key: [] for key in REPORT_FIELD}
    cv_results["object_detections"] = detection.detections
    nlp_state: dict[str, list[dict[str, Any]]] = {
        "aligned_segments": [],
        "parsed_events": [],
    }

    for name in plan.active:
        start = time.perf_counter()
        try:
            if name in CV_FEATURES:
                items, stats = cv_pipeline.run_cv_feature(name, context, detection)
            else:
                items, stats = nlp_pipeline.run_nlp_feature(
                    name,
                    context,
                    aligned=nlp_state["aligned_segments"],
                    events=nlp_state["parsed_events"],
                    cv_results=dict(cv_results),
                    query=query,
                )
        except Exception as exc:  # keep one bad feature from killing the run
            report["skipped_features"].append(
                {"feature": name, "status": "SKIPPED", "reason": f"FEATURE_ERROR: {exc}"}
            )
            per_feature_ms[name] = _elapsed_ms(start)
            continue
        per_feature_ms[name] = _elapsed_ms(start)
        report["executed_features"].append(name)
        report[REPORT_FIELD[name]] = items
        cv_results[REPORT_FIELD[name]] = items
        if name == "transcript_log_alignment":
            nlp_state["aligned_segments"] = items
        elif name == "rule_schema_event_parsing":
            nlp_state["parsed_events"] = items
        if name == "multimodal_bm25_tfidf_retrieval":
            report["retrieval_stats"] = stats
        if name == "grounding_consistency_check":
            report["grounding_stats"] = stats

    grounding_stats = report.get("grounding_stats", {})
    # Frame-skip ratio is derived from the keyframe feature's own selection,
    # so the media is never decoded twice.
    frames_skipped_ratio = 0.0
    if report["keyframes"] and context.sampled_frames:
        frames_skipped_ratio = round(
            1.0 - len(report["keyframes"]) / max(len(context.sampled_frames), 1), 4
        )

    report["telemetry"] = {
        "total_ms": _elapsed_ms(run_start),
        "probe_ms": probe_ms,
        "per_feature_ms": per_feature_ms,
        "frames_skipped_ratio": frames_skipped_ratio,
        "grounding_match_rate": float(grounding_stats.get("grounding_match_rate", 0.0)),
        "detector_mode": detection.mode,
        "detector_reason": detection.reason,
        "detector_model": detection.model,
        "detector_latency_ms": detection.latency_ms,
        "object_detections": len(detection.detections),
        "active_feature_count": len(report["executed_features"]),
        "skipped_feature_count": len(report["skipped_features"]),
    }
    write_report(report, output_dir)
    return report


def write_report(report: dict[str, Any], output_dir: Path) -> Path:
    """Write ``<output_dir>/<video_id>/report.json`` as UTF-8."""
    destination = Path(output_dir) / str(report["video_id"]) / "report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return destination


def run_all_videos(
    config: dict[str, Any],
    *,
    videos_dir: Path,
    logs_dir: Path,
    output_dir: Path,
    query: str = "",
    real_dir: Path | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Process every manifest video in order."""
    return [
        process_video(
            str(entry["video_id"]),
            config=config,
            videos_dir=videos_dir,
            logs_dir=logs_dir,
            output_dir=output_dir,
            query=query,
            real_dir=real_dir,
            model=model,
        )
        for entry in config.get("videos", [])
    ]
