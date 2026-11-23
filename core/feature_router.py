"""Content-aware feature gating.

Every feature owns one explicit ``can_run(context) -> (bool, reason)`` check.
A feature that fails its precondition is skipped with a stable reason code;
nothing is fabricated to fill a gap.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.media_probe import MIN_MOTION_ENERGY, MIN_TEXT_DENSITY, MediaContext

CV_FEATURES: tuple[str, ...] = (
    "motion_keyframe_extraction",
    "roi_zone_tracking",
    "visual_anomaly_scoring",
    "on_screen_text_detection",
    "action_state_indexing",
)
NLP_FEATURES: tuple[str, ...] = (
    "transcript_log_alignment",
    "multimodal_bm25_tfidf_retrieval",
    "rule_schema_event_parsing",
    "entity_graph_linking",
    "grounding_consistency_check",
)
FEATURE_ORDER: tuple[str, ...] = CV_FEATURES + NLP_FEATURES


@dataclass(frozen=True)
class FeatureDecision:
    """Outcome of one feature's precondition check."""

    feature: str
    category: str
    enabled: bool
    reason: str


@dataclass(frozen=True)
class FeaturePlan:
    """Active feature order plus auditable skip records."""

    active: tuple[str, ...]
    skipped: tuple[dict[str, str], ...]

    @property
    def skipped_names(self) -> tuple[str, ...]:
        """Return the names of skipped features."""
        return tuple(item["feature"] for item in self.skipped)


# ---------------------------------------------------------------------------
# CV preconditions
# ---------------------------------------------------------------------------


def can_run_motion_keyframe_extraction(context: MediaContext) -> tuple[bool, str]:
    """Need decodable frames with measurable frame-to-frame change."""
    if not context.sampled_frames:
        return False, "NO_FRAMES"
    if context.motion_energy < MIN_MOTION_ENERGY:
        return False, "LOW_MOTION_VARIANCE"
    return True, "MOTION_SIGNAL_PRESENT"


def can_run_roi_zone_tracking(context: MediaContext) -> tuple[bool, str]:
    """Need an authored ROI or a zone derived from real detections."""
    if context.detected_zone_ids:
        return True, "ZONES_DETECTED"
    if not context.roi_presence or not context.rois:
        return False, "NO_SPATIAL_ZONES"
    return True, "ZONES_DEFINED"


def can_run_visual_anomaly_scoring(context: MediaContext) -> tuple[bool, str]:
    """Need a stable baseline; noisy or crowded scenes are skipped honestly."""
    if not context.sampled_frames:
        return False, "NO_FRAMES"
    mode = str(context.profile.get("baseline_mode", "continuous_motion"))
    reasons = {
        "continuous_motion": "NO_STABLE_BASELINE",
        "high_variance": "HIGH_MOTION_VARIANCE",
        "dynamic_scene": "DYNAMIC_SCENE_BASELINE",
    }
    if mode in reasons:
        return False, reasons[mode]
    if mode != "stable":
        return False, "NO_STABLE_BASELINE"
    return True, "STABLE_BASELINE_AVAILABLE"


def can_run_on_screen_text_detection(context: MediaContext) -> tuple[bool, str]:
    """Need readable, high-contrast text regions in the frame."""
    if not context.sampled_frames:
        return False, "NO_FRAMES"
    mode = str(context.profile.get("text_mode", "no_text"))
    if mode == "unreadable_resolution":
        return False, "NO_READABLE_TEXT"
    if mode == "illegible_signage":
        return False, "NO_LEGIBLE_SIGNAGE"
    if mode != "readable":
        return False, "NO_TEXT_REGION"
    if context.text_region_density < MIN_TEXT_DENSITY:
        return False, "NO_TEXT_REGION"
    return True, "TEXT_REGION_DETECTED"


def can_run_action_state_indexing(context: MediaContext) -> tuple[bool, str]:
    """Need a frame timeline to divide into motion states."""
    if not context.sampled_frames:
        return False, "NO_FRAMES"
    return True, "TIMELINE_AVAILABLE"


# ---------------------------------------------------------------------------
# NLP preconditions
# ---------------------------------------------------------------------------


def can_run_transcript_log_alignment(context: MediaContext) -> tuple[bool, str]:
    """Need at least one timestamped transcript or log source."""
    if not context.text_source_available:
        return False, "NO_AUDIO_TRANSCRIPT"
    return True, "TRANSCRIPT_OR_LOG_AVAILABLE"


def can_run_multimodal_bm25_tfidf_retrieval(context: MediaContext) -> tuple[bool, str]:
    """Need searchable text or authored visual tags."""
    if not context.text_source_available and not context.visual_tags:
        return False, "NO_TEXT_SOURCE"
    return True, "SEARCHABLE_INDEX_AVAILABLE"


def can_run_rule_schema_event_parsing(context: MediaContext) -> tuple[bool, str]:
    """Need text containing timestamped claims or key/value event tokens."""
    if not context.text_source_available:
        return False, "NO_EVENT_TEXT"
    return True, "EVENT_TEXT_AVAILABLE"


def can_run_entity_graph_linking(context: MediaContext) -> tuple[bool, str]:
    """Need entities from text, authored zones or detector-derived zones."""
    if not context.text_source_available and not context.rois and not context.detected_zone_ids:
        return False, "NO_LINKABLE_ENTITIES"
    return True, "ENTITIES_AVAILABLE"


def can_run_grounding_consistency_check(context: MediaContext) -> tuple[bool, str]:
    """Need visual labels to corroborate spoken/logged claims."""
    if not bool(context.profile.get("groundable", False)):
        return False, "INSUFFICIENT_VISUAL_LABELS"
    if not context.text_source_available:
        return False, "NO_CLAIM_SOURCE"
    return True, "GROUNDABLE_CLAIMS_AVAILABLE"


CHECKS = {
    "motion_keyframe_extraction": (can_run_motion_keyframe_extraction, "cv"),
    "roi_zone_tracking": (can_run_roi_zone_tracking, "cv"),
    "visual_anomaly_scoring": (can_run_visual_anomaly_scoring, "cv"),
    "on_screen_text_detection": (can_run_on_screen_text_detection, "cv"),
    "action_state_indexing": (can_run_action_state_indexing, "cv"),
    "transcript_log_alignment": (can_run_transcript_log_alignment, "nlp"),
    "multimodal_bm25_tfidf_retrieval": (can_run_multimodal_bm25_tfidf_retrieval, "nlp"),
    "rule_schema_event_parsing": (can_run_rule_schema_event_parsing, "nlp"),
    "entity_graph_linking": (can_run_entity_graph_linking, "nlp"),
    "grounding_consistency_check": (can_run_grounding_consistency_check, "nlp"),
}


def evaluate_features(context: MediaContext) -> list[FeatureDecision]:
    """Run all ten precondition checks in canonical feature order."""
    decisions: list[FeatureDecision] = []
    for feature in FEATURE_ORDER:
        check, category = CHECKS[feature]
        enabled, reason = check(context)
        decisions.append(FeatureDecision(feature, category, enabled, reason))
    return decisions


def build_feature_plan(context: MediaContext) -> FeaturePlan:
    """Split decisions into an execution order and skip audit records."""
    decisions = evaluate_features(context)
    active = tuple(decision.feature for decision in decisions if decision.enabled)
    skipped = tuple(
        {"feature": decision.feature, "status": "SKIPPED", "reason": decision.reason}
        for decision in decisions
        if not decision.enabled
    )
    return FeaturePlan(active=active, skipped=skipped)
