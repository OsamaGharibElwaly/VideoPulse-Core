"""Five local NLP features: alignment, retrieval, parsing, graphs, grounding."""

from __future__ import annotations

import re
from typing import Any, Callable

import networkx as nx
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from core.media_probe import MediaContext

FeatureOutput = tuple[list[dict[str, Any]], dict[str, Any]]

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_KEYVALUE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*([A-Za-z0-9_./-]+)")
_TIME_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[,.]\d+)?")
_SRT_RE = re.compile(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})")


def _to_seconds(match: re.Match[str]) -> float:
    """Convert a matched HH:MM:SS or MM:SS timestamp to seconds."""
    hours, minutes, seconds = match.group(1), match.group(2), match.group(3)
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)


def _parse_time(text: str) -> float | None:
    """Parse a leading timestamp from a log/SRT line."""
    match = _TIME_RE.search(text.strip())
    return float(_to_seconds(match)) if match else None


# ---------------------------------------------------------------------------
# 1. Transcript & log alignment
# ---------------------------------------------------------------------------


def _align_srt(text: str) -> list[dict[str, Any]]:
    """Align SRT cues to video time windows (with or without blank separators)."""
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "-->" in line:
            times = _SRT_RE.search(line)
            if not times:
                current = None
                continue
            start = _to_seconds(_TIME_RE.search(times.group(1)))
            end = _to_seconds(_TIME_RE.search(times.group(2)))
            current = {
                "source": "transcript",
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "text": "",
                "line_no": len(segments) + 1,
            }
            segments.append(current)
            trailing = re.sub(r"\d{2}:\d{2}:\d{2}[,.]\d{3}", "", line.split("-->", 1)[1]).strip()
            if trailing:
                current["text"] = trailing
        elif current is not None and line and not line.isdigit():
            current["text"] = f"{current['text']} {line}".strip()
    return segments


def _align_log(text: str, fallback_window: float) -> list[dict[str, Any]]:
    """Align timestamped log lines to short windows after each event."""
    segments: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        start = _parse_time(line)
        if start is None:
            continue
        segments.append(
            {
                "source": "log",
                "start_sec": round(start, 3),
                "end_sec": round(start + fallback_window, 3),
                "text": line.strip(),
                "line_no": len(segments) + 1,
            }
        )
    return segments


def align_transcript_log(context: MediaContext) -> FeatureOutput:
    """Sync spoken captions and/or log lines into a common timeline."""
    segments: list[dict[str, Any]] = []
    if context.transcript_text.strip():
        segments.extend(_align_srt(context.transcript_text))
        if not segments:
            segments.extend(_align_log(context.transcript_text, 2.0))
    if context.log_text.strip():
        segments.extend(_align_log(context.log_text, 2.0))
    segments.sort(key=lambda segment: (segment["start_sec"], segment["source"]))
    stats = {
        "segments": len(segments),
        "transcript_segments": sum(s["source"] == "transcript" for s in segments),
        "log_segments": sum(s["source"] == "log" for s in segments),
    }
    return segments, stats


# ---------------------------------------------------------------------------
# 2. Multimodal BM25 + TF-IDF retrieval
# ---------------------------------------------------------------------------


def _visual_tag_terms(cv_results: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Derive honest searchable tags from features that actually ran."""
    terms: list[str] = []
    for keyframe in cv_results.get("keyframes", []):
        terms.append("motion keyframe")
        _ = keyframe
    for zone in cv_results.get("zones", []):
        terms.extend(["zone", str(zone.get("zone_id", ""))])
    for anomaly in cv_results.get("anomalies", []):
        terms.extend(["anomaly", str(anomaly.get("kind", ""))])
    for region in cv_results.get("text_regions", [])[:3]:
        terms.append("text region")
        _ = region
    for state in cv_results.get("action_states", []):
        terms.append(str(state.get("state", "")))
    # Phase 2: real detector labels are searchable, with confidence gating.
    for detection in cv_results.get("object_detections", []):
        if float(detection.get("confidence", 0.0)) >= 0.35:
            terms.append(str(detection.get("label", "")).lower())
            terms.append("object")
    return list(dict.fromkeys(t for t in terms if t))


def multimodal_bm25_tfidf_retrieval(
    context: MediaContext,
    aligned: list[dict[str, Any]],
    cv_results: dict[str, list[dict[str, Any]]],
    query: str,
) -> FeatureOutput:
    """Rank moments by combined BM25 and TF-IDF similarity to a query."""
    visual_terms = _visual_tag_terms(cv_results) + [str(tag) for tag in context.visual_tags]
    documents = []
    for segment in aligned:
        documents.append(
            {
                "start_sec": segment["start_sec"],
                "end_sec": segment["end_sec"],
                "source": segment["source"],
                "text": segment["text"],
                "corpus": segment["text"],
            }
        )
    # Visual tags form their own moment so text-only relevance is not diluted.
    if visual_terms:
        documents.append(
            {
                "start_sec": 0.0,
                "end_sec": context.duration_sec,
                "source": "visual_tags",
                "text": " ".join(visual_terms),
                "corpus": " ".join(visual_terms),
            }
        )
    stats = {"documents_indexed": len(documents), "query": query, "scorer": "BM25+TFIDF"}
    if not documents or not query.strip():
        return [], stats

    tokenised = [_TOKEN_RE.findall(doc["corpus"].lower()) for doc in documents]
    bm25 = BM25Okapi(tokenised)
    bm25_scores = [
        max(0.0, float(score))
        for score in bm25.get_scores(_TOKEN_RE.findall(query.lower()))
    ]

    try:
        vectoriser = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
        matrix = vectoriser.fit_transform([doc["corpus"] for doc in documents])
        query_vector = vectoriser.transform([query])
        tfidf_scores = [float(score) for score in linear_kernel(query_vector, matrix).flatten()]
    except ValueError:  # empty vocabulary
        tfidf_scores = [0.0] * len(documents)

    bm25_peak = max(bm25_scores, default=0.0) or 1.0
    tfidf_peak = max(tfidf_scores, default=0.0) or 1.0
    ranked: list[dict[str, Any]] = []
    for doc, bm25_score, tfidf_score in zip(documents, bm25_scores, tfidf_scores):
        combined = 0.5 * (bm25_score / bm25_peak) + 0.5 * (tfidf_score / tfidf_peak)
        if doc["source"] == "visual_tags":
            combined *= 0.9  # timestamped text is more specific than global tags
        if combined <= 0.0:
            continue
        ranked.append(
            {
                "start_sec": doc["start_sec"],
                "end_sec": doc["end_sec"],
                "source": doc["source"],
                "text": doc["text"][:180],
                "bm25_score": round(bm25_score, 4),
                "tfidf_score": round(tfidf_score, 4),
                "score": round(combined, 4),
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    stats["results"] = len(ranked[:5])
    return ranked[:5], stats


# ---------------------------------------------------------------------------
# 3. Rule-schema event parsing
# ---------------------------------------------------------------------------


def _event_fields(text: str) -> dict[str, str]:
    """Extract key=value tokens from one aligned text segment."""
    return {match.group(1).lower(): match.group(2) for match in _KEYVALUE_RE.finditer(text)}


def rule_schema_event_parsing(
    context: MediaContext, aligned: list[dict[str, Any]]
) -> FeatureOutput:
    """Extract structured JSON event records via regex and domain rules."""
    events: list[dict[str, Any]] = []
    for segment in aligned:
        text = segment["text"]
        fields = _event_fields(text)
        bracketed = re.search(r"\[([A-Za-z0-9_-]+)\]", text)
        step_match = re.search(r"\bstep\s*[:=]?\s*(\d+)", text, re.IGNORECASE)
        definition = re.search(r"definition\s*:?\s*([^.]+)", text, re.IGNORECASE)
        entity = (
            fields.get("part")
            or fields.get("term")
            or fields.get("location")
            or fields.get("entity")
            or fields.get("station")
            or ""
        )
        action = (
            fields.get("action")
            or fields.get("event")
            or fields.get("alert")
            or ("define" if definition else "")
        )
        status = fields.get("status") or fields.get("severity") or ""
        zone = fields.get("zone") or (bracketed.group(1) if bracketed else "")
        if not any([entity, action, status, zone, step_match, fields.get("defect")]):
            continue
        events.append(
            {
                "event_id": f"{segment['source']}_L{segment['line_no']}",
                "timestamp_sec": segment["start_sec"],
                "end_sec": segment["end_sec"],
                "entity": entity,
                "action": action,
                "status": status,
                "zone": zone,
                "defect": fields.get("defect", ""),
                "step": int(step_match.group(1)) if step_match else None,
                "claim": (definition.group(1).strip() if definition else text[:120]),
                "fields": fields,
                "raw_text": text[:180],
            }
        )
    stats = {
        "events": len(events),
        "statuses": sorted({event["status"] for event in events if event["status"]}),
    }
    return events, stats


# ---------------------------------------------------------------------------
# 4. Entity graph linking
# ---------------------------------------------------------------------------


def _node_id(prefix: str, value: str) -> str:
    """Build a typed graph node id."""
    return f"{prefix}:{value}"


def entity_graph_linking(
    context: MediaContext,
    aligned: list[dict[str, Any]],
    events: list[dict[str, Any]],
    object_detections: list[dict[str, Any]] | None = None,
) -> FeatureOutput:
    """Connect events, zones, entities and detected objects in a graph."""
    graph = nx.DiGraph()
    for roi in context.rois:
        graph.add_node(_node_id("zone", str(roi.get("zone_id", "zone"))), node_type="zone")

    for event in events:
        graph.add_node(event["event_id"], node_type="event")
        links = (
            ("ent", "entity", "MENTIONS"),
            ("zone", "zone", "AT_ZONE"),
            ("act", "action", "HAS_ACTION"),
            ("stat", "status", "HAS_STATUS"),
            ("defect", "defect", "HAS_DEFECT"),
        )
        for prefix, field, relation in links:
            value = str(event.get(field, "")).strip()
            if not value:
                continue
            target = _node_id(prefix, value)
            graph.add_node(target, node_type=prefix)
            graph.add_edge(event["event_id"], target, relation=relation)

    # Phase 2: link confident detector labels to zones and to claims that
    # mention them, so graph traversal reflects what was actually seen.
    for detection in object_detections or []:
        label = str(detection.get("label", "")).strip()
        if not label or float(detection.get("confidence", 0.0)) < 0.35:
            continue
        object_node = _node_id("object", label)
        graph.add_node(object_node, node_type="object", confidence=detection.get("confidence"))
        zone_id = str(detection.get("zone_id", "")).strip()
        zone_node = _node_id("zone", zone_id)
        if zone_id and graph.has_node(zone_node):
            graph.add_edge(object_node, zone_node, relation="DETECTED_IN")
        for event in events:
            if label.lower() in str(event.get("raw_text", "")).lower():
                graph.add_edge(event["event_id"], object_node, relation="SUPPORTED_BY_OBJECT")

    corpus = " ".join(segment["text"].lower() for segment in aligned)
    for tag in context.visual_tags:
        if str(tag).lower() in corpus:
            tag_node = _node_id("tag", str(tag))
            graph.add_node(tag_node, node_type="tag")
            for zone in context.rois:
                zone_node = _node_id("zone", str(zone.get("zone_id", "zone")))
                if graph.has_node(zone_node):
                    graph.add_edge(zone_node, tag_node, relation="OBSERVED_WITH")

    edges = [
        {
            "source": source,
            "relation": str(data.get("relation", "RELATED_TO")),
            "target": target,
        }
        for source, target, data in graph.edges(data=True)
    ]
    edges.sort(key=lambda edge: (edge["source"], edge["relation"], edge["target"]))
    stats = {"nodes": graph.number_of_nodes(), "edges": len(edges)}
    return edges, stats


# ---------------------------------------------------------------------------
# 5. Grounding consistency check
# ---------------------------------------------------------------------------


_ACTIVE_STATES = {"BURST", "CONGESTION"}


def _visual_evidence(cv_results: dict[str, list[dict[str, Any]]]) -> list[tuple[float, str, str]]:
    """Flatten CV detections into (timestamp, evidence type, zone) tuples."""
    evidence: list[tuple[float, str, str]] = []
    for keyframe in cv_results.get("keyframes", []):
        evidence.append((float(keyframe["timestamp_sec"]), "keyframe", ""))
    for anomaly in cv_results.get("anomalies", []):
        evidence.append((float(anomaly["timestamp_sec"]), "anomaly_spike", ""))
    for region in cv_results.get("text_regions", []):
        evidence.append((float(region["timestamp_sec"]), "text_region", ""))
    for zone in cv_results.get("zones", []):
        zone_id = str(zone.get("zone_id", ""))
        for timestamp in zone.get("active_timestamps", []):
            evidence.append((float(timestamp), "zone_motion", zone_id))
        evidence.append((float(zone.get("peak_timestamp_sec", 0.0)), "zone_motion", zone_id))
    for state in cv_results.get("action_states", []):
        if str(state.get("state", "")).upper() in _ACTIVE_STATES:
            midpoint = (float(state["start_sec"]) + float(state["end_sec"])) / 2.0
            evidence.append((midpoint, "action_state", ""))
    # Phase 2: detector output provides timestamped, zone-aware evidence.
    for detection in cv_results.get("object_detections", []):
        if float(detection.get("confidence", 0.0)) >= 0.35:
            evidence.append(
                (
                    float(detection.get("timestamp_sec", 0.0)),
                    "object_detection",
                    str(detection.get("zone_id", "")),
                )
            )
    return evidence


def grounding_consistency_check(
    context: MediaContext,
    events: list[dict[str, Any]],
    cv_results: dict[str, list[dict[str, Any]]],
    *,
    window_sec: float = 2.5,
) -> FeatureOutput:
    """Verify each parsed claim against timestamped visual evidence."""
    evidence = _visual_evidence(cv_results)
    checks: list[dict[str, Any]] = []
    for event in events:
        timestamp = float(event["timestamp_sec"])
        claim_zone = str(event.get("zone", ""))
        matched: set[str] = set()
        for evidence_time, evidence_type, evidence_zone in evidence:
            if abs(evidence_time - timestamp) > window_sec:
                continue
            if evidence_zone and claim_zone and evidence_zone != claim_zone:
                continue
            matched.add(evidence_type)
        # A claimed object is only credited when a confident detection of that
        # exact label exists near the claim (and in the same zone, if stated).
        claim_lower = str(event.get("claim", "")).lower()
        for detection in cv_results.get("object_detections", []):
            label = str(detection.get("label", "")).strip().lower()
            if not label or float(detection.get("confidence", 0.0)) < 0.35:
                continue
            if abs(float(detection.get("timestamp_sec", 0.0)) - timestamp) > window_sec:
                continue
            detected_zone = str(detection.get("zone_id", ""))
            if detected_zone and claim_zone and detected_zone != claim_zone:
                continue
            if label in claim_lower:
                matched.add(f"object:{label}")
        if len(matched) >= 2:
            status = "GROUNDED"
        elif len(matched) == 1:
            status = "PARTIAL_GROUNDING"
        else:
            status = "UNGROUNDED_CLAIM"
        checks.append(
            {
                "event_id": event["event_id"],
                "claim": event.get("claim", "")[:120],
                "expected_timestamp_sec": timestamp,
                "zone": claim_zone,
                "status": status,
                "supporting_evidence": sorted(matched),
            }
        )
    grounded = sum(check["status"] == "GROUNDED" for check in checks)
    partial = sum(check["status"] == "PARTIAL_GROUNDING" for check in checks)
    ungrounded = sum(check["status"] == "UNGROUNDED_CLAIM" for check in checks)
    match_rate = (grounded + 0.5 * partial) / len(checks) if checks else 0.0
    stats = {
        "checks": len(checks),
        "grounded": grounded,
        "partial": partial,
        "ungrounded": ungrounded,
        "grounding_match_rate": round(match_rate, 4),
        "window_sec": window_sec,
    }
    return checks, stats


DISPATCH: dict[str, Callable[..., FeatureOutput]] = {
    "transcript_log_alignment": lambda context, aligned, events, cv_results, query: align_transcript_log(context),
    "multimodal_bm25_tfidf_retrieval": lambda context, aligned, events, cv_results, query: multimodal_bm25_tfidf_retrieval(context, aligned, cv_results, query),
    "rule_schema_event_parsing": lambda context, aligned, events, cv_results, query: rule_schema_event_parsing(context, aligned),
    "entity_graph_linking": lambda context, aligned, events, cv_results, query: entity_graph_linking(context, aligned, events, cv_results.get("object_detections", [])),
    "grounding_consistency_check": lambda context, aligned, events, cv_results, query: grounding_consistency_check(context, events, cv_results),
}


def run_nlp_feature(
    name: str,
    context: MediaContext,
    *,
    aligned: list[dict[str, Any]],
    events: list[dict[str, Any]],
    cv_results: dict[str, list[dict[str, Any]]],
    query: str = "",
) -> FeatureOutput:
    """Execute one NLP feature by its canonical name."""
    return DISPATCH[name](context, aligned, events, cv_results, query)
