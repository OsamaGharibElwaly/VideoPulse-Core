"""Build README figures/screenshots from the actual local reports and videos.

Everything written here is derived from files in ``outputs/`` and the real
sample footage in ``data/real/videos/``. No metrics are invented. Run after the
generated and real pipeline commands have produced their reports::

    python tools/build_readme_assets.py
"""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.media_probe import load_manifest  # noqa: E402
from core.real_media_ingest import probe_real_stream, read_real_frames  # noqa: E402
from core.yolo_detector import DetectorConfig, YOLODetector  # noqa: E402

FIGURES = ROOT / "docs" / "figures"
SCREENSHOTS = ROOT / "docs" / "screenshots"
REPORTS = ROOT / "docs" / "reports"

GENERATED_LABELS = {
    "edu_01_slide_lecture": "edu_01\nslide",
    "edu_02_hands_on_demo": "edu_02\ndemo",
    "mfg_01_conveyor_inspection": "mfg_01\nconveyor",
    "mfg_02_assembly_station": "mfg_02\nassembly",
    "ret_01_aisle_traffic": "ret_01\naisle",
    "ret_02_facility_monitoring": "ret_02\nfacility",
}
REAL_LABELS = {
    "real_edu_classroom": "real_edu\nclassroom",
    "real_mfg_worker_zone": "real_mfg\nworker zone",
    "real_ret_store_aisle": "real_ret\nstore aisle",
    "real_sec_people": "real_sec\npeople",
}
ACTIVE = "#2e7d5b"
SKIPPED = "#d99a2b"
YOLO_COLOR = "#2f5d8c"
FALLBACK_COLOR = "#8a6d9c"
INK = "#1d2733"
GRID = "#d9e0e7"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.edgecolor": GRID,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def _load_reports() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    generated = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "outputs").glob("*/report.json"))
    ]
    real = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "outputs" / "real").glob("*/report.json"))
    ]
    fallback_path = ROOT / "outputs" / "fallback" / "ret_02_facility_monitoring" / "report.json"
    fallback = json.loads(fallback_path.read_text(encoding="utf-8"))
    return generated, real, fallback


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    assert path.is_file() and path.stat().st_size > 1000, f"empty asset: {path}"


def _bar_labels(ax: plt.Axes, bars: Any, values: list[Any], fmt: str = "{:.0f}") -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_title(
        "VideoPulse Core — architecture and evidence flow",
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=12,
    )

    def box(x: float, y: float, w: float, h: float, text: str, color: str, fs: int = 9) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.10",
            linewidth=1.4,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, linespacing=1.35,
        )

    def arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                linewidth=1.3, color="#617385",
                connectionstyle="arc3,rad=0",
            )
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, label, ha="center", va="bottom", fontsize=7.5, color="#617385")

    # Inputs
    box(0.25, 4.75, 2.55, 1.05, "Generated sample videos\n6 slots · 3 industries", "#3a7d8c")
    box(0.25, 3.35, 2.55, 1.05, "Real public videos\n4 clips · CC BY 4.0", "#3a7d8c")
    box(0.25, 1.95, 2.55, 1.05, "Side-cars\nSRT/VTT/TXT/LOG/JSON/CSV", "#3a7d8c")
    box(0.25, 0.55, 2.55, 1.05, "YOLO26n weights\nlocal-only resolution", YOLO_COLOR)

    # Core stages
    box(3.35, 4.05, 2.15, 1.15, "Probe + ingest\nstatus codes\nframes · fps · zones", "#2e7d5b")
    box(6.05, 4.05, 2.15, 1.15, "Feature router\n10 can_run() checks\nskip reasons", "#2e7d5b")
    box(3.35, 2.15, 2.15, 1.15, "YOLO26 detector\nyolo / opencv_fallback\nlabels · conf · boxes", YOLO_COLOR)
    box(6.05, 2.15, 2.15, 1.15, "CV features\nkeyframes · zones\nanomalies · text · states", "#2e7d5b")
    box(8.75, 2.15, 2.15, 1.15, "NLP features\nalign · BM25/TF-IDF\nevents · graph · grounding", "#2e7d5b")
    box(6.05, 0.45, 4.85, 1.05, "Fusion + evaluation\nunified report · telemetry · detector mode/reason · generated assets", "#8a6d9c", fs=9)

    arrow(2.80, 5.27, 3.35, 4.72)
    arrow(2.80, 3.87, 3.35, 4.55)
    arrow(2.80, 2.47, 3.35, 2.67)
    arrow(2.80, 1.07, 3.35, 2.34)
    arrow(5.50, 4.62, 6.05, 4.62)
    arrow(7.12, 4.05, 7.12, 3.30, "context")
    arrow(7.12, 3.30, 4.42, 3.30, "detections")
    arrow(5.50, 2.72, 6.05, 2.72)
    arrow(8.20, 2.72, 8.75, 2.72)
    arrow(7.12, 2.15, 7.12, 1.50)
    arrow(9.82, 2.15, 9.82, 1.50)
    ax.text(0.25, 0.16, "Every skipped feature carries an explicit reason code; missing media never crashes the run.", fontsize=8.5, color="#617385")
    _save(fig, FIGURES / "pipeline_architecture.png")


def feature_coverage(generated: list[dict[str, Any]], real: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), sharex=True)
    for ax, reports, labels, title in (
        (axes[0], generated, GENERATED_LABELS, "Generated sample track (6 videos)"),
        (axes[1], real, REAL_LABELS, "Real footage track (4 videos)"),
    ):
        names = [labels.get(r["video_id"], r["video_id"]) for r in reports]
        active = [r["telemetry"]["active_feature_count"] for r in reports]
        skipped = [r["telemetry"]["skipped_feature_count"] for r in reports]
        y = np.arange(len(names))
        ax.barh(y, active, color=ACTIVE, label="active")
        ax.barh(y, skipped, left=active, color=SKIPPED, label="skipped")
        ax.set_yticks(y, names, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 10.4)
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlabel("features out of 10")
        ax.set_title(title)
        for i, (a, s) in enumerate(zip(active, skipped)):
            ax.text(a + s + 0.12, i, f"{a}+{s}", va="center", fontsize=8)
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Content-aware gating: all 10 features checked per video", fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.945, "A skip is an explicit can_run() decision, not a crash or an unreported failure.", ha="center", fontsize=9, color="#617385")
    _save(fig, FIGURES / "feature_coverage.png")


def detections_and_grounding(generated: list[dict[str, Any]], real: list[dict[str, Any]]) -> None:
    reports = generated + real
    labels = [GENERATED_LABELS.get(r["video_id"], REAL_LABELS.get(r["video_id"], r["video_id"])).replace("\n", " ") for r in reports]
    x = np.arange(len(reports))
    colors = [YOLO_COLOR if r["video_id"] in GENERATED_LABELS else "#c06b3d" for r in reports]

    fig, axes = plt.subplots(2, 1, figsize=(13.4, 7.2), sharex=True)
    detections = [r["telemetry"]["object_detections"] for r in reports]
    bars = axes[0].bar(x, detections, color=colors)
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylabel("object detections")
    axes[0].set_title("YOLO26 detections on 24–25 sampled frames per video")
    _bar_labels(axes[0], bars, detections)
    axes[0].text(0.01, 0.78, "Scene content drives count; this is not an accuracy metric.", transform=axes[0].transAxes, fontsize=8, color="#617385")

    grounding = [r["telemetry"]["grounding_match_rate"] * 100 for r in reports]
    bars = axes[1].bar(x, grounding, color=colors)
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("grounding match rate (%)")
    axes[1].set_title("Claim grounding over time-aligned CV + text evidence")
    _bar_labels(axes[1], bars, grounding, "{:.1f}")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    axes[1].set_xticks(x, labels, rotation=42, ha="right", fontsize=8)
    fig.suptitle("What the pipeline actually observed in the two local runs", fontsize=14, fontweight="bold", y=1.01)
    _save(fig, FIGURES / "detections_and_grounding.png")


def skip_reasons(generated: list[dict[str, Any]], real: list[dict[str, Any]]) -> None:
    counts: Counter[str] = Counter()
    for report in generated + real:
        counts.update(item["reason"] for item in report["skipped_features"])
    reasons = sorted(counts, key=lambda key: (-counts[key], key))
    values = [counts[reason] for reason in reasons]
    fig, ax = plt.subplots(figsize=(10.6, 4.7))
    bars = ax.barh(reasons[::-1], values[::-1], color=SKIPPED)
    ax.set_xlabel("number of videos (out of 10) where the feature was skipped")
    ax.set_title("Skip reasons are first-class outputs")
    ax.grid(axis="y", visible=False)
    for bar, value in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9)
    ax.text(0.01, -0.23, "Reasons come from the explicit can_run(context) checks in core/feature_router.py.", transform=ax.transAxes, fontsize=8.5, color="#617385")
    _save(fig, FIGURES / "skip_reasons.png")


def latency_profile(generated: list[dict[str, Any]], real: list[dict[str, Any]], fallback: dict[str, Any]) -> None:
    reports = generated + real + [fallback]
    labels = [
        GENERATED_LABELS.get(r["video_id"], REAL_LABELS.get(r["video_id"], "fallback\nret_02"))
        .replace("\n", " ")
        for r in reports
    ]
    totals = [r["telemetry"]["total_ms"] for r in reports]
    detection = [r["detector"]["latency_ms"] for r in reports]
    colors = [FALLBACK_COLOR if r is fallback else (YOLO_COLOR if r["video_id"] in GENERATED_LABELS else "#c06b3d") for r in reports]
    x = np.arange(len(reports))
    fig, ax = plt.subplots(figsize=(13.4, 5.2))
    ax.bar(x, totals, color=colors, label="total run ms")
    ax.bar(x, detection, color="#1b2733", alpha=0.72, label="detector ms (inside total)")
    ax.set_xticks(x, labels, rotation=42, ha="right", fontsize=8)
    ax.set_ylabel("milliseconds")
    ax.set_title("Single local run timings — not a performance benchmark")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    ax.text(0.01, 0.96, "First-run model load and 1080p real footage dominate the tall bars.", transform=ax.transAxes, va="top", fontsize=8.5, color="#617385")
    _save(fig, FIGURES / "latency_profile.png")


def _frame_near(video: Path, timestamp: float, target: tuple[int, int] = (320, 240)) -> np.ndarray:
    probe = probe_real_stream(video)
    frames, stamps = read_real_frames(video, probe, 24, target_size=target)
    index = int(np.argmin(np.abs(np.asarray(stamps) - timestamp)))
    return cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)


def real_detection_frame(video_id: str, filename: str, output: str) -> None:
    manifest = load_manifest(ROOT / "config" / "real_videos_manifest.json")
    entry = next(item for item in manifest["videos"] if item["video_id"] == video_id)
    video = ROOT / "data" / "real" / "videos" / filename
    probe = probe_real_stream(video)
    frames, stamps = read_real_frames(video, probe, 24, target_size=(320, 240))
    detector = YOLODetector(DetectorConfig(model="yolo26n.pt"))
    result = detector.detect(
        frames, stamps, width=320, height=240,
        rois=entry.get("rois", []), scene=str(entry.get("content_profile", {}).get("scene", "")),
    )
    counts = Counter(item["frame_index"] for item in result.detections)
    frame_index = counts.most_common(1)[0][0] if counts else 0
    frame = cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2RGB)
    shown = [item for item in result.detections if item["frame_index"] == frame_index]

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.imshow(frame)
    for roi in entry.get("rois", []):
        x1, y1, x2, y2 = roi["coords"]
        ax.add_patch(Rectangle((x1 * 320, y1 * 240), (x2 - x1) * 320, (y2 - y1) * 240,
                               fill=False, edgecolor="#42d6ff", linewidth=1.4, linestyle="--",
                               label="manifest zone"))
        ax.text(x1 * 320 + 3, y1 * 240 + 11, roi["zone_id"], color="#42d6ff", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1))
    for item in shown:
        x, y, w, h = item["bbox"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#ffd166", linewidth=2))
        ax.text(x, max(y - 3, 0), f"{item['label']} {item['confidence']:.2f}",
                color="#101820", fontsize=7.5,
                bbox=dict(facecolor="#ffd166", edgecolor="none", pad=1.2))
    ax.set_xlim(0, 320)
    ax.set_ylim(240, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Real footage · {filename} · {result.mode.upper()} {result.model}\n"
        f"frame @ {stamps[frame_index]:.1f}s · {len(result.detections)} detections across {len(frames)} sampled frames · "
        f"labels: {', '.join(sorted({d['label'] for d in result.detections}))}",
        fontsize=11, fontweight="bold",
    )
    ax.text(0.01, -0.06, "Cyan dashed = manifest ROI · amber = YOLO26 box", transform=ax.transAxes,
            fontsize=8.5, color="#617385")
    _save(fig, SCREENSHOTS / output)


def generated_cv_outputs(generated: list[dict[str, Any]]) -> None:
    by_id = {report["video_id"]: report for report in generated}
    mfg = by_id["mfg_01_conveyor_inspection"]
    edu = by_id["edu_01_slide_lecture"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0))

    peak = max(mfg["anomalies"], key=lambda item: item["score"])
    frame = _frame_near(ROOT / "data" / "videos" / "mfg_01_conveyor_inspection.mp4", peak["timestamp_sec"])
    axes[0, 0].imshow(frame)
    for zone in mfg["zones"]:
        x, y, w, h = zone["bounds_px"]
        axes[0, 0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#42d6ff", linewidth=1.8, linestyle="--"))
        axes[0, 0].text(x + 3, y + 11, zone["zone_id"], color="#42d6ff", fontsize=8,
                        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1))
    for item in mfg["keyframes"]:
        axes[0, 0].axvline(item["timestamp_sec"] * 320 / max(mfg["duration_sec"], 1), color="white", alpha=0.25)
    axes[0, 0].set_title(f"Zones + keyframes · peak anomaly @ {peak['timestamp_sec']:.1f}s")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    text_frame = _frame_near(ROOT / "data" / "videos" / "edu_01_slide_lecture.mp4", 0.0)
    axes[0, 1].imshow(text_frame)
    for region in [r for r in edu["text_regions"] if r["frame_index"] == 0][:12]:
        x, y, w, h = region["bbox"]
        axes[0, 1].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#7CFFB2", linewidth=1.2))
    axes[0, 1].set_title(f"On-screen text detection · {len(edu['text_regions'])} regions in report")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    states = mfg["action_states"]
    axes[1, 0].barh([s["state"] for s in states], [s["end_sec"] - s["start_sec"] for s in states], color=ACTIVE)
    axes[1, 0].set_xlabel("seconds")
    axes[1, 0].set_title("Action-state indexing (object counts feed the states)")
    axes[1, 0].grid(axis="y", visible=False)

    stamps = [a["timestamp_sec"] for a in mfg["anomalies"]]
    scores = [a["score"] for a in mfg["anomalies"]]
    axes[1, 1].scatter(stamps, scores, c=SKIPPED, s=42, zorder=3)
    axes[1, 1].set_xlabel("timestamp (s)")
    axes[1, 1].set_ylabel("anomaly score")
    axes[1, 1].set_title(f"Visual anomaly scoring · {len(stamps)} scored moments")
    fig.suptitle("Generated-data track — actual CV feature outputs", fontsize=14, fontweight="bold", y=1.01)
    _save(fig, SCREENSHOTS / "generated_cv_outputs.png")


def nlp_report_card(real: list[dict[str, Any]]) -> None:
    report = next(item for item in real if item["video_id"] == "real_sec_people")
    aligned = report["aligned_segments"][0]
    event = report["parsed_events"][0]
    edges = [e for e in report["graph_edges"] if e["source"] == event["event_id"]]
    grounding = report["grounding_checks"][0]
    hit = report["retrieval_results"][0]
    lines = [
        "REAL VIDEO NLP TRACK — real_sec_people",
        "",
        f"Aligned text  [{aligned['source']} @ {aligned['start_sec']:.1f}-{aligned['end_sec']:.1f}s]",
        *textwrap.wrap(aligned["text"], 78, initial_indent="  ", subsequent_indent="  "),
        "",
        f"Parsed event  {event['event_id']} @ {event['timestamp_sec']:.1f}s  zone={event['zone']}",
        "  " + ", ".join(f"{k}={v}" for k, v in event["fields"].items()),
        "",
        "Graph edges   " + "  |  ".join(f"{e['source']} -[{e['relation']}]-> {e['target']}" for e in edges),
        "",
        f"Grounding     {grounding['status']} — {', '.join(grounding['supporting_evidence'])}",
        "",
        f"Top retrieval @{hit['start_sec']:.1f}s score={hit['score']:.2f}  bm25={hit['bm25_score']:.3f} tfidf={hit['tfidf_score']:.3f}",
        "  " + hit["text"],
        "",
        f"Report totals: {len(report['aligned_segments'])} aligned · {len(report['parsed_events'])} events · "
        f"{len(report['graph_edges'])} edges · {len(report['grounding_checks'])} grounding checks",
    ]
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    ax.axis("off")
    ax.set_title("CV detections → timestamped log → NLP evidence", fontsize=14, fontweight="bold", loc="left", pad=14)
    ax.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9.2, family="DejaVu Sans Mono",
            linespacing=1.45, bbox=dict(boxstyle="round,pad=0.7", facecolor="#f5f8fb", edgecolor=GRID))
    _save(fig, SCREENSHOTS / "nlp_report_card.png")


def real_event_timeline(real: list[dict[str, Any]]) -> None:
    report = next(item for item in real if item["video_id"] == "real_sec_people")
    events = report["parsed_events"]
    stamps = [event["timestamp_sec"] for event in events]
    labels = [event["fields"].get("label", event.get("entity", "event")) for event in events]
    unique = sorted(set(labels))
    y = [unique.index(label) for label in labels]
    fig, ax = plt.subplots(figsize=(12.2, 4.9))
    ax.scatter(stamps, y, s=60, c=[YOLO_COLOR if label == "person" else SKIPPED for label in labels], zorder=3)
    for item in report["keyframes"]:
        ax.axvline(item["timestamp_sec"], color=GRID, linewidth=0.9, zorder=0)
    for hit in report["retrieval_results"][:5]:
        ax.axvline(hit["start_sec"], color="#2e7d5b", linestyle=":", linewidth=1.1, alpha=0.75, zorder=1)
    ax.set_yticks(range(len(unique)), unique)
    ax.set_xlabel("video time (seconds)")
    ax.set_title("Real detection-derived events aligned with keyframes and retrieval hits")
    ax.set_xlim(0, report["duration_sec"])
    ax.text(0.01, -0.22, "Grey = sampled keyframe · green dotted = top retrieval hit", transform=ax.transAxes, fontsize=8.5, color="#617385")
    _save(fig, FIGURES / "real_event_timeline.png")


def copy_reports(generated: list[dict[str, Any]], real: list[dict[str, Any]], fallback: dict[str, Any]) -> None:
    targets = [
        ("generated", generated),
        ("real", real),
        ("fallback", [fallback]),
    ]
    for subdir, reports in targets:
        destination = REPORTS / subdir
        destination.mkdir(parents=True, exist_ok=True)
        for report in reports:
            source = ROOT / "outputs" / ("" if subdir == "generated" else subdir) / report["video_id"] / "report.json"
            shutil.copyfile(source, destination / f"{report['video_id']}.json")
            assert (destination / f"{report['video_id']}.json").stat().st_size > 1000


def verify(expected: list[Path]) -> None:
    for path in expected:
        assert path.is_file(), f"missing asset: {path}"
        assert path.stat().st_size > 1000, f"empty asset: {path}"
        if path.suffix == ".png":
            image = cv2.imread(str(path))
            assert image is not None and min(image.shape[:2]) > 100, f"unreadable image: {path}"
    print(f"verified {len(expected)} assets")


def main() -> int:
    _style()
    generated, real, fallback = _load_reports()
    architecture()
    feature_coverage(generated, real)
    detections_and_grounding(generated, real)
    skip_reasons(generated, real)
    latency_profile(generated, real, fallback)
    real_detection_frame("real_sec_people", "people_detection.mp4", "real_yolo_people.png")
    real_detection_frame("real_mfg_worker_zone", "worker-zone-detection.mp4", "real_yolo_worker_zone.png")
    real_detection_frame("real_edu_classroom", "classroom.mp4", "real_yolo_classroom.png")
    generated_cv_outputs(generated)
    nlp_report_card(real)
    real_event_timeline(real)
    copy_reports(generated, real, fallback)
    expected = sorted(FIGURES.glob("*.png")) + sorted(SCREENSHOTS.glob("*.png"))
    verify(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
