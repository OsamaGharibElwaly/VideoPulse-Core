# PRD.md: Unified Zero-Cost Computer Vision + NLP Video Intelligence Engine

## 1. Executive Summary & Engineering Principles

This PRD defines **VideoPulse Core**, a single unified Python system combining **5 Computer Vision (CV)** features and **5 Natural Language Processing (NLP)** features to prove senior-level competence across both domains in one cohesive codebase.

### Core Constraints

* **Zero Paid LLM APIs:** No external commercial LLM, vision-language, or transcription API keys required.
* **Zero Giant Model Downloads:** Avoid multi-gigabyte neural weights or massive pretrained foundation checkpoints. Use lightweight, deterministic, statistical, and classical methods (**OpenCV**, **NumPy**, **scikit-learn**, **Pillow**, **NetworkX**, **rank_bm25**, **re**, plus optional lightweight local tools only if needed).
* **Zero Massive Datasets:** Operates on **6 short target videos** (2 per industry across the 3 top CV+NLP video industries) plus an optional tiny synthetic clip generator for fast local testing.
* **Content-Aware Feature Routing (No Forced Features):** Not every video contains every signal (for example, a silent warehouse clip has no spoken transcript; a talking-head lecture has minimal conveyor motion). The engine evaluates **content preconditions** first and dynamically activates only matching features, recording skipped features with explicit skip reasons (`NO_AUDIO_TRANSCRIPT`, `LOW_MOTION_VARIANCE`, `NO_TEXT_REGION`, `NO_ZONE_DEFINITION`) instead of producing fake or mismatched outputs.

---

## 2. The 10 Core Features (5 CV + 5 NLP)

### Computer Vision Features (5)

1. **Motion Keyframe Extraction:** Selects high-change frames using frame differencing and optical flow to skip redundant frames.
2. **ROI & Zone Tracking:** Tracks moving regions or objects inside defined polygon/box zones across frames.
3. **Visual Anomaly Scoring:** Flags visually unusual frames or surface/scene deviations relative to a baseline profile.
4. **On-Screen Text Detection:** Locates slide titles, labels, signs, or numeric panel regions for text extraction.
5. **Action State Indexing:** Classifies timeline segments into discrete motion states (`STATIC`, `ACTIVE`, `TRANSITION`, `BURST`).

### NLP Features (5)

1. **Transcript & Log Alignment:** Syncs time-stamped spoken segments, captions, or event logs with video time windows.
2. **Multimodal BM25/TF-IDF Retrieval:** Ranks video segments by matching query terms against transcripts, logs, and visual tags.
3. **Rule-Schema Event Parsing:** Extracts structured JSON records (`timestamp`, `entity`, `action`, `status`) via regex and domain rules.
4. **Entity Graph Linking:** Connects visual zones/objects and text entities in a graph to answer relationship queries.
5. **Grounding Consistency Check:** Verifies text claims or log entries against observed visual activity and timestamps to flag mismatches.

---

## 3. Target Industries & 6 Video Specifications (2 per Industry)

The three highest-volume industries analyzing video with CV + NLP are:

1. **Media, Education & Video Platforms**
2. **Manufacturing & Industrial Operations**
3. **Retail, Security & Smart Spaces**

Below are **6 concrete videos** (2 per industry) designed so that each video exercises a realistic subset of the 10 features without forcing mismatched features.

---

### Industry 1: Media, Education & Video Platforms

#### Video 1.1: `edu_01_slide_lecture.mp4` (+ accompanying `.srt`/`.vtt` transcript or lecture notes)

* **Content Description:** A 30 to 90 second educational tutorial or lecture clip showing slide transitions, bullet points or code/diagram text on screen, and a speaker explaining concepts.
* **Content Signals Present:** Slide/frame transitions, high-contrast text blocks, moderate presenter/cursor motion, aligned timestamped transcript.
* **Active Features (9 of 10):**
* **CV:** Motion Keyframe Extraction, Visual Anomaly Scoring (detects sudden slide/layout shifts), On-Screen Text Detection, Action State Indexing.
* **NLP:** Transcript & Log Alignment, Multimodal BM25/TF-IDF Retrieval, Rule-Schema Event Parsing (extracts definitions, steps, key terms), Entity Graph Linking (links terms, slides, sections), Grounding Consistency Check (checks if spoken key terms align with on-screen text/keyframes).


* **Skipped Feature (1 of 10):**
* **ROI & Zone Tracking:** Skipped (`NO_SPATIAL_ZONES`) unless specific screen regions (such as a live coding pane vs. slide area) are explicitly defined.



#### Video 1.2: `edu_02_hands_on_demo.mp4` (+ step-by-step lab transcript/notes)

* **Content Description:** A 30 to 90 second practical workbench or software/hardware demonstration with clear hand/tool/UI movement and spoken instructions, but minimal full-screen slides.
* **Content Signals Present:** Active tool/cursor/object motion, distinct action phases, spoken instructions synced to steps, occasional labels.
* **Active Features (8 of 10):**
* **CV:** Motion Keyframe Extraction, ROI & Zone Tracking (work area / tool zone), On-Screen Text Detection (where labels exist), Action State Indexing.
* **NLP:** Transcript & Log Alignment, Multimodal BM25/TF-IDF Retrieval, Rule-Schema Event Parsing, Entity Graph Linking.


* **Skipped Features (2 of 10):**
* **Visual Anomaly Scoring:** Skipped (`NO_STABLE_BASELINE`) when continuous demonstration motion prevents a clean stationary baseline, unless a reference normal frame is provided.
* **Grounding Consistency Check:** Skipped (`INSUFFICIENT_VISUAL_LABELS`) when spoken instructions describe abstract concepts not visually labeled on screen.



---

### Industry 2: Manufacturing & Industrial Operations

#### Video 2.1: `mfg_01_conveyor_inspection.mp4` (+ operator shift/QC log)

* **Content Description:** A fixed-camera clip of items moving along a conveyor or inspection station with a defined inspection zone, occasional defective/unusual items, and a text QC log.
* **Content Signals Present:** Fixed camera view, defined conveyor ROI zone, periodic item passage, visual deviations, timestamped QC log entries.
* **Active Features (9 of 10):**
* **CV:** Motion Keyframe Extraction, ROI & Zone Tracking (conveyor belt / inspection box), Visual Anomaly Scoring (flags color/shape/area outliers against normal items), Action State Indexing (`IDLE`, `MOVING`, `BURST`).
* **NLP:** Transcript & Log Alignment (aligns QC log timestamps to video), Multimodal BM25/TF-IDF Retrieval, Rule-Schema Event Parsing (parses station ID, part ID, status `PASS`/`DEFECT`), Entity Graph Linking (links part, station, defect code), Grounding Consistency Check (verifies whether logged defect timestamps match high visual anomaly / zone activity).


* **Skipped Feature (1 of 10):**
* **On-Screen Text Detection:** Skipped (`NO_READABLE_TEXT`) if camera resolution is too low or no gauges/labels appear in frames.



#### Video 2.2: `mfg_02_assembly_station.mp4` (+ SOP checklist / operator notes)

* **Content Description:** A 20 to 60 second clip of manual or semi-automated assembly at a single workstation with operator hand/part movement and written SOP steps.
* **Content Signals Present:** Workstation zone activity, distinct work/pause cycles, SOP step text, operator notes.
* **Active Features (8 of 10):**
* **CV:** Motion Keyframe Extraction, ROI & Zone Tracking (assembly zone), Action State Indexing (work vs. idle intervals).
* **NLP:** Transcript & Log Alignment, Multimodal BM25/TF-IDF Retrieval, Rule-Schema Event Parsing (extracts SOP step numbers and actions), Entity Graph Linking, Grounding Consistency Check (checks if logged steps occur during active motion windows rather than idle periods).


* **Skipped Features (2 of 10):**
* **Visual Anomaly Scoring:** Skipped (`HIGH_MOTION_VARIANCE`) when hand/part occlusion makes baseline comparison noisy.
* **On-Screen Text Detection:** Skipped (`NO_TEXT_REGION`) when no signage, part tags, or displays are visible.



---

### Industry 3: Retail, Security & Smart Spaces

#### Video 3.1: `ret_01_aisle_traffic.mp4` (+ store shift/footfall notes)

* **Content Description:** A wide-angle clip of a retail aisle, entrance, or shelf zone showing people/carts moving through zones and written shift notes.
* **Content Signals Present:** Defined aisle/shelf zones, entry/exit movement, crowd/activity bursts, timestamped shift notes.
* **Active Features (8 of 10):**
* **CV:** Motion Keyframe Extraction, ROI & Zone Tracking (aisle/shelf/entry zones), Action State Indexing (`QUIET`, `FLOW`, `CONGESTION`).
* **NLP:** Transcript & Log Alignment, Multimodal BM25/TF-IDF Retrieval, Rule-Schema Event Parsing (extracts zone, event type, count/severity), Entity Graph Linking (links zone, event, note), Grounding Consistency Check (verifies whether "busy/crowded" notes match elevated zone motion/keyframe density).


* **Skipped Features (2 of 10):**
* **Visual Anomaly Scoring:** Skipped (`DYNAMIC_SCENE_BASELINE`) in crowded multi-actor scenes where normal motion varies widely.
* **On-Screen Text Detection:** Skipped (`NO_LEGIBLE_SIGNAGE`) when price tags or aisle signs are too small or angled to detect reliably.



#### Video 3.2: `ret_02_facility_monitoring.mp4` (+ incident/security log)

* **Content Description:** A stationary monitoring clip of a doorway, corridor, or storage bay with a security/shift log recording events or alerts.
* **Content Signals Present:** Static background camera, clear entry/restricted zone, isolated motion events, structured incident log text.
* **Active Features (9 of 10):**
* **CV:** Motion Keyframe Extraction, ROI & Zone Tracking (doorway/restricted zone), Visual Anomaly Scoring (detects sudden scene/lighting/object changes against static background), Action State Indexing.
* **NLP:** Transcript & Log Alignment, Multimodal BM25/TF-IDF Retrieval, Rule-Schema Event Parsing (extracts time, location, alert/status), Entity Graph Linking, Grounding Consistency Check (verifies whether logged alerts coincide with zone motion or visual anomaly spikes).


* **Skipped Feature (1 of 10):**
* **On-Screen Text Detection:** Skipped (`NO_TEXT_REGION`) when no readable signs or timestamps exist inside the frame area.



---

## 4. Video-to-Feature Applicability Matrix

| Feature | `edu_01` Slide Lecture | `edu_02` Hands-On Demo | `mfg_01` Conveyor QC | `mfg_02` Assembly SOP | `ret_01` Retail Aisle | `ret_02` Facility Log |
| --- | --- | --- | --- | --- | --- | --- |
| **1. Motion Keyframe Extraction** | Yes | Yes | Yes | Yes | Yes | Yes |
| **2. ROI & Zone Tracking** | Conditional | Yes | Yes | Yes | Yes | Yes |
| **3. Visual Anomaly Scoring** | Yes | Conditional | Yes | Conditional | Conditional | Yes |
| **4. On-Screen Text Detection** | Yes | Conditional | Conditional | Conditional | Conditional | Conditional |
| **5. Action State Indexing** | Yes | Yes | Yes | Yes | Yes | Yes |
| **6. Transcript & Log Alignment** | Yes | Yes | Yes | Yes | Yes | Yes |
| **7. Multimodal BM25/TF-IDF Retrieval** | Yes | Yes | Yes | Yes | Yes | Yes |
| **8. Rule-Schema Event Parsing** | Yes | Yes | Yes | Yes | Yes | Yes |
| **9. Entity Graph Linking** | Yes | Yes | Yes | Yes | Yes | Yes |
| **10. Grounding Consistency Check** | Yes | Conditional | Yes | Yes | Yes | Yes |

Legend: **Yes** = Run by default when content signals are present; **Conditional** = Runs only if the content probe passes minimum signal/quality thresholds (otherwise cleanly skipped with a reason code).

---

## 5. Architecture & Code Execution Design

### Repository Layout

```text
videopulse_core/
  config/
    videos_manifest.json        # Paths, industry, ROIs, optional transcript/log paths
  core/
    media_probe.py              # FPS, resolution, duration, motion variance, text/audio presence
    feature_router.py           # Evaluates preconditions and decides which of the 10 features run
    cv_pipeline.py              # 5 lightweight OpenCV/NumPy CV features
    nlp_pipeline.py             # 5 lightweight BM25/TF-IDF/regex/NetworkX NLP features
    fusion_and_eval.py          # Cross-modal alignment, grounding checks, latency & skip metrics
  run_pipeline.py               # Single CLI entry point for all 6 videos
  outputs/
    <video_id>/report.json      # Structured output + active/skipped feature audit trail

```

### Execution & Gating Rules in Code

1. **Pre-Run Media Probe (`media_probe.py`):** Samples frames, measures mean frame-diff motion energy, checks edge/high-contrast region density (candidate text regions), validates whether an ROI polygon/box is defined, and checks whether a transcript or log file exists.
2. **Feature Gating (`feature_router.py`):** Each feature registers a `can_run(context) -> (bool, reason)` check. If `False`, the feature does not execute and appends `{"feature": name, "status": "SKIPPED", "reason": reason}` to the output report.
3. **Unified Output Schema (`report.json`):**
* `video_id`, `industry`, `duration_sec`, `processed_fps`
* `executed_features` and `skipped_features` (with explicit reasons)
* `keyframes`, `zones`, `anomalies`, `text_regions`, `action_states`
* `aligned_segments`, `parsed_events`, `graph_edges`, `grounding_checks`
* `telemetry` (`total_ms`, `per_feature_ms`, `frames_skipped_ratio`, `grounding_match_rate`)



---

## 6. Verification & Proof of Senior Engineering

To make this portfolio-grade without heavy tooling:

* **Deterministic Unit & Integration Tests:** Test all 10 features on small synthetic clips and logs, and verify that each of the 6 industry videos triggers the expected subset and skips mismatched features cleanly.
* **Grounding Audit Metric:** Report the percentage of parsed text/log events that have supporting visual activity or keyframe evidence within a configurable time window, plus explicit `UNGROUNDED_CLAIM` flags when text and video disagree.
* **Runtime & Efficiency Telemetry:** Log exact millisecond latency per feature and frame-skip ratio so reviewers see practical systems engineering alongside CV and NLP logic.

---

[OpenCV Python Crash Course](https://www.youtube.com/watch?v=x7n85SJMjUA)

This video is relevant because it covers foundational OpenCV video processing techniques in Python that support the lightweight computer vision features in this pipeline.