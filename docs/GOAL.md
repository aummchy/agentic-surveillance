# GOAL — Use Cases · What's Already Done · Improvements

> Where camera surveillance can be used, what already exists commercially, and where this project differentiates.

---

## 0. Priority — Pipeline First

**Before any new feature (zones, forensic search, PPE, crowd, sound…): first make the existing pipeline clear and properly working.**

1. **Stabilize** — one clean end-to-end flow: capture → detect → track → recognize → decide → alert → dashboard, verified with a real camera.
2. **Understand** — know exactly how each stage works (see §4 reference projects) and fix known issues in `docs/10 - Problems`.
3. **Only then change** — no new modules, no refactors, no feature work until the pipeline runs reliably and its behavior is predictable.

Rule: *pipeline correctness > new features.*

---

## 1. Where Camera Surveillance Can Be Used

Every use case below runs on **camera only** or **camera + sound** (audio adds: glass break, screams, gunshots, alarms, machine noise).

| # | Use case | What the system does | Needed beyond current stack |
|---|----------|----------------------|-----------------------------|
| 1 | **Forensic search** | Query: *Person · red shirt · 3 PM–4 PM · Camera 4* → returns relevant video segments instead of a guard reviewing 12 h of footage | Attribute metadata per event (clothing color, direction, zone) + structured/NL query over events |
| 2 | **Authenticated vs not** | Person matched → authenticated. If yes (clothing rule followed) → allow; if not → restrict. Alternative: vector search on camera face | Zone access rules + appearance/cloth attribute check |
| 3 | **Occupancy / crowd** | Count people in a place (crowd & public management); crowd graph over time; compare crowd now vs previous | Zone definitions + time-series aggregation + dashboard graph |
| 4 | **Retail (D-Mart etc.)** | How many customers entered, which section got most traffic, dwell heatmaps | **Zone tracking** on top of person tracking (entry/exit per section) |
| 5 | **Queue / line monitoring** (bank, airport) | Queue length, wait time, congestion alerts | Queue region definition + dwell/flow metrics |
| 6 | **Danger monitoring** | No helmet · no safety vest · person falls · too close to equipment · enters restricted production area | PPE detection (helmet/vest), fall detection, proximity + restricted-zone rules |

---

## 2. What Is Already Done (Commercial Reality)

**Most use cases above are already deployed commercially — they are not research ideas.**
Caveat: "implemented" ≠ works perfectly with any cheap camera — performance depends on placement, lighting, resolution, scene, model.

| Use case | Deployed? | Maturity |
|----------|-----------|----------|
| Person / vehicle detection | ✅ | Very mature |
| Person tracking | ✅ | Very mature |
| People counting / occupancy | ✅ | Mature |
| Intrusion / restricted zone / line crossing | ✅ | Very mature |
| Loitering / dwell time / tailgating | ✅ | Mature |
| Queue detection / traffic flow | ✅ | Mature |
| ANPR (license plates) | ✅ | Mature |
| Face recognition | ✅ | Mature, context-dependent |
| PPE / helmet detection | ✅ | Mature for defined scenes |
| Crowd density | ✅ | Mature |
| Re-identification | ✅ | Commercially available |
| Forensic video search | ✅ | Mature |
| Fall detection · fire/smoke | ✅ | Commercially available |
| Behavioral / anomaly detection | ✅ | Harder, less reliable |
| Natural-language video search | 🟡 | Emerging |
| Agentic reasoning over video events | ⚠️ | Emerging, not standardized |

Real vendors: **AXIS Object Analytics** (detection, counting, intrusion, restricted areas, loitering, occupancy, queue, dwell, wrong-way, tailgating) · **Johnson Controls** (face rec, LPR, behavioral metadata, forensic search) · Indian vendors (face rec + ANPR + PPE + intrusion + footfall + crowd on existing CCTV).

**Conclusion:** do not invent another "YOLO + face recognition CCTV" — that is a commodity.

---

## 3. What Improvement We Can Do (This Project)

### 3.1 The 5 levels of surveillance

| Level | Question | Example | Industry state |
|-------|----------|---------|----------------|
| 1 — Perception | What is visible? | Person, face, helmet | Very strong |
| 2 — Tracking | Where is it going? | Track 27: Gate → Corridor → Warehouse | Very strong |
| 3 — Events | What happened? | Entered restricted area | Strong |
| 4 — Context | Is it unusual? | Entry after working hours | Partial |
| 5 — Reasoning | What should be done? | Check history → access logs → incident → notify operator | **Our differentiation** |

### 3.2 Spectrum

- **Commoditized (boring alone):** person/vehicle detection, basic tracking, basic counting, intrusion, line crossing.
- **Still technically interesting:** multi-camera re-ID, long-term identity tracking, low-light/occlusion, false-alarm reduction, cross-camera correlation, NL video search, privacy-preserving, edge inference.
- **Most interesting for us:** event reasoning — turning raw events into understood incidents.

### 3.3 Already done in this project

Perception → identity → policy → alerts → LLM already built (see [[FEATURES]]):

- YOLOv8 + ByteTrack tracking, InsightFace embeddings + Atlas vector search, quality gates
- 9-level `Status` enum, policy rules R1–R9, visit memory, confidence scoring
- Alerts (console/email/SMS/webhook), LLM summaries + incident/person/daily reports, dashboard + NL chat

→ **Levels 1–3 covered, parts of Level 4** (after-hours rules, memory context).

### 3.4 Improvements to add

1. **Zone tracking** — overlay zones on the frame; per-zone entry/exit/dwell. Unlocks: occupancy counting, retail section traffic, restricted production areas, queue monitoring — all from one primitive.
2. **Attribute / forensic search** — store clothing color + attributes per event; query "red shirt, 3–4 PM, Camera 4"; NL search over event metadata.
3. **Safety module** — helmet/vest (PPE) detection, fall detection, proximity-to-equipment rules → danger alerts.
4. **Crowd analytics** — occupancy time-series, crowd graph, now-vs-previous comparison.
5. **Agentic event reasoning (Level 5)** — replace `"Person detected, Track 47, Zone 3, 23:42"` with:
   > "An unknown person entered the restricted warehouse at 23:42 and remained 6 minutes."
   then: search previous events → has this person appeared before? → check access records → check other cameras → generate incident report → notify.
6. **Sound channel (optional)** — camera + microphone: anomaly sounds (glass break, scream, alarm) as an extra alert signal.

### 3.5 Target architecture

```text
Camera → Vision → Events → Memory → Reasoning → Action
```

Use the existing CV stack as the **perception layer**; build the differentiating part above it: long-term event memory, context, reasoning, and action — an **operational system**, not a model demo.

---

## 4. Reference — How Recognition Pipelines Actually Work (Open Source)

See **[[REFERENCES]]** (`docs/REFERENCES.md`) — curated open-source projects to study before changing our pipeline: DeepFace, InsightFace, CompreFace, Frigate (zones), high-star video building blocks, and anti-spoofing/deepfake defenses (§4–§5).
