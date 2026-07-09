# Professor Feedback Notes
## Verbatim key points organized for revision

**Source:** "Changes to research paper" Google Doc (Drive ID: 1-xBkJVoiux7mu0zZbvLK5dY4mXtVzON5BhPG3buk48w)  
**Rounds:** Two feedback rounds

---

## Feedback Round 1 — Structural

> "The biggest issue is that the paper currently reads like multiple individual mini-papers stitched together instead of one unified journal-style research paper."

1. Technical Contributions → end of Introduction (not its own major section)
2. System Overview → own section AFTER Related Work
3. Architecture figure → NOT in Introduction, top of System Overview section
4. One combined Motivation section only
5. Remove individual author names from section titles
6. Individual Author Contributions → after References, not in body
7. Remove bug-fix narratives (belong in GitHub, not the paper)
8. Combine all motivation sections → one

## Feedback Round 2 — Content/Results

> "Your professor is saying: Show your work with real numbers, not claims. Concrete measurements (even if modest) beat vague estimates every time."

### Priority 1 — Real Detection Performance (HIGHEST)
> "Use Rice_Classification dataset with proper 80/10/10 train/val/test split. Run pycocotools mAP on held-out test set. Report AP@0.5 and AP@0.75. Even 40% is fine — it shows the model actually learned something. This is the single most important fix."

### Priority 2 — Detection Visualizations
> "Pick 3 representative images: one dense paddy, one aerial view, one post-flood image. Draw bounding boxes with confidence scores. One good detection visualization figure is worth more than like two pages of you describing scores."

### Priority 3 — ROI Compute Reduction
> "Run inference on ~50 images with and without the ROI mask. Time both runs. Report actual measured number. Much stronger than repeated 'estimated' claims."

### Priority 4 — GNSS Position Error
> "You already have the data from the 20-second outage simulation (Figure 3). Compare EKF estimated position at t=35s against ground truth. Report: 'Position error was X.XX meters after 20 seconds of dead reckoning.'"

### Priority 5 — Fusion Bridge Status Table
> "Add a small table showing mechanism/status. Be upfront about what's done and what's not — it's way more credible than being vague."

---

## Phrases to Avoid (AI tells flagged by professor)
- "robust framework"
- "seamlessly integrates"
- "leverages cutting-edge"
- "plays a crucial role"
- "significantly enhances"
- "novel and comprehensive approach"
- "underscores the importance"
- "persistent challenges"
- "primary system-level contribution"
- "zero additional hardware cost" (repeated too often)

## Replacement Style
> "This paper presents a rice detection pipeline using annotated field images and a ResNet-based object detection model. The system focuses on binary detection of rice plants under noisy paddy-field conditions."

---

## Status Table (Professor's Assessment)

| Request | Status |
|---|---|
| Remove bug-fix narratives | ✅ Done |
| Remove author names from section titles | ✅ Done |
| Remove unimplemented GCN section | ✅ Done |
| Move Individual Author Contributions out of body | ⚠️ Half done (removed from body, not yet added at end) |
| Technical Contributions at end of Introduction | ⚠️ Partly — still comes after System Overview |
| One Motivation section | ❌ Not done |
| Related Work before System Overview | ❌ Not done |
| Move architecture figure out of Introduction | ❌ Not done |
| Redraw architecture in draw.io | ❓ Unknown |
| Convert modules to subsections | ❌ Not done |
| Unified Experiments/Results section | ❌ Not done |
| Rewrite AI-sounding abstract/intro/conclusion | ⚠️ Partly done |
| Same-distribution mAP | ❌ Not done — BLOCKING |
