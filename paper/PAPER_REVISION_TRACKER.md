# Paper Revision Tracker
## AgriNav — IEEE Conference Paper

**Format:** IEEE conference, LaTeX on Overleaf  
**Last updated:** 2026-06-01

---

## ✅ Completed Revisions

- [x] Removed Anthony Raphael / mission controller from all sections and contribution list
- [x] Removed GCN section and related-work subsection
- [x] Fixed broken LaTeX syntax in author contributions block
- [x] Removed AI-tone markers from abstract, introduction, conclusion
- [x] Removed bug-fix narrative from WeedDet section
- [x] Removed individual author names from section titles and body
- [x] Removed duplicate motivation/background sections from individual modules
- [x] Replaced AI-generated architecture figure with draw.io version *(confirm final status)*

---

## ⚠️ Open — Structural (Required by Professor)

- [ ] **System Overview still inside Introduction** → Move to new `\section{System Architecture}` AFTER Related Work
- [ ] **Architecture figure still in Introduction** → Move to top of System Architecture section
- [ ] **No standalone Motivation section** → Add `\section{Motivation}` after Introduction, before Related Work
- [ ] **Module sections are `\section{}`** → Convert all to `\subsection{}` under System Architecture
- [ ] **No unified Experiments/Results section** → Create `\section{Experiments and Results}` with all results
- [ ] **Author Contributions not at end** → Add `\section*{Author Contributions}` after `\end{thebibliography}`
- [ ] **Abstract inconsistency** → Says "four modules", Introduction says "three" — fix one

---

## ⚠️ Open — Content (Required by Professor)

- [ ] **Same-distribution mAP (HIGHEST PRIORITY)** — AP@0.5 and AP@0.75 required. Run WeedDet v5 Cell 6. Report even 40% — shows model learned something.
- [ ] **Detection visualization figure** — 3 representative images (dense paddy, aerial, post-flood) with bounding boxes and confidence scores drawn. "One good detection visualization figure is worth more than two pages describing scores."
- [ ] **Discrimination logic text has contradiction** — Current text says "predicted class must not equal rice" in context of rice detections. Replace with: hard veto = any footprint overlapping rice detection with IACS ≥ 0.5 gets spray command rejected.
- [ ] **LiDAR-camera bridge status table** — Add small table showing which of the 4 mechanisms are implemented vs. proposed
- [ ] **ROS version inconsistency** — Detection section says "ROS 2", navigation says "ROS Noetic" (ROS 1). Pick one or explain the split.
- [ ] **"GPU training resolves this"** — Too confident. Replace with "GPU training is expected to improve localization, but must be measured."
- [ ] **"Weeds physically cannot grow on row surfaces"** — Biologically too absolute. Replace with inter-row treatment zone softer framing.
- [ ] **ROI compute reduction "estimated 30–50%"** — Replace with actual measured number OR remove the claim.
- [ ] **GNSS drift number** — Ask Krish/Bilal for actual position error (meters) after 20-second outage simulation.

---

## Paper Structure Target

```latex
\section{Introduction}
  \subsection{Technical Contributions}   % at END of intro

\section{Motivation}                     % one unified section

\section{Related Work}

\section{System Architecture}            % architecture figure at TOP
  \subsection{WeedDet-Based Rice Detection}
    \subsubsection{Architecture}
    \subsubsection{Loss Formulation}
    \subsubsection{Data Preparation}
    \subsubsection{Training Configuration}
  \subsection{Lightweight CNN-FPN Detection Pipeline}
  \subsection{LiDAR Crop Row Detection}
  \subsection{EKF-Based Localization and Row Following}
  \subsection{System Integration}

\section{Experiments and Results}
  \subsection{WeedDet Training and Qualitative Inference}
  \subsection{Lightweight CNN-FPN Detection Results}
  \subsection{LiDAR Row Detection Results}
  \subsection{GNSS Outage Bridging Results}
  \subsection{System-Level Summary}

\section{Limitations}
\section{Future Work}
\section{Conclusion}

\begin{thebibliography}{00}
...
\end{thebibliography}

\section*{Author Contributions}
```

---

## Priority Order for Next Revision Session

1. Run v5 → get mAP numbers (unblocks everything)
2. Move System Overview out of Introduction
3. Add standalone Motivation section
4. Convert module sections to subsections
5. Create unified Experiments/Results section
6. Add Author Contributions after References
7. Fix abstract "four vs three modules"
8. Fix discrimination logic text
9. Add detection visualization figure (3 images)
10. Add LiDAR-camera bridge status table
