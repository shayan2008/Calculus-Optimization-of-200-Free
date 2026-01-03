# 200m Freestyle (LCM) Optimizer + Race Analyzer (Calculus/Physiology Modeling)

A publication-style, coach-grade **Streamlit mini-app** for analyzing and fitting **200m freestyle (LCM)** race plans using a simplified but structured **physics + physiology** performance model.

This tool is built to support **elite / high-level age group swimmers and coaches** who want to:
- test pacing strategies (fast-out, even-ish, negative split),
- quantify how **underwaters vs surface pacing** affect total time,
- evaluate feasibility using **CV, D′, and lactate dynamics**,
- and export results as **PNG / CSV / PDF**.

> Important: Default parameter values are illustrative. You get meaningful athlete-specific results by fitting parameters from real data (trial times, 15m timings, stroke metrics, optional lactate samples).

---

## What’s inside (everything the app does)

### Core capabilities
- **Race decomposition** (LCM 200 free):
  - Start + underwater (0–15 m)
  - Surface swimming
  - Turns + underwater (0–15 m) × 3
  - Finish
- **State-based simulation** across the full 200 m:
  - velocity profile vs distance
  - **D′ reserve** evolution (spend/recover relative to CV)
  - **lactate proxy** evolution (production + clearance)
- **Strategy system**
  - Presets: **Fast-out**, **Even-ish**, **Negative-split**
  - Custom strategy: set surface speeds per 50 + underwater distances + push-off speeds

### Two “analysis modes” you asked for
1) **Target Time → Fit Plan**
   - Enter a **goal time** (`1:47.23` or `107.23`)
   - Pick a base pacing shape (fast-out / even-ish / negative split)
   - App fits a scaling factor **k** to hit the target time by scaling surface speeds
   - Outputs:
     - predicted 50 splits
     - min D′, peak lactate proxy
     - feasibility warnings (D′ bottoming out)

2) **Real Splits → Analyze**
   - Enter your **4×50 splits**
   - Provide underwater distances + push speeds (or leave defaults)
   - The app reconstructs surface speeds per lap by subtracting estimated underwater time
   - Outputs:
     - pacing classification (positive / even-ish / negative split)
     - inferred surface speeds per 50
     - “underwater time-positive?” diagnostic per wall
     - plots + feasibility

### Exports
- Downloadable **workflow diagram (PNG)**
- Downloadable **results table (CSV)**
- Downloadable **mini report (PDF)**:
  - includes latest analysis table
  - includes plots + workflow diagram

### App-like features
- Parameter panel in sidebar (CV, D′, lactate, underwater decay)
- Optional save/export of parameter settings (**JSON download**)
- Optional load of parameter settings (**JSON upload**)
- Multi-tab layout:
  - Simulator
  - Target Time → Fit Plan
  - Real Splits → Analyze
  - Exports

---

## Screens / Outputs (what you will see)
- **Splits table** (50/100/150/200 + total)
- **Velocity vs distance plot**
- **Lactate proxy + D′ vs time plot**
- **Lap diagnostics table**:
  - UW distance, UW time, UW average speed, surface speed estimate
  - “UW time-positive?” quick indicator

---

## Mathematical model (high-level, but explicit)

### Underwater speed decay (distance-domain)
We model underwater velocity as an exponential decay with distance:

\[
v_u(x) = v_\infty + (v_0 - v_\infty)\exp(-x/d_c)
\]

where:
- \(v_0\) is push-off initial speed,
- \(v_\infty\) is asymptotic underwater speed,
- \(d_c\) is a decay length constant.

Underwater time is computed by integrating forward in small steps.

### D′ (anaerobic reserve) balance
We treat D′ as a limited “above-CV budget”:
- If speed \(v > CV\), D′ is spent
- If speed \(v \le CV\), D′ partially recovers toward \(D′_0\)

This is a simplified discrete dynamic model intended for pacing feasibility diagnostics.

### Lactate proxy dynamics
We model lactate as a one-compartment proxy:

\[
\frac{dL}{dt} = \alpha\max(0, I-1)^\gamma - k_{clr}(L-L_0)
\quad \text{where} \quad I = v/CV
\]

Interpretation: higher intensity above CV drives nonlinear production; clearance pulls back toward baseline.

---

## Data requirements (for real athlete use)

### Minimum viable dataset (to get useful outputs)
- 200 free **4×50 splits**
- Underwater breakout distances per wall (start + 3 turns), OR at least a
