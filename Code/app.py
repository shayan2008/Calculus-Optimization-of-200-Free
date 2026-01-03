import io
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# PDF export
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# =========================
# Core model
# =========================
@dataclass
class Params:
    # Physiology
    CV: float = 1.78          # m/s (critical velocity)
    Dprime0: float = 26.0     # m-equivalent anaerobic reserve
    tau_rec: float = 40.0     # s (recovery time constant for D′)
    # Lactate proxy
    L0: float = 1.2           # mmol/L baseline
    alpha: float = 10.0       # lactate production scale
    gamma: float = 2.6        # lactate nonlinearity
    k_clear: float = 0.015    # 1/s lactate clearance
    # Underwater
    uw_vinf: float = 1.55     # m/s asymptotic underwater speed
    uw_dc: float = 7.5        # m underwater decay length


@dataclass
class Strategy:
    seg_v: List[float]        # surface target speeds per 50 (m/s)
    uw_dist: List[float]      # underwater distance per wall (m), 0–15
    uw_v0: List[float]        # initial push speed per wall (m/s)


def v_uw(x: float, v0: float, v_inf: float, d_c: float) -> float:
    return v_inf + (v0 - v_inf) * math.exp(-x / d_c)


def simulate_200_free(strategy: Strategy, p: Params, dt: float = 0.1):
    """
    Returns:
      ts, xs, vs, Dps, Ls, split_times (cumulative times at 50/100/150/200)
    """
    t = 0.0
    x_total = 0.0
    Dp = p.Dprime0
    L = p.L0

    ts, xs, vs, Dps, Ls = [], [], [], [], []
    split_times = []

    for lap in range(4):
        # ---- Underwater phase ----
        uw = float(strategy.uw_dist[lap])
        v0 = float(strategy.uw_v0[lap])

        x = 0.0
        while x < uw - 1e-9:
            v = v_uw(x, v0, p.uw_vinf, p.uw_dc)

            dx = v * dt
            if x + dx > uw:
                dt_eff = (uw - x) / v
                dx = uw - x
            else:
                dt_eff = dt

            # D′ balance
            if v > p.CV:
                Dp -= (v - p.CV) * dt_eff
            else:
                Dp += (p.CV - v) * (p.Dprime0 - Dp) / p.Dprime0 * (dt_eff / p.tau_rec)
            Dp = float(np.clip(Dp, 0.0, p.Dprime0))

            # Lactate proxy
            I = v / p.CV
            prod = p.alpha * max(0.0, I - 1.0) ** p.gamma
            L += (prod - p.k_clear * (L - p.L0)) * dt_eff

            t += dt_eff
            x_total += dx
            x += dx

            ts.append(t); xs.append(x_total); vs.append(v); Dps.append(Dp); Ls.append(L)

        # ---- Surface phase ----
        surf_dist = 50.0 - uw
        v_s = float(strategy.seg_v[lap])

        x = 0.0
        while x < surf_dist - 1e-9:
            v = v_s

            dx = v * dt
            if x + dx > surf_dist:
                dt_eff = (surf_dist - x) / v
                dx = surf_dist - x
            else:
                dt_eff = dt

            if v > p.CV:
                Dp -= (v - p.CV) * dt_eff
            else:
                Dp += (p.CV - v) * (p.Dprime0 - Dp) / p.Dprime0 * (dt_eff / p.tau_rec)
            Dp = float(np.clip(Dp, 0.0, p.Dprime0))

            I = v / p.CV
            prod = p.alpha * max(0.0, I - 1.0) ** p.gamma
            L += (prod - p.k_clear * (L - p.L0)) * dt_eff

            t += dt_eff
            x_total += dx
            x += dx

            ts.append(t); xs.append(x_total); vs.append(v); Dps.append(Dp); Ls.append(L)

        split_times.append(t)

    return np.array(ts), np.array(xs), np.array(vs), np.array(Dps), np.array(Ls), split_times


def splits_from_cumulative(split_times: List[float]) -> List[float]:
    prev = 0.0
    out = []
    for t in split_times:
        out.append(t - prev)
        prev = t
    return out


# =========================
# Parsing + fitting helpers
# =========================
def parse_time_to_seconds(s: str) -> float:
    """
    Accepts:
      "1:47.23" -> 107.23
      "107.23"  -> 107.23
      "1:47"    -> 107.00
    """
    s = s.strip()
    if ":" in s:
        mm, ss = s.split(":")
        return float(mm) * 60.0 + float(ss)
    return float(s)


def underwater_time_for_lap(uw_dist: float, uw_v0: float, p: Params, dt: float = 0.02) -> float:
    x = 0.0
    t = 0.0
    while x < uw_dist - 1e-9:
        v = v_uw(x, uw_v0, p.uw_vinf, p.uw_dc)
        dx = v * dt
        if x + dx > uw_dist:
            dt_eff = (uw_dist - x) / v
            dx = uw_dist - x
        else:
            dt_eff = dt
        t += dt_eff
        x += dx
    return t


def build_strategy_from_splits(
    splits_4x50: List[float],
    uw_dist: List[float],
    uw_v0: List[float],
    p: Params
) -> Strategy:
    """
    Given 4 x 50 splits (s), UW distances, and push speeds,
    estimate surface speeds per lap by subtracting UW time from the split.
    """
    seg_v = []
    for i in range(4):
        t_uw = underwater_time_for_lap(uw_dist[i], uw_v0[i], p, dt=0.02)
        t_surface = max(0.01, splits_4x50[i] - t_uw)
        dist_surface = 50.0 - uw_dist[i]
        seg_v.append(dist_surface / t_surface)
    return Strategy(seg_v=seg_v, uw_dist=uw_dist, uw_v0=uw_v0)


def fit_strategy_to_total_time(
    base: Strategy,
    p: Params,
    target_time: float,
    dt: float = 0.10,
    lo: float = 0.80,
    hi: float = 1.35,
    iters: int = 45
):
    """
    Fits a scalar multiplier k so seg_v_scaled = k*base.seg_v reaches total time ~= target_time.
    Underwaters held fixed.
    Returns: (scaled_strategy, k, achieved_time, bracketed)
    """
    def total_time_for(k: float) -> float:
        s = Strategy(
            seg_v=[k * v for v in base.seg_v],
            uw_dist=base.uw_dist[:],
            uw_v0=base.uw_v0[:],
        )
        *_, split_times = simulate_200_free(s, p, dt=dt)
        return split_times[-1]

    t_lo = total_time_for(lo)
    t_hi = total_time_for(hi)

    # Typically: higher k -> faster -> smaller time
    # If target not bracketed, we return boundary
    bracketed = (t_hi <= target_time <= t_lo)
    if not bracketed:
        best_k = lo if target_time > t_lo else hi
        best_t = total_time_for(best_k)
        scaled = Strategy(
            seg_v=[best_k * v for v in base.seg_v],
            uw_dist=base.uw_dist[:],
            uw_v0=base.uw_v0[:],
        )
        return scaled, best_k, best_t, False

    a, b = lo, hi
    for _ in range(iters):
        mid = 0.5 * (a + b)
        t_mid = total_time_for(mid)
        if t_mid > target_time:
            a = mid
        else:
            b = mid

    best_k = 0.5 * (a + b)
    best_t = total_time_for(best_k)
    scaled = Strategy(
        seg_v=[best_k * v for v in base.seg_v],
        uw_dist=base.uw_dist[:],
        uw_v0=base.uw_v0[:],
    )
    return scaled, best_k, best_t, True


def pacing_type_from_splits(splits_4x50: List[float]) -> str:
    first100 = splits_4x50[0] + splits_4x50[1]
    last100 = splits_4x50[2] + splits_4x50[3]
    if last100 < first100 - 0.30:
        return "negative split"
    if last100 > first100 + 0.30:
        return "positive split"
    return "even-ish"


# =========================
# “App-like” visuals + exports
# =========================
def make_workflow_diagram_png() -> bytes:
    from matplotlib.patches import FancyBboxPatch

    border = "#0F2747"
    fill = "#F6FAFF"
    accent = "#9CB3D9"
    textc = "#0F2747"

    fig = plt.figure(figsize=(12, 7))
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def add_box(x, y, w, h, text, fontsize=13):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=2,
            edgecolor=border,
            facecolor=fill
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=textc, family="DejaVu Sans")
        return (x, y, w, h)

    def arrow(x1, y1, x2, y2, color=border, style="-", lw=2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=lw, color=color, linestyle=style, shrinkA=6, shrinkB=6)
        )

    b1 = add_box(0.06, 0.76, 0.38, 0.16, "Video + timing\n(0–15m, 15–45m,\nturn-in/turn-out)")
    b2 = add_box(0.56, 0.76, 0.38, 0.16, "Stroke metrics\n(SR, SL, stroke count,\ntempo per 25/50)")

    b3 = add_box(0.06, 0.49, 0.38, 0.16, "Physiology\n(CV from trials,\noptional lactate samples)")
    b4 = add_box(0.56, 0.49, 0.38, 0.16, "Model fitting\n(uw decay, SR–SL,\nlactate params)")

    b5 = add_box(0.31, 0.20, 0.38, 0.16, "Optimization\n(pacing + UW + SR)")

    arrow(b1[0]+b1[2], b1[1]+b1[3]/2, b2[0], b2[1]+b2[3]/2)
    arrow(b2[0]+b2[2]/2, b2[1], b4[0]+b4[2]/2, b4[1]+b4[3])
    arrow(b3[0]+b3[2], b3[1]+b3[3]/2, b4[0], b4[1]+b4[3]/2)
    arrow(b4[0]+b4[2]/2, b4[1], b5[0]+b5[2]/2, b5[1]+b5[3])
    arrow(b3[0]+b3[2]/2, b3[1], b5[0]+0.02, b5[1]+b5[3], color=accent, style="--", lw=2)

    ax.text(0.5, 0.97, "Data → Model → Optimization Workflow (200m Freestyle LCM)",
            ha="center", va="center", fontsize=14, color=border, fontweight="bold", family="DejaVu Sans")
    ax.text(0.5, 0.08, "Validate vs race splits and 15m times; iterate parameters.",
            ha="center", va="center", fontsize=12, color="#2b2b2b", family="DejaVu Sans")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_pdf_summary(
    title: str,
    splits_df: pd.DataFrame,
    figs: List[Tuple[str, bytes]]
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 50, title)
    c.setFont("Helvetica", 10)
    c.drawString(40, h - 70, "Generated by the 200m Free Optimizer app (illustrative model outputs).")

    # Table
    y = h - 110
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Splits summary (s)")
    y -= 18

    cols = list(splits_df.columns)
    # dynamic column spacing (simple)
    col_x = [40, 160, 230, 300, 370, 450]
    col_x = col_x[:len(cols)]

    c.setFont("Helvetica-Bold", 9)
    for i, col in enumerate(cols):
        c.drawString(col_x[i], y, str(col))
    y -= 14

    c.setFont("Helvetica", 9)
    for _, row in splits_df.iterrows():
        for i, col in enumerate(cols):
            c.drawString(col_x[i], y, str(row[col]))
        y -= 12
        if y < 110:
            c.showPage()
            y = h - 60

    # Figures (each on new page)
    for name, fb in figs:
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, h - 45, name)
        img = ImageReader(io.BytesIO(fb))
        margin = 40
        c.drawImage(img, margin, margin, width=w - 2*margin, height=h - 2*margin - 20,
                    preserveAspectRatio=True, anchor='c')

    c.save()
    buf.seek(0)
    return buf.read()


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="200 Free Optimizer", layout="wide")
st.title("200m Freestyle (LCM) — Optimizer + Race Analyzer")
st.caption("Enter a target time or real splits → get pacing, underwaters, lactate/D′ feasibility + exports.")

# Presets
presets: Dict[str, Strategy] = {
    "Fast-out": Strategy(
        seg_v=[1.98, 1.90, 1.84, 1.78],
        uw_dist=[15.0, 11.0, 10.0, 9.0],
        uw_v0=[3.20, 2.60, 2.55, 2.50],
    ),
    "Even-ish": Strategy(
        seg_v=[1.94, 1.90, 1.88, 1.86],
        uw_dist=[15.0, 10.5, 10.0, 10.0],
        uw_v0=[3.20, 2.60, 2.55, 2.55],
    ),
    "Negative-split": Strategy(
        seg_v=[1.90, 1.88, 1.90, 1.92],
        uw_dist=[15.0, 10.0, 10.0, 10.5],
        uw_v0=[3.20, 2.60, 2.55, 2.60],
    ),
}

# Sidebar: parameters
st.sidebar.header("Physiology + underwater parameters")
p = Params(
    CV=st.sidebar.number_input("CV (m/s)", 1.2, 2.5, 1.78, 0.01),
    Dprime0=st.sidebar.number_input("D′0 (m-equivalent)", 5.0, 100.0, 26.0, 0.5),
    tau_rec=st.sidebar.number_input("tau_rec (s)", 5.0, 200.0, 40.0, 1.0),
    L0=st.sidebar.number_input("L0 baseline lactate (mmol/L)", 0.5, 4.0, 1.2, 0.1),
    alpha=st.sidebar.number_input("alpha lactate production scale", 0.1, 50.0, 10.0, 0.5),
    gamma=st.sidebar.number_input("gamma lactate nonlinearity", 1.0, 6.0, 2.6, 0.1),
    k_clear=st.sidebar.number_input("k_clear (1/s)", 0.001, 0.2, 0.015, 0.001),
    uw_vinf=st.sidebar.number_input("uw v_inf (m/s)", 0.8, 2.2, 1.55, 0.01),
    uw_dc=st.sidebar.number_input("uw decay length d_c (m)", 2.0, 20.0, 7.5, 0.1),
)

dt = st.sidebar.slider("Simulation dt (s)", 0.02, 0.25, 0.10, 0.01)

st.sidebar.header("Profile save/load")
profile_name = st.sidebar.text_input("Profile name", value="default_profile")
if st.sidebar.button("Download profile JSON"):
    st.sidebar.download_button(
        "Click to download",
        data=json.dumps(asdict(p), indent=2),
        file_name=f"{profile_name}_params.json",
        mime="application/json",
        key="dl_profile"
    )

uploaded = st.sidebar.file_uploader("Load params JSON", type=["json"])
if uploaded is not None:
    try:
        loaded = json.load(uploaded)
        for k in loaded:
            if hasattr(p, k):
                setattr(p, k, float(loaded[k]))
        st.sidebar.success("Loaded params (you may need to re-open sidebar sliders to reflect values).")
    except Exception as e:
        st.sidebar.error(f"Could not load: {e}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Simulator", "Target Time → Fit Plan", "Real Splits → Analyze", "Exports"])

# =========================
# TAB 1: Simulator
# =========================
with tab1:
    st.subheader("Simulator (presets + custom)")

    mode = st.selectbox("Choose a pacing template", ["Fast-out", "Even-ish", "Negative-split", "Custom"], index=1)

    if mode != "Custom":
        strat = presets[mode]
    else:
        c = st.columns(3)
        with c[0]:
            st.markdown("**Surface speeds (m/s)**")
            seg_v = [
                st.number_input("v_50 #1", 1.2, 2.5, 1.94, 0.01),
                st.number_input("v_50 #2", 1.2, 2.5, 1.90, 0.01),
                st.number_input("v_50 #3", 1.2, 2.5, 1.88, 0.01),
                st.number_input("v_50 #4", 1.2, 2.5, 1.86, 0.01),
            ]
        with c[1]:
            st.markdown("**Underwater distances (m)**")
            uw_dist = [
                st.number_input("UW start", 0.0, 15.0, 15.0, 0.1),
                st.number_input("UW turn 1", 0.0, 15.0, 10.5, 0.1),
                st.number_input("UW turn 2", 0.0, 15.0, 10.0, 0.1),
                st.number_input("UW turn 3", 0.0, 15.0, 10.0, 0.1),
            ]
        with c[2]:
            st.markdown("**Push speeds v0 (m/s)**")
            uw_v0 = [
                st.number_input("v0 start", 1.5, 4.0, 3.2, 0.05),
                st.number_input("v0 turn 1", 1.5, 4.0, 2.6, 0.05),
                st.number_input("v0 turn 2", 1.5, 4.0, 2.55, 0.05),
                st.number_input("v0 turn 3", 1.5, 4.0, 2.55, 0.05),
            ]
        strat = Strategy(seg_v=seg_v, uw_dist=uw_dist, uw_v0=uw_v0)

    ts, xs, vs, Dps, Ls, split_times = simulate_200_free(strat, p, dt=dt)
    splits = splits_from_cumulative(split_times)
    total = split_times[-1]

    cols = st.columns(4)
    cols[0].metric("Total (s)", f"{total:.2f}")
    cols[1].metric("Min D′", f"{Dps.min():.2f}")
    cols[2].metric("Peak lactate proxy", f"{Ls.max():.2f}")
    cols[3].metric("Pacing type", pacing_type_from_splits(splits))

    df = pd.DataFrame([{
        "Strategy": mode,
        "50": f"{splits[0]:.2f}",
        "100": f"{splits[1]:.2f}",
        "150": f"{splits[2]:.2f}",
        "200": f"{splits[3]:.2f}",
        "Total": f"{total:.2f}",
    }])
    st.dataframe(df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = plt.figure(figsize=(7, 4))
        plt.plot(xs, vs)
        plt.xlabel("Distance (m)")
        plt.ylabel("Velocity (m/s)")
        plt.title("Velocity vs Distance")
        st.pyplot(fig)
        v_fig = fig_to_png_bytes(fig)
    with c2:
        fig = plt.figure(figsize=(7, 4))
        plt.plot(ts, Ls, label="L(t)")
        plt.plot(ts, Dps, label="D′(t)")
        plt.xlabel("Time (s)")
        plt.title("Lactate + D′ vs Time")
        plt.legend()
        st.pyplot(fig)
        ld_fig = fig_to_png_bytes(fig)

    # Quick diagnostic tips
    st.markdown("**Quick model-based diagnostics**")
    if Dps.min() <= 0.2:
        st.warning("D′ nearly hits zero → in this model that usually means your early speed is too high for your current CV/D′.")
    if Ls.max() > (p.L0 + 10.0):
        st.warning("Very high lactate proxy → likely technique decay risk (SL collapse) late race in this model.")
    st.info("For athlete-specific accuracy: fit CV/D′, underwater decay, and lactate parameters from real trials.")

# =========================
# TAB 2: Target time fitter
# =========================
with tab2:
    st.subheader("Enter a target total time → Fit a plan + feasibility")

    tcol1, tcol2, tcol3 = st.columns([1.2, 1.2, 1.0])
    with tcol1:
        time_str = st.text_input("Target time (e.g., 1:47.23 or 107.23)", value="1:47.23")
    with tcol2:
        base_name = st.selectbox("Base strategy shape", ["Fast-out", "Even-ish", "Negative-split"], index=1)
    with tcol3:
        fit_button = st.button("Fit plan to target time", type="primary")

    st.caption("This fits a single scalar k on surface speeds (seg_v). Underwaters are held fixed unless you edit them in the simulator.")

    if fit_button:
        target_sec = parse_time_to_seconds(time_str)
        base = presets[base_name]

        fitted, k, achieved, bracketed = fit_strategy_to_total_time(
            base, p, target_sec, dt=dt,
            lo=st.slider("k min (slower)", 0.50, 1.20, 0.80, 0.01),
            hi=st.slider("k max (faster)", 0.90, 1.80, 1.35, 0.01),
            iters=45
        )

        tsF, xsF, vsF, DpF, LF, split_timesF = simulate_200_free(fitted, p, dt=dt)
        splitsF = splits_from_cumulative(split_timesF)

        if bracketed:
            st.success(f"Fit complete: k = {k:.4f} → achieved {achieved:.2f}s "
                       f"({int(achieved//60)}:{achieved%60:05.2f}).")
        else:
            st.warning(f"Target not bracketed by k-range. Returned closest boundary: k = {k:.4f} → {achieved:.2f}s.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Target (s)", f"{target_sec:.2f}")
        m2.metric("Achieved (s)", f"{achieved:.2f}")
        m3.metric("Min D′", f"{DpF.min():.2f}")
        m4.metric("Peak L", f"{LF.max():.2f}")

        df_fit = pd.DataFrame([{
            "BaseShape": base_name,
            "k(scale)": f"{k:.4f}",
            "50": f"{splitsF[0]:.2f}",
            "100": f"{splitsF[1]:.2f}",
            "150": f"{splitsF[2]:.2f}",
            "200": f"{splitsF[3]:.2f}",
            "Total": f"{split_timesF[-1]:.2f}",
        }])
        st.dataframe(df_fit, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = plt.figure(figsize=(7, 4))
            plt.plot(xsF, vsF)
            plt.xlabel("Distance (m)")
            plt.ylabel("Velocity (m/s)")
            plt.title("Fitted plan: velocity vs distance")
            st.pyplot(fig)
            fit_vel_fig = fig_to_png_bytes(fig)
        with c2:
            fig = plt.figure(figsize=(7, 4))
            plt.plot(tsF, LF, label="L(t)")
            plt.plot(tsF, DpF, label="D′(t)")
            plt.xlabel("Time (s)")
            plt.title("Fitted plan: lactate + D′")
            plt.legend()
            st.pyplot(fig)
            fit_ld_fig = fig_to_png_bytes(fig)

        if DpF.min() <= 0.2:
            st.warning("Feasibility warning: D′ nearly hits zero in the model. "
                       "To make this time realistic, you usually need a higher CV, larger D′, better clearance, or softer early pacing.")
        else:
            st.info("Feasibility looks acceptable under current parameters (D′ stays > ~0).")

        # store for exports
        st.session_state["last_table"] = df_fit
        st.session_state["last_figs"] = [("Fitted velocity", fit_vel_fig), ("Fitted lactate + D′", fit_ld_fig)]

# =========================
# TAB 3: Real splits analyzer
# =========================
with tab3:
    st.subheader("Enter real 4×50 splits → Reconstruct + analyze")
    st.caption("If you also enter your underwater distances and push speeds, the app subtracts underwater time and estimates surface speeds per lap.")

    c = st.columns(4)
    s1 = c[0].text_input("Split 1 (0–50)", "26.50")
    s2 = c[1].text_input("Split 2 (50–100)", "28.00")
    s3 = c[2].text_input("Split 3 (100–150)", "28.50")
    s4 = c[3].text_input("Split 4 (150–200)", "29.20")

    st.markdown("**Underwaters + push speeds** (use your measured breakout distances if possible)")
    u = st.columns(4)
    uwd = [
        u[0].number_input("UW start (m)", 0.0, 15.0, 15.0, 0.1),
        u[1].number_input("UW turn 1 (m)", 0.0, 15.0, 10.5, 0.1),
        u[2].number_input("UW turn 2 (m)", 0.0, 15.0, 10.0, 0.1),
        u[3].number_input("UW turn 3 (m)", 0.0, 15.0, 10.0, 0.1),
    ]
    v = st.columns(4)
    uwv0 = [
        v[0].number_input("v0 start (m/s)", 1.5, 4.0, 3.2, 0.05),
        v[1].number_input("v0 turn 1 (m/s)", 1.5, 4.0, 2.6, 0.05),
        v[2].number_input("v0 turn 2 (m/s)", 1.5, 4.0, 2.55, 0.05),
        v[3].number_input("v0 turn 3 (m/s)", 1.5, 4.0, 2.55, 0.05),
    ]

    if st.button("Analyze race from splits", type="primary"):
        splits_in = [parse_time_to_seconds(x) for x in [s1, s2, s3, s4]]
        pace_type = pacing_type_from_splits(splits_in)

        s_est = build_strategy_from_splits(splits_in, uwd, uwv0, p)
        tsA, xsA, vsA, DpA, LA, split_timesA = simulate_200_free(s_est, p, dt=dt)

        totalA = split_timesA[-1]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total from splits (s)", f"{sum(splits_in):.2f}")
        m2.metric("Model reconstructed total (s)", f"{totalA:.2f}")
        m3.metric("Min D′", f"{DpA.min():.2f}")
        m4.metric("Peak L", f"{LA.max():.2f}")

        st.write(f"**Pacing classification (from your splits):** {pace_type}")

        # Lap detail table
        rows = []
        for i in range(4):
            t_uw = underwater_time_for_lap(uwd[i], uwv0[i], p, dt=0.02)
            dist_surface = 50.0 - uwd[i]
            t_surface = max(0.01, splits_in[i] - t_uw)
            v_surface = dist_surface / t_surface

            # diagnostics: when would UW be time-positive?
            # rough: compare average UW speed to surface speed
            v_uw_avg = uwd[i] / max(0.01, t_uw)
            uw_good = "YES" if v_uw_avg > v_surface else "MAYBE SHORTEN"

            rows.append({
                "Lap": f"50 #{i+1}",
                "Split (s)": f"{splits_in[i]:.2f}",
                "UW dist (m)": f"{uwd[i]:.1f}",
                "UW time est (s)": f"{t_uw:.2f}",
                "UW avg v (m/s)": f"{v_uw_avg:.2f}",
                "Surface v est (m/s)": f"{v_surface:.3f}",
                "UW time-positive?": uw_good
            })

        df_race = pd.DataFrame(rows)
        st.dataframe(df_race, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = plt.figure(figsize=(7, 4))
            plt.plot(xsA, vsA)
            plt.xlabel("Distance (m)")
            plt.ylabel("Velocity (m/s)")
            plt.title("Reconstructed: velocity vs distance")
            st.pyplot(fig)
            race_vel_fig = fig_to_png_bytes(fig)

        with c2:
            fig = plt.figure(figsize=(7, 4))
            plt.plot(tsA, LA, label="L(t)")
            plt.plot(tsA, DpA, label="D′(t)")
            plt.xlabel("Time (s)")
            plt.title("Reconstructed: lactate + D′")
            plt.legend()
            st.pyplot(fig)
            race_ld_fig = fig_to_png_bytes(fig)

        # Coaching-style summary
        st.markdown("### Model-based takeaways")
        if DpA.min() <= 0.2:
            st.warning("Your split pattern likely spends anaerobic reserve too early (in this model). Consider smoothing the first 100 or improving CV/D′.")
        else:
            st.info("D′ stays above ~0 → pacing may be physiologically sustainable under these parameters.")

        # Store for exports
        export_table = pd.DataFrame([{
            "Type": "Race analysis",
            "50": f"{splits_in[0]:.2f}",
            "100": f"{splits_in[1]:.2f}",
            "150": f"{splits_in[2]:.2f}",
            "200": f"{splits_in[3]:.2f}",
            "Total": f"{sum(splits_in):.2f}",
        }])
        st.session_state["last_table"] = export_table
        st.session_state["last_figs"] = [("Race velocity", race_vel_fig), ("Race lactate + D′", race_ld_fig)]

# =========================
# TAB 4: Exports
# =========================
with tab4:
    st.subheader("Exports (workflow diagram, PDF, CSV)")

    wf = make_workflow_diagram_png()
    st.image(wf, caption="Workflow diagram", use_container_width=True)

    st.download_button(
        "Download workflow diagram (PNG)",
        data=wf,
        file_name="workflow_diagram_clean.png",
        mime="image/png",
    )

    last_table = st.session_state.get("last_table")
    last_figs = st.session_state.get("last_figs")

    st.markdown("**Export last analysis** (from Target Time Fit or Real Splits tab).")
    if last_table is None:
        st.info("Run a Target Time Fit or Real Splits Analysis first — then come back here to export.")
    else:
        st.dataframe(last_table, use_container_width=True)

        csv_bytes = last_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results table (CSV)",
            data=csv_bytes,
            file_name="analysis_table.csv",
            mime="text/csv",
        )

        figs_for_pdf = [("Workflow diagram", wf)]
        if last_figs:
            figs_for_pdf += last_figs

        pdf_bytes = build_pdf_summary(
            title="200m Freestyle (LCM) — Mini Report Export",
            splits_df=last_table,
            figs=figs_for_pdf
        )

        st.download_button(
            "Download mini PDF report",
            data=pdf_bytes,
            file_name="200free_mini_report.pdf",
            mime="application/pdf",
        )
