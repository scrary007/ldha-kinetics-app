import hashlib
import re
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit
from scipy.stats import linregress


# =============================================================================
# PAGE SETUP
# =============================================================================
st.set_page_config(page_title="LDHA Kinetics Analyzer", layout="wide")

st.title("LDHA Kinetics Analyzer")
st.caption(
    "Determine initial velocities from A340 time courses, review each fit, "
    "and fit the accepted rates to a kinetic model."
)


# =============================================================================
# COURSE / ASSAY DEFAULTS
# These defaults match the 2026 LDHA Michaelis–Menten protocol.
# =============================================================================
DEFAULT_EPSILON_NADH = 6220.0          # M^-1 cm^-1 at 340 nm
DEFAULT_PATH_LENGTH_CM = 1.0
DEFAULT_NADH_UM = 75.0
DEFAULT_ENZYME_NM = 0.667              # 20 nM stock * 100 uL / 3000 uL
DEFAULT_MIXING_RECOVERY_S = 10.0       # empirical course mixing/dead-time experiment
DEFAULT_PREFERRED_PROGRESS = 5.0       # % of limiting reactant
DEFAULT_MAX_PROGRESS = 10.0            # warning / search ceiling


# =============================================================================
# SIDEBAR: ASSAY SETTINGS
# =============================================================================
st.sidebar.header("Assay settings")

epsilon_nadh = st.sidebar.number_input(
    "NADH extinction coefficient at 340 nm (M⁻¹ cm⁻¹)",
    min_value=1.0,
    value=DEFAULT_EPSILON_NADH,
    step=10.0,
    help="Used to convert ΔA340/s to concentration change per second.",
)

path_length_cm = st.sidebar.number_input(
    "Cuvette path length (cm)",
    min_value=0.01,
    value=DEFAULT_PATH_LENGTH_CM,
    step=0.1,
)

nadh_uM = st.sidebar.number_input(
    "Initial NADH concentration (µM)",
    min_value=0.0,
    value=DEFAULT_NADH_UM,
    step=1.0,
)

enzyme_nM = st.sidebar.number_input(
    "Final enzyme concentration (nM)",
    min_value=0.0,
    value=DEFAULT_ENZYME_NM,
    step=0.001,
    format="%.3f",
    help="Used only to calculate kcat. The course protocol gives ~0.667 nM final enzyme.",
)

with st.sidebar.expander("Advanced initial-rate settings"):
    mixing_recovery_s = st.number_input(
        "Mixing recovery after detected disturbance (s)",
        min_value=0.0,
        value=DEFAULT_MIXING_RECOVERY_S,
        step=0.5,
        help=(
            "The course water-addition experiment showed that the main manual mixing "
            "disturbance generally settled within about 6–10 s."
        ),
    )
    preferred_progress_pct = st.number_input(
        "Preferred maximum reaction progress (%)",
        min_value=0.1,
        max_value=50.0,
        value=DEFAULT_PREFERRED_PROGRESS,
        step=0.5,
    )
    max_progress_pct = st.number_input(
        "Absolute reaction-progress warning limit (%)",
        min_value=preferred_progress_pct,
        max_value=100.0,
        value=DEFAULT_MAX_PROGRESS,
        step=0.5,
    )
    half_slope_diff_limit = st.number_input(
        "Half-to-half slope difference limit (%)",
        min_value=1.0,
        value=15.0,
        step=1.0,
        help="Used for the automatic suggestion. It is not a pass/fail rule for students.",
    )
    thirds_spread_limit = st.number_input(
        "Three-segment slope spread limit (%)",
        min_value=1.0,
        value=30.0,
        step=1.0,
    )


with st.expander("Assay details used in this course"):
    st.markdown(
        """
- **Wavelength:** 340 nm  
- **Temperature:** 25 °C  
- **Total reaction volume:** 3.00 mL  
- **NADH:** 75 µM  
- **Buffer:** 50 mM Tris-Cl, pH 7.5, with 50 mM NaCl  
- **Enzyme:** 100 µL of a 20 nM working stock in 3.00 mL total volume → approximately 0.667 nM final enzyme  
- **Sampling interval:** approximately 0.1 s  
- **Mixing:** enzyme is added after data collection begins; the beginning of the trace therefore includes addition and manual mixing.

The app treats pyruvate values parsed from filenames as the **final concentration in the reaction, in mM**.  
For the kinetic-model fit, pyruvate remains numerically in mM, so fitted **Km and Ki are also in mM**.
"""
    )


# =============================================================================
# DATA HANDLING
# =============================================================================
def load_and_clean_csv(uploaded_file):
    """Read Cary-style CSV output and return Time (s), Absorbance dataframe."""
    content = uploaded_file.getvalue().decode("utf-8", errors="replace")
    lines = content.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if "Time (sec)" in line:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("Could not find a 'Time (sec)' header.")

    data_lines = [lines[start_idx]]
    for line in lines[start_idx + 1 :]:
        parts = line.split(",")
        if len(parts) < 2:
            break
        try:
            float(parts[0])
            float(parts[1])
            data_lines.append(line)
        except ValueError:
            break

    df = pd.read_csv(StringIO("\n".join(data_lines))).iloc[:, :2]
    df.columns = ["Time", "Abs"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Too few numeric data points were found.")
    return df


def parse_pyruvate_mM(filename):
    """Parse a value such as 3,5mM or 3.5mM from the filename."""
    match = re.search(r"(\d+[,.]?\d*)\s*mM", filename, re.IGNORECASE)
    if not match:
        return 0.0
    return float(match.group(1).replace(",", "."))


def parse_enzyme_label(filename):
    """Use the second underscore-separated filename field as the default enzyme label."""
    stem = re.sub(r"\.csv$", "", filename, flags=re.IGNORECASE)
    parts = stem.split("_")
    if len(parts) >= 2 and parts[1].strip():
        return parts[1].strip().upper()
    return "UNKNOWN"


def safe_key(filename):
    return hashlib.md5(filename.encode("utf-8")).hexdigest()[:10]


# =============================================================================
# INITIAL-RATE ANALYSIS
# =============================================================================
def detect_mixing_event(df, recovery_s):
    """
    Detect the strongest early disturbance and place the suggested fit start
    recovery_s seconds later.

    This is intentionally empirical: it uses the shape of the measured trace,
    rather than a theoretical starting absorbance.
    """
    time = df["Time"].to_numpy()
    absorbance = df["Abs"].to_numpy()

    if len(df) < 5:
        return float(time[0]), float(time[0])

    dt = float(np.median(np.diff(time)))
    smooth_points = max(3, int(round(0.5 / max(dt, 1e-6))))
    smoothed = pd.Series(absorbance).rolling(
        smooth_points, center=True, min_periods=1
    ).median().to_numpy()
    derivative = np.gradient(smoothed, time)

    search_end = min(float(time[0] + 12.0), float(time[-1]))
    early_mask = time <= search_end
    early_indices = np.flatnonzero(early_mask)

    if len(early_indices) == 0:
        onset_idx = 0
    else:
        onset_idx = early_indices[np.argmax(np.abs(derivative[early_indices]))]

    onset_time = float(time[onset_idx])
    suggested_start = min(onset_time + recovery_s, float(time[-1]))
    return onset_time, suggested_start


def regression_diagnostics(df, start_s, end_s, pyruvate_mM, eps, path_cm, nadh_initial_uM):
    data = df[(df["Time"] >= start_s) & (df["Time"] <= end_s)].copy()
    if len(data) < 4:
        return None

    reg = linregress(data["Time"], data["Abs"])
    slope = float(reg.slope)
    intercept = float(reg.intercept)
    r2 = float(reg.rvalue ** 2)

    duration = float(end_s - start_s)
    rate_uM_s = abs(slope) / (eps * path_cm) * 1e6

    # Use the fitted line rather than two noisy endpoints to estimate reaction progress.
    consumed_uM = abs(slope) * duration / (eps * path_cm) * 1e6
    limiting_uM = min(float(nadh_initial_uM), float(pyruvate_mM) * 1000.0)
    progress_pct = (100.0 * consumed_uM / limiting_uM) if limiting_uM > 0 else np.nan

    midpoint = (start_s + end_s) / 2.0
    first = data[data["Time"] <= midpoint]
    second = data[data["Time"] > midpoint]

    if len(first) >= 2 and len(second) >= 2:
        slope1 = float(linregress(first["Time"], first["Abs"]).slope)
        slope2 = float(linregress(second["Time"], second["Abs"]).slope)
        denom = 0.5 * (abs(slope1) + abs(slope2))
        half_diff_pct = 100.0 * abs(slope1 - slope2) / denom if denom > 1e-12 else np.inf
    else:
        slope1 = slope2 = np.nan
        half_diff_pct = np.nan

    # Split into thirds as a second curvature / slope-drift diagnostic.
    edges = np.linspace(start_s, end_s, 4)
    third_slopes = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        segment = data[(data["Time"] >= lo) & (data["Time"] <= hi)]
        if len(segment) >= 2:
            third_slopes.append(float(linregress(segment["Time"], segment["Abs"]).slope))

    if len(third_slopes) == 3:
        mean_abs = float(np.mean(np.abs(third_slopes)))
        thirds_spread_pct = (
            100.0 * (max(third_slopes) - min(third_slopes)) / mean_abs
            if mean_abs > 1e-12
            else np.inf
        )
        first_last_denom = 0.5 * (abs(third_slopes[0]) + abs(third_slopes[-1]))
        first_last_diff_pct = (
            100.0 * abs(third_slopes[0] - third_slopes[-1]) / first_last_denom
            if first_last_denom > 1e-12
            else np.inf
        )
    else:
        thirds_spread_pct = np.nan
        first_last_diff_pct = np.nan

    fitted = slope * data["Time"].to_numpy() + intercept
    residuals = data["Abs"].to_numpy() - fitted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "data": data,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "v0_uM_s": rate_uM_s,
        "progress_pct": progress_pct,
        "half_diff_pct": half_diff_pct,
        "thirds_spread_pct": thirds_spread_pct,
        "first_last_diff_pct": first_last_diff_pct,
        "rmse": rmse,
        "residuals": residuals,
    }


def suggest_initial_rate_interval(
    df,
    pyruvate_mM,
    eps,
    path_cm,
    nadh_initial_uM,
    recovery_s,
    preferred_progress,
    max_progress,
    half_limit,
    thirds_limit,
):
    """
    Suggest the earliest defensible linear interval after mixing.

    Strategy:
      1. Detect the strongest early mixing disturbance.
      2. Begin searching recovery_s later.
      3. Search the first ~3 s of possible start positions.
      4. Prefer the longest interval that remains <= preferred reaction progress
         and has reasonably stable slopes across halves and thirds.
      5. If no interval passes all preferred criteria, choose the least-problematic
         candidate under the absolute reaction-progress ceiling and flag it.
    """
    onset_s, base_start_s = detect_mixing_event(df, recovery_s)
    time = df["Time"].to_numpy()
    dt = float(np.median(np.diff(time)))
    scan_step = max(0.5, dt * 5.0)
    min_duration = 20.0
    max_duration = 70.0

    latest_start = min(base_start_s + 3.0, float(time[-1]) - min_duration)
    if latest_start < base_start_s:
        latest_start = base_start_s

    start_candidates = np.arange(base_start_s, latest_start + scan_step / 2.0, scan_step)

    for start_s in start_candidates:
        last_end = min(start_s + max_duration, float(time[-1]))
        end_candidates = np.arange(start_s + min_duration, last_end + scan_step / 2.0, scan_step)
        accepted = []

        for end_s in end_candidates:
            diag = regression_diagnostics(
                df, start_s, end_s, pyruvate_mM, eps, path_cm, nadh_initial_uM
            )
            if diag is None:
                continue

            if (
                diag["slope"] < 0
                and diag["progress_pct"] <= preferred_progress
                and diag["half_diff_pct"] <= half_limit
                and diag["thirds_spread_pct"] <= thirds_limit
                and diag["first_last_diff_pct"] <= half_limit
            ):
                accepted.append((end_s, diag))

        if accepted:
            end_s, diag = accepted[-1]  # longest acceptable interval at earliest acceptable start
            return {
                "mixing_onset_s": onset_s,
                "mixing_end_s": base_start_s,
                "start_s": float(start_s),
                "end_s": float(end_s),
                "diagnostics": diag,
                "preferred": True,
            }

    # Fallback: score all candidates below the absolute reaction-progress limit.
    candidates = []
    for start_s in start_candidates:
        last_end = min(start_s + max_duration, float(time[-1]))
        end_candidates = np.arange(start_s + min_duration, last_end + scan_step / 2.0, scan_step)

        for end_s in end_candidates:
            diag = regression_diagnostics(
                df, start_s, end_s, pyruvate_mM, eps, path_cm, nadh_initial_uM
            )
            if diag is None or diag["slope"] >= 0 or diag["progress_pct"] > max_progress:
                continue

            score = (
                max(0.0, diag["half_diff_pct"] - half_limit)
                + max(0.0, diag["thirds_spread_pct"] - thirds_limit)
                + max(0.0, diag["first_last_diff_pct"] - half_limit)
                + 2.0 * max(0.0, diag["progress_pct"] - preferred_progress)
                + 2.0 * (start_s - base_start_s)
            )
            candidates.append((score, start_s, end_s, diag))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, start_s, end_s, diag = candidates[0]
        return {
            "mixing_onset_s": onset_s,
            "mixing_end_s": base_start_s,
            "start_s": float(start_s),
            "end_s": float(end_s),
            "diagnostics": diag,
            "preferred": False,
        }

    # Very short or unusual trace: provide a visible, editable fallback.
    start_s = min(base_start_s, float(time[-1]))
    end_s = min(start_s + 20.0, float(time[-1]))
    if end_s <= start_s:
        start_s = float(time[0])
        end_s = float(time[-1])
    diag = regression_diagnostics(df, start_s, end_s, pyruvate_mM, eps, path_cm, nadh_initial_uM)
    return {
        "mixing_onset_s": onset_s,
        "mixing_end_s": base_start_s,
        "start_s": start_s,
        "end_s": end_s,
        "diagnostics": diag,
        "preferred": False,
    }


# =============================================================================
# KINETIC MODELS
# =============================================================================
def michaelis_menten(S, Vmax, Km):
    return Vmax * S / (Km + S)


def substrate_inhibition(S, Vmax, Km, Ki):
    """Briggs–Haldane substrate-inhibition form."""
    return Vmax * S / (Km + S + (S ** 2) / Ki)


def aicc(y_obs, y_pred, n_parameters):
    n = len(y_obs)
    residuals = np.asarray(y_obs) - np.asarray(y_pred)
    rss = float(np.sum(residuals ** 2))
    rss = max(rss, np.finfo(float).tiny)
    aic = n * np.log(rss / n) + 2 * n_parameters
    if n <= n_parameters + 1:
        return np.nan
    return aic + (2 * n_parameters * (n_parameters + 1)) / (n - n_parameters - 1)


def fit_mm_model(S, v):
    vmax0 = max(float(np.max(v)), 1e-9)
    positive_s = S[S > 0]
    km0 = float(np.median(positive_s)) if len(positive_s) else 1.0
    popt, pcov = curve_fit(
        michaelis_menten,
        S,
        v,
        p0=[vmax0, max(km0, 1e-6)],
        bounds=(0, np.inf),
        maxfev=50000,
    )
    pred = michaelis_menten(S, *popt)
    return popt, pcov, pred, aicc(v, pred, 2)


def fit_inhibition_model(S, v):
    vmax0 = max(float(np.max(v)) * 1.2, 1e-9)
    positive_s = S[S > 0]
    km0 = float(np.median(positive_s)) if len(positive_s) else 1.0
    ki0 = max(float(np.max(S)), 1.0)
    popt, pcov = curve_fit(
        substrate_inhibition,
        S,
        v,
        p0=[vmax0, max(km0, 1e-6), ki0],
        bounds=(0, np.inf),
        maxfev=100000,
    )
    pred = substrate_inhibition(S, *popt)
    return popt, pcov, pred, aicc(v, pred, 3)


# =============================================================================
# WIDGET STATE SYNCHRONIZATION
# =============================================================================
def sync_numbers_from_slider(interval_key, start_key, end_key):
    start_s, end_s = st.session_state[interval_key]
    st.session_state[start_key] = float(start_s)
    st.session_state[end_key] = float(end_s)


def sync_slider_from_numbers(interval_key, start_key, end_key, min_time, max_time):
    start_s = float(st.session_state[start_key])
    end_s = float(st.session_state[end_key])
    start_s = min(max(start_s, min_time), max_time)
    end_s = min(max(end_s, min_time), max_time)
    if end_s > start_s:
        st.session_state[interval_key] = (start_s, end_s)


def reset_interval_to_suggestion(interval_key, start_key, end_key, suggested_start, suggested_end):
    """Reset slider and exact-value fields before Streamlit redraws the widgets."""
    start_s = float(suggested_start)
    end_s = float(suggested_end)
    st.session_state[interval_key] = (start_s, end_s)
    st.session_state[start_key] = start_s
    st.session_state[end_key] = end_s


# =============================================================================
# MAIN WORKFLOW
# =============================================================================
st.header("1. Upload kinetic traces")
st.markdown(
    "Upload the Cary CSV files. The app reads pyruvate concentration from filenames such as "
    "`3,5mM` or `3.5mM`; you can correct the concentration for any run below."
)

uploaded_files = st.file_uploader(
    "Kinetic CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more kinetic CSV files to begin.")
    st.stop()


loaded_runs = []
for uploaded_file in uploaded_files:
    try:
        df = load_and_clean_csv(uploaded_file)
        loaded_runs.append(
            {
                "filename": uploaded_file.name,
                "df": df,
                "parsed_pyruvate": parse_pyruvate_mM(uploaded_file.name),
                "parsed_enzyme": parse_enzyme_label(uploaded_file.name),
            }
        )
    except Exception as exc:
        st.error(f"{uploaded_file.name}: {exc}")

if not loaded_runs:
    st.stop()



st.header("2. Review the initial-rate fit for each run")
st.markdown(
    "The app suggests the **earliest stable linear region after the mixing disturbance**. "
    "Drag the range slider immediately under each plot to change the interval. "
    "Exact Start and End values can also be entered directly below the plot."
)
st.caption(
    "Each run is handled independently. Changing the interval for one run redraws only that run. "
    "When you are finished making changes, use **Update summary and kinetic fit** below."
)


@st.fragment
def render_run_review(
    run,
    epsilon_nadh,
    path_length_cm,
    nadh_uM,
    mixing_recovery_s,
    preferred_progress_pct,
    max_progress_pct,
    half_slope_diff_limit,
    thirds_spread_limit,
):
    filename = run["filename"]
    df = run["df"]
    key = safe_key(filename)

    pyr_key = f"pyr_{key}"
    enzyme_key = f"enzyme_{key}"
    include_key = f"include_{key}"
    interval_key = f"interval_{key}"
    start_key = f"start_exact_{key}"
    end_key = f"end_exact_{key}"
    result_key = f"result_{key}"

    if pyr_key not in st.session_state:
        st.session_state[pyr_key] = float(run["parsed_pyruvate"])
    if enzyme_key not in st.session_state:
        st.session_state[enzyme_key] = run["parsed_enzyme"]
    if include_key not in st.session_state:
        st.session_state[include_key] = True

    pyruvate_mM = float(st.session_state[pyr_key])

    suggestion = suggest_initial_rate_interval(
        df=df,
        pyruvate_mM=pyruvate_mM,
        eps=epsilon_nadh,
        path_cm=path_length_cm,
        nadh_initial_uM=nadh_uM,
        recovery_s=mixing_recovery_s,
        preferred_progress=preferred_progress_pct,
        max_progress=max_progress_pct,
        half_limit=half_slope_diff_limit,
        thirds_limit=thirds_spread_limit,
    )

    min_time = float(df["Time"].min())
    max_time = float(df["Time"].max())
    dt = float(np.median(np.diff(df["Time"])))
    slider_step = max(round(dt, 4), 0.001)

    if interval_key not in st.session_state:
        st.session_state[interval_key] = (suggestion["start_s"], suggestion["end_s"])
    if start_key not in st.session_state:
        st.session_state[start_key] = float(st.session_state[interval_key][0])
    if end_key not in st.session_state:
        st.session_state[end_key] = float(st.session_state[interval_key][1])

    with st.expander(f"{filename}", expanded=True):
        meta1, meta2, meta3 = st.columns([1, 1, 1])
        meta1.checkbox("Include in kinetic-model fit", key=include_key)
        meta2.number_input(
            "Final pyruvate [S] (mM)",
            min_value=0.0,
            step=0.001,
            format="%.4f",
            key=pyr_key,
        )
        meta3.text_input("Enzyme / variant label", key=enzyme_key)

        pyruvate_mM = float(st.session_state[pyr_key])
        selected_start, selected_end = st.session_state[interval_key]
        diagnostics = regression_diagnostics(
            df,
            selected_start,
            selected_end,
            pyruvate_mM,
            epsilon_nadh,
            path_length_cm,
            nadh_uM,
        )

        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.plot(df["Time"], df["Abs"], linewidth=1.0, label="Recorded A340")
        ax.axvspan(
            suggestion["mixing_onset_s"],
            suggestion["mixing_end_s"],
            alpha=0.12,
            label="Enzyme addition / mixing",
        )
        ax.axvspan(selected_start, selected_end, alpha=0.10, label="Selected V₀ interval")

        if diagnostics is not None:
            fit_data = diagnostics["data"]
            fit_y = diagnostics["slope"] * fit_data["Time"] + diagnostics["intercept"]
            ax.scatter(fit_data["Time"], fit_data["Abs"], s=9, label="Points used for V₀")
            ax.plot(fit_data["Time"], fit_y, linewidth=2.0, label="Linear fit")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Absorbance at 340 nm")

        # Make the time axis easier to read when choosing exact Start/End times.
        # Major tick labels stay reasonably spaced; faint vertical guides appear every 5 s.
        trace_duration = max_time - min_time
        major_x_spacing = 10 if trace_duration <= 120 else 20
        ax.xaxis.set_major_locator(mticker.MultipleLocator(major_x_spacing))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))

        # Major gridlines in both directions, plus extra faint vertical minor gridlines.
        ax.grid(True, which="major", alpha=0.18, linewidth=0.7)
        ax.grid(True, which="minor", axis="x", alpha=0.10, linewidth=0.5)

        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.slider(
            "Drag to choose the V₀ interval (s)",
            min_value=min_time,
            max_value=max_time,
            step=slider_step,
            key=interval_key,
            on_change=sync_numbers_from_slider,
            args=(interval_key, start_key, end_key),
        )

        exact1, exact2, reset_col = st.columns([1, 1, 1])
        exact1.number_input(
            "Start (s)",
            min_value=min_time,
            max_value=max_time,
            step=slider_step,
            key=start_key,
            on_change=sync_slider_from_numbers,
            args=(interval_key, start_key, end_key, min_time, max_time),
        )
        exact2.number_input(
            "End (s)",
            min_value=min_time,
            max_value=max_time,
            step=slider_step,
            key=end_key,
            on_change=sync_slider_from_numbers,
            args=(interval_key, start_key, end_key, min_time, max_time),
        )
        reset_col.button(
            "Reset to suggested interval",
            key=f"reset_{key}",
            on_click=reset_interval_to_suggestion,
            args=(
                interval_key,
                start_key,
                end_key,
                suggestion["start_s"],
                suggestion["end_s"],
            ),
        )

        selected_start, selected_end = st.session_state[interval_key]
        diagnostics = regression_diagnostics(
            df,
            selected_start,
            selected_end,
            pyruvate_mM,
            epsilon_nadh,
            path_length_cm,
            nadh_uM,
        )

        if diagnostics is None:
            st.session_state.pop(result_key, None)
            st.error("The selected interval contains too few points.")
            return

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("V₀", f"{diagnostics['v0_uM_s']:.4f} µM/s")
        m2.metric("Slope", f"{diagnostics['slope']:.6f} Abs/s")
        m3.metric("R²", f"{diagnostics['r2']:.4f}")
        m4.metric("Reaction progress", f"{diagnostics['progress_pct']:.1f}%")
        m5.metric("Half-slope difference", f"{diagnostics['half_diff_pct']:.1f}%")

        if selected_start < suggestion["mixing_end_s"]:
            st.warning(
                "This interval begins inside the empirically estimated mixing/dead-time region. "
                "The fitted slope may include mixing rather than enzyme kinetics."
            )

        if diagnostics["progress_pct"] > max_progress_pct:
            st.warning(
                f"This interval corresponds to about {diagnostics['progress_pct']:.1f}% reaction progress, "
                f"above the {max_progress_pct:.1f}% maximum used for an initial-rate fit."
            )
        elif diagnostics["progress_pct"] > preferred_progress_pct:
            st.info(
                f"This fit extends beyond the preferred {preferred_progress_pct:.1f}% reaction-progress region "
                "but remains below the warning limit."
            )

        if diagnostics["half_diff_pct"] > half_slope_diff_limit:
            st.warning(
                "Possible curvature: the fitted slopes in the first and second halves of this interval "
                f"differ by {diagnostics['half_diff_pct']:.1f}%. Consider shortening or shifting the interval."
            )
        elif diagnostics["thirds_spread_pct"] > thirds_spread_limit:
            st.warning(
                "The local slopes vary across this interval, suggesting noise or curvature. "
                "Inspect the trace before accepting V₀."
            )
        else:
            st.success(
                "The slope is reasonably stable across the selected interval. "
                "R² is shown as a diagnostic, but the app does not use an R² cutoff by itself."
            )

        with st.expander("What is the app calculating here?"):
            st.markdown("The selected points are fit by ordinary linear regression:")
            st.latex(r"A_{340}=m t+b")
            st.markdown("The magnitude of the slope is converted to an initial velocity using Beer–Lambert law:")
            st.latex(r"V_0=\frac{|\Delta A_{340}/\Delta t|}{\epsilon_{NADH}\,l}\times 10^6")
            st.markdown(
                "Because time is recorded in seconds and ε is in M⁻¹ cm⁻¹, this gives **µM/s**. "
                "The beginning of the trace is excluded because enzyme addition and manual mixing occur after data collection begins."
            )

        # Save this run's current accepted result in Session State.
        st.session_state[result_key] = {
            "Include": bool(st.session_state[include_key]),
            "File": filename,
            "Enzyme": st.session_state[enzyme_key],
            "Pyruvate_mM": pyruvate_mM,
            "Start_s": selected_start,
            "End_s": selected_end,
            "Slope_Abs_s": diagnostics["slope"],
            "V0_uM_s": diagnostics["v0_uM_s"],
            "R2": diagnostics["r2"],
            "Progress_pct": diagnostics["progress_pct"],
            "Half_slope_diff_pct": diagnostics["half_diff_pct"],
            "Thirds_slope_spread_pct": diagnostics["thirds_spread_pct"],
        }


result_keys = []
for run in loaded_runs:
    key = safe_key(run["filename"])
    result_keys.append(f"result_{key}")
    render_run_review(
        run,
        epsilon_nadh,
        path_length_cm,
        nadh_uM,
        mixing_recovery_s,
        preferred_progress_pct,
        max_progress_pct,
        half_slope_diff_limit,
        thirds_spread_limit,
    )


# =============================================================================
# SUMMARY / DOWNLOAD + KINETIC MODEL
# This is a separate fragment, so refreshing the model does not redraw all runs.
# =============================================================================
@st.fragment
def render_summary_and_models(result_keys, enzyme_nM):
    st.header("3. Review accepted initial velocities")
    st.button(
        "Update summary and kinetic fit",
        type="primary",
        use_container_width=True,
        help="Use this after changing one or more run intervals. It updates this section without redrawing every time-course plot.",
    )
    st.caption(
        "The individual run controls above update independently for speed. "
        "Click this button after making changes to bring the summary and kinetic model up to date."
    )

    current_results = [
        st.session_state[k]
        for k in result_keys
        if k in st.session_state
    ]
    results_df = pd.DataFrame(current_results)

    if len(results_df) == 0:
        st.warning("No valid initial-rate fits are available yet.")
        st.stop()

    summary_display = results_df[
        [
            "Include",
            "File",
            "Enzyme",
            "Pyruvate_mM",
            "V0_uM_s",
            "R2",
            "Progress_pct",
            "Start_s",
            "End_s",
        ]
    ].copy()
    summary_display.columns = [
        "Include",
        "File",
        "Enzyme",
        "Pyruvate (mM)",
        "V₀ (µM/s)",
        "R²",
        "Reaction progress (%)",
        "Start (s)",
        "End (s)",
    ]
    st.dataframe(summary_display, hide_index=True, use_container_width=True)

    st.download_button(
        "Download V₀ results as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="ldha_v0_results.csv",
        mime="text/csv",
    )


    # =============================================================================
    # KINETIC-MODEL FITTING
    # =============================================================================
    st.header("4. Fit the relationship between V₀ and pyruvate concentration")
    st.markdown(
        "Use **Michaelis–Menten** when velocity approaches a plateau without a high-substrate decline. "
        "Use the **substrate-inhibition (Briggs–Haldane)** model when velocity decreases at high pyruvate. "
        "The app does not add an artificial (0,0) data point."
    )

    included_df = results_df[results_df["Include"]].copy()

    if len(included_df) < 3:
        st.warning("Include at least three runs to fit a kinetic model.")
        st.stop()

    S = included_df["Pyruvate_mM"].to_numpy(dtype=float)
    v = included_df["V0_uM_s"].to_numpy(dtype=float)

    model_choice = st.radio(
        "Kinetic model",
        [
            "Michaelis–Menten",
            "Substrate inhibition (Briggs–Haldane)",
            "Compare both models",
        ],
        horizontal=True,
    )

    st.markdown("**Michaelis–Menten**")
    st.latex(r"v=\frac{V_{max}[S]}{K_m+[S]}")
    st.markdown("**Substrate inhibition (Briggs–Haldane)**")
    st.latex(r"v=\frac{V_{max}[S]}{K_m+[S]+[S]^2/K_i}")

    fit_outputs = {}
    fit_errors = []

    if model_choice in ("Michaelis–Menten", "Compare both models"):
        try:
            fit_outputs["MM"] = fit_mm_model(S, v)
        except Exception as exc:
            fit_errors.append(f"Michaelis–Menten fit failed: {exc}")

    if model_choice in ("Substrate inhibition (Briggs–Haldane)", "Compare both models"):
        if len(included_df) < 4:
            fit_errors.append("The substrate-inhibition model needs at least four included observations for a useful three-parameter fit.")
        else:
            try:
                fit_outputs["INH"] = fit_inhibition_model(S, v)
            except Exception as exc:
                fit_errors.append(f"Substrate-inhibition fit failed: {exc}")

    for err in fit_errors:
        st.warning(err)

    if not fit_outputs:
        st.stop()

    fig_mm, ax_mm = plt.subplots(figsize=(8, 5.5))
    ax_mm.scatter(S, v, s=45, label="Experimental V₀")

    s_max = max(float(np.max(S)) * 1.08, 1.0)
    s_plot = np.linspace(0.0, s_max, 400)

    parameter_rows = []

    if "MM" in fit_outputs:
        popt, pcov, pred, mm_aicc = fit_outputs["MM"]
        vmax, km = popt
        errors = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan, np.nan]
        ax_mm.plot(s_plot, michaelis_menten(s_plot, *popt), linewidth=2.0, label="Michaelis–Menten")
        parameter_rows.append(
            {
                "Model": "Michaelis–Menten",
                "Vmax (µM/s)": vmax,
                "Vmax SE": errors[0],
                "Km (mM)": km,
                "Km SE": errors[1],
                "Ki (mM)": np.nan,
                "Ki SE": np.nan,
                "AICc": mm_aicc,
            }
        )

    if "INH" in fit_outputs:
        popt, pcov, pred, inh_aicc = fit_outputs["INH"]
        vmax, km, ki = popt
        errors = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan, np.nan, np.nan]
        ax_mm.plot(s_plot, substrate_inhibition(s_plot, *popt), linewidth=2.0, label="Substrate inhibition")
        parameter_rows.append(
            {
                "Model": "Substrate inhibition",
                "Vmax (µM/s)": vmax,
                "Vmax SE": errors[0],
                "Km (mM)": km,
                "Km SE": errors[1],
                "Ki (mM)": ki,
                "Ki SE": errors[2],
                "AICc": inh_aicc,
            }
        )

    ax_mm.set_xlabel("Pyruvate concentration [S] (mM)")
    ax_mm.set_ylabel("Initial velocity V₀ (µM/s)")
    ax_mm.grid(True, alpha=0.18, linewidth=0.7)
    ax_mm.legend()
    fig_mm.tight_layout()
    st.pyplot(fig_mm, clear_figure=True)

    params_df = pd.DataFrame(parameter_rows)
    st.dataframe(params_df, hide_index=True, use_container_width=True)

    # kcat is calculated from Vmax using final enzyme concentration.
    if enzyme_nM > 0:
        enzyme_uM = enzyme_nM / 1000.0
        st.subheader("Turnover number")
        for row in parameter_rows:
            kcat = row["Vmax (µM/s)"] / enzyme_uM
            st.write(f"**{row['Model']}:** kcat = {kcat:.1f} s⁻¹")

    if model_choice == "Compare both models" and len(parameter_rows) == 2:
        st.subheader("Model comparison")
        aicc_values = params_df.set_index("Model")["AICc"]
        if aicc_values.notna().all():
            best = aicc_values.idxmin()
            delta = aicc_values - aicc_values.min()
            compare_df = pd.DataFrame({"AICc": aicc_values, "ΔAICc": delta})
            st.dataframe(compare_df, use_container_width=True)
            if len(included_df) < 7:
                st.warning(
                    "This is a very small dataset for comparing a two-parameter model with a three-parameter model. "
                    "AICc can be strongly affected by the extra-parameter penalty here; inspect the curve, residuals, "
                    "and parameter uncertainty rather than using AICc alone to choose the biological model."
                )
            st.markdown(
                f"The smaller AICc is for **{best}**. AICc rewards fit quality but penalizes the "
                "extra parameter in the inhibition model; use it as evidence about model support, not as an automatic biological decision."
            )
        else:
            st.info("There are too few observations for a finite AICc comparison of both models.")

    with st.expander("Model diagnostics and interpretation"):
        for model_name, output in fit_outputs.items():
            popt, pcov, pred, model_aicc = output
            residuals = v - pred
            label = "Michaelis–Menten" if model_name == "MM" else "Substrate inhibition"
            st.markdown(f"**{label}**")
            residual_df = pd.DataFrame(
                {
                    "Pyruvate (mM)": S,
                    "Observed V₀ (µM/s)": v,
                    "Predicted V₀ (µM/s)": pred,
                    "Residual (µM/s)": residuals,
                }
            )
            st.dataframe(residual_df, hide_index=True, use_container_width=True)

            fig_res, ax_res = plt.subplots(figsize=(7, 3.2))
            ax_res.axhline(0.0, linewidth=1.0)
            ax_res.scatter(S, residuals, s=35)
            ax_res.set_xlabel("Pyruvate concentration (mM)")
            ax_res.set_ylabel("Residual (µM/s)")
            ax_res.grid(True, alpha=0.18, linewidth=0.7)
            fig_res.tight_layout()
            st.pyplot(fig_res, clear_figure=True)

            if model_name == "INH":
                ki = popt[2]
                ki_se = np.sqrt(pcov[2, 2]) if np.all(np.isfinite(pcov)) and pcov.shape == (3, 3) else np.nan
                weak_ki = (ki > 100 * max(np.max(S), 1e-9)) or (np.isfinite(ki_se) and ki_se > ki)
                if weak_ki:
                    st.info(
                        "The inhibition constant Ki is weakly constrained by these data. This can happen when the "
                        "high-substrate decline is absent or too small to estimate substrate inhibition reliably. "
                        "Do not interpret a very large or highly uncertain Ki as a precise biological value."
                    )

    st.divider()
    st.caption(
        "Original LDHA Streamlit analyzer developed by Roee Sela '27. "
        "This revision makes initial-rate selection and kinetic-model fitting explicit and reviewable."
    )


render_summary_and_models(result_keys, enzyme_nM)
