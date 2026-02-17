import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="SOOT Explorer", layout="wide")
st.title("SOOT Explorer — Time Series + Vertical Profile")

# -----------------------------
# Data registry (add your CSVs here)
# -----------------------------
DATA_DIR = Path(__file__).parent / "data"

DATASETS = {
    "STAQS (trimmed)": {
        "path": DATA_DIR / "soot_trimmed.csv",
        "alt_col": "Altitude_m_MSL",
        "o3_col": "Ozone_ppbv",
        # pick the time column your file actually has:
        "time_col": "time_mid",   # change if needed (e.g., "Time" or "UTC_time")
    },
    "Other dataset": {
        "path": DATA_DIR / "soot_other.csv",
        "alt_col": "Altitude_m_MSL",
        "o3_col": "Ozone_ppbv",
        "time_col": "time_mid",
    },
}

FILL_VALUES = [-9999, -9999.0, -8888, -8888.0, -7777, -7777.0]

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Dataset")
dataset_name = st.sidebar.selectbox("Choose a CSV", list(DATASETS.keys()))
cfg = DATASETS[dataset_name]

st.sidebar.header("Cleaning / Smoothing")
bin_m = st.sidebar.slider("Altitude bin size (m)", 10, 500, 50, 10)
window = st.sidebar.slider("Rolling window (bins)", 3, 51, 11, 2)

show_raw_profile = st.sidebar.checkbox("Profile: show raw points", True)
show_ci_profile = st.sidebar.checkbox("Profile: show ~95% CI", True)

show_raw_ts = st.sidebar.checkbox("Time series: show raw points", True)
ts_smooth_window = st.sidebar.slider("Time series smoothing window (points)", 3, 401, 51, 2)

# -----------------------------
# Load + cache
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, na_values=FILL_VALUES)

def require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable columns: {list(df.columns)}")

def clean_common(df: pd.DataFrame, alt_col: str, o3_col: str, time_col: str) -> pd.DataFrame:
    x = df.copy()

    # Numeric conversions
    x[alt_col] = pd.to_numeric(x[alt_col], errors="coerce")
    x[o3_col] = pd.to_numeric(x[o3_col], errors="coerce")

    # Time handling: try numeric first; if not, parse as datetime
    # (works for most cases; if yours is special, adjust)
    if pd.api.types.is_numeric_dtype(x[time_col]):
        x[time_col] = pd.to_numeric(x[time_col], errors="coerce")
    else:
        x[time_col] = pd.to_datetime(x[time_col], errors="coerce")

    # Physical cleaning
    x.loc[x[o3_col] <= 0, o3_col] = np.nan

    return x

# -----------------------------
# Prepare data
# -----------------------------
try:
    df = load_csv(str(cfg["path"]))
    require_cols(df, [cfg["alt_col"], cfg["o3_col"], cfg["time_col"]])
    x = clean_common(df, cfg["alt_col"], cfg["o3_col"], cfg["time_col"])
except Exception as e:
    st.error(f"Dataset '{dataset_name}' failed to load/validate.\n\n{e}")
    st.stop()

ALT_COL = cfg["alt_col"]
O3_COL = cfg["o3_col"]
TIME_COL = cfg["time_col"]

# For profile: require alt & ozone
d_profile = x.dropna(subset=[ALT_COL, O3_COL]).copy()
if d_profile.empty:
    st.warning("No valid rows for profile after cleaning.")
# For time series: require time & ozone
d_ts = x.dropna(subset=[TIME_COL, O3_COL]).copy()
if d_ts.empty:
    st.warning("No valid rows for time series after cleaning.")

# -----------------------------
# Layout: two plots
# -----------------------------
col1, col2 = st.columns([1, 1], gap="large")

# ---------- TIME SERIES ----------
with col1:
    st.subheader("Time Series — Ozone vs Time")

    if not d_ts.empty:
        d_ts = d_ts.sort_values(TIME_COL)

        # Build a simple smooth line (rolling mean). For datetime, rolling on index requires set_index.
        if np.issubdtype(d_ts[TIME_COL].dtype, np.number):
            y_smooth = d_ts[O3_COL].rolling(ts_smooth_window, center=True, min_periods=3).mean()
            x_time = d_ts[TIME_COL]
        else:
            # datetime
            tmp = d_ts.set_index(TIME_COL)
            y_smooth = tmp[O3_COL].rolling(ts_smooth_window, center=True, min_periods=3).mean()
            x_time = y_smooth.index

        fig_ts = matplotlib.figure.Figure(figsize=(6.5, 4.6), dpi=150)
        ax_ts = fig_ts.add_subplot(111)
        ax_ts.spines["top"].set_visible(False)
        ax_ts.spines["right"].set_visible(False)

        if show_raw_ts:
            ax_ts.scatter(d_ts[TIME_COL], d_ts[O3_COL], s=6, alpha=0.12, linewidths=0, label="Raw", color="#9aa0a6")

        ax_ts.plot(x_time, y_smooth, linewidth=2.0, label=f"Rolling mean ({ts_smooth_window})", color="#1f77b4")

        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Ozone (ppbv)")
        ax_ts.grid(True, alpha=0.22)
        ax_ts.legend(frameon=False, loc="best")
        fig_ts.tight_layout()

        st.pyplot(fig_ts)
    else:
        st.info("No time-series data to plot for this dataset.")

# ---------- VERTICAL PROFILE ----------
with col2:
    st.subheader("Vertical Profile — Ozone vs Altitude")

    if not d_profile.empty:
        d_profile["alt_bin"] = (d_profile[ALT_COL] / bin_m).round() * bin_m

        profile = (
            d_profile.groupby("alt_bin")[O3_COL]
            .agg(mean="mean", n="size", std="std")
            .reset_index()
            .sort_values("alt_bin")
        )
        profile["sem"] = profile["std"] / np.sqrt(profile["n"])
        profile.loc[profile["n"] < 5, "sem"] = np.nan
        profile["mean_smooth"] = profile["mean"].rolling(window=window, center=True, min_periods=3).mean()

        fig_pr = matplotlib.figure.Figure(figsize=(6.5, 4.6), dpi=150)
        ax_pr = fig_pr.add_subplot(111)
        ax_pr.spines["top"].set_visible(False)
        ax_pr.spines["right"].set_visible(False)

        if show_raw_profile:
            ax_pr.scatter(d_profile[O3_COL], d_profile[ALT_COL], s=6, alpha=0.10, linewidths=0,
                          label="Raw", color="#9aa0a6")

        ax_pr.plot(profile["mean"], profile["alt_bin"], linewidth=1.3, alpha=0.75,
                   label=f"Binned mean ({bin_m} m)", color="#ff7f0e")
        ax_pr.plot(profile["mean_smooth"], profile["alt_bin"], linewidth=2.4,
                   label=f"Smoothed (rolling {window} bins)", color="#d62728")

        if show_ci_profile:
            mask = profile["sem"].notna() & profile["mean_smooth"].notna()
            if mask.any():
                lower = profile.loc[mask, "mean_smooth"] - 1.96 * profile.loc[mask, "sem"]
                upper = profile.loc[mask, "mean_smooth"] + 1.96 * profile.loc[mask, "sem"]
                ax_pr.fill_betweenx(profile.loc[mask, "alt_bin"], lower, upper, alpha=0.18,
                                    label="~95% CI (SEM)", color="#1f77b4")

        ax_pr.set_xlabel("Ozone (ppbv)")
        ax_pr.set_ylabel("Altitude (m MSL)")
        ax_pr.grid(True, alpha=0.22)
        ax_pr.legend(frameon=False, loc="best")
        fig_pr.tight_layout()

        st.pyplot(fig_pr)
    else:
        st.info("No profile data to plot for this dataset.")

# Helpful debug: show column names if you need to map time columns per dataset
with st.expander("Dataset info / columns"):
    st.write("File:", str(cfg["path"]))
    st.write("Rows:", len(df))
    st.write("Columns:", list(df.columns))
