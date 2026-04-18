
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import os

WT_COLOR = "#648FFF"
MUT_COLOR = "#DC267F"
EH_MM_COLOR = "#785EF0"

WT_MARKER = "o"
MUT_MARKER = "s"
WT_LINE = "-"
MUT_LINE = "--"

MARKER_SIZE = 6
LINE_WIDTH = 2

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=15, fontweight="bold", color="black")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold", color="black")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold", color="black")

def style_legend(ax):
    leg = ax.legend(fontsize=10, frameon=True)
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontweight("bold")

def michaelis_menten(S, Vmax, Km):
    return (Vmax * S) / (Km + S)

def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def load_plate_reader_file(file_path, ignored_substrates=None):
    if ignored_substrates is None:
        ignored_substrates = []
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path, header=None)
    else:
        df = pd.read_excel(file_path, header=None)
    if df.shape[0] < 3 or df.shape[1] < 2:
        raise ValueError("File format not recognized. Expected at least 3 rows and 2 columns.")

    time = pd.to_numeric(df.iloc[2:, 0], errors="coerce").values
    substrate_headers = df.iloc[0, 1:].tolist()
    grouped = {}
    preview_rows = []
    trace_counter = {}

    for col_offset, raw_substrate in enumerate(substrate_headers, start=1):
        substrate = safe_float(raw_substrate)
        if substrate is None:
            preview_rows.append({"Column_Index": col_offset, "Header_Value": raw_substrate, "Parsed_Substrate_uM": np.nan, "Replicate_Name": "", "Used": "No", "Reason": "Non-numeric or blank header", "Valid_Points": 0})
            continue
        if substrate in ignored_substrates:
            preview_rows.append({"Column_Index": col_offset, "Header_Value": raw_substrate, "Parsed_Substrate_uM": substrate, "Replicate_Name": "", "Used": "No", "Reason": "Ignored by user", "Valid_Points": 0})
            continue
        absorbance = pd.to_numeric(df.iloc[2:, col_offset], errors="coerce").values
        mask = (~np.isnan(time)) & (~np.isnan(absorbance))
        t = time[mask]
        a = absorbance[mask]
        if len(t) < 3:
            preview_rows.append({"Column_Index": col_offset, "Header_Value": raw_substrate, "Parsed_Substrate_uM": substrate, "Replicate_Name": "", "Used": "No", "Reason": "Fewer than 3 valid points", "Valid_Points": int(len(t))})
            continue

        trace_counter[substrate] = trace_counter.get(substrate, 0) + 1
        replicate_name = f"rep{trace_counter[substrate]}"
        grouped.setdefault(substrate, []).append({"substrate_uM": substrate, "replicate_name": replicate_name, "column_index": col_offset, "time_s": t, "absorbance": a})
        preview_rows.append({"Column_Index": col_offset, "Header_Value": raw_substrate, "Parsed_Substrate_uM": substrate, "Replicate_Name": replicate_name, "Used": "Yes", "Reason": "Used", "Valid_Points": int(len(t))})

    if not grouped:
        raise ValueError("No usable traces found. Check that row 1 contains numeric substrate values in uM.")
    return grouped, pd.DataFrame(preview_rows)

def fit_initial_velocity(time_s, signal, method="first_n", first_n=5, auto_min_points=4, auto_max_fraction=0.5):
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time_s) < 3:
        raise ValueError("Need at least 3 points to fit a line.")
    if method == "first_n":
        n = min(max(3, int(first_n)), len(time_s))
        x = time_s[:n]
        y = signal[:n]
        fit = linregress(x, y)
        slope = fit.slope
        intercept = fit.intercept
        r2 = fit.rvalue ** 2
        fit_idx = np.arange(n)
    else:
        min_pts = max(3, int(auto_min_points))
        max_pts = max(min_pts, int(np.floor(len(time_s) * auto_max_fraction)))
        max_pts = min(max_pts, len(time_s))
        best = None
        best_score = None
        for n in range(min_pts, max_pts + 1):
            x = time_s[:n]
            y = signal[:n]
            fit = linregress(x, y)
            yhat = fit.intercept + fit.slope * x
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
            score = r2 + 0.0005 * n
            if best_score is None or score > best_score:
                best_score = score
                best = (fit.slope, fit.intercept, r2, np.arange(n))
        slope, intercept, r2, fit_idx = best
    velocity_abs_per_s = -slope
    return {"slope": slope, "intercept": intercept, "r2": r2, "fit_idx": fit_idx, "velocity_abs_per_s": velocity_abs_per_s}

def convert_absorbance_rate_to_concentration_rate(rate_abs_per_s, epsilon_M_cm=6220.0, pathlength_cm=1.0):
    return (rate_abs_per_s / (epsilon_M_cm * pathlength_cm)) * 1e6

def analyze_grouped_traces(grouped_traces, method, first_n, auto_min_points, auto_max_fraction, convert_to_conc=False, epsilon=6220.0, pathlength_cm=1.0):
    analyzed_traces = []
    for substrate, trace_list in grouped_traces.items():
        for trace in trace_list:
            fitres = fit_initial_velocity(trace["time_s"], trace["absorbance"], method=method, first_n=first_n, auto_min_points=auto_min_points, auto_max_fraction=auto_max_fraction)
            if convert_to_conc:
                velocity = convert_absorbance_rate_to_concentration_rate(fitres["velocity_abs_per_s"], epsilon, pathlength_cm)
                units = "uM/s"
            else:
                velocity = fitres["velocity_abs_per_s"]
                units = "Abs/s"
            analyzed_traces.append({**trace, **fitres, "velocity_raw": velocity, "velocity_units": units})
    trace_df = pd.DataFrame({
        "Substrate_uM": [x["substrate_uM"] for x in analyzed_traces],
        "Replicate": [x["replicate_name"] for x in analyzed_traces],
        "Velocity_Raw": [x["velocity_raw"] for x in analyzed_traces],
        "Velocity_units": [x["velocity_units"] for x in analyzed_traces],
        "Slope_Abs_per_s": [x["slope"] for x in analyzed_traces],
        "R2_initial_fit": [x["r2"] for x in analyzed_traces],
        "Column_Index": [x["column_index"] for x in analyzed_traces],
    }).sort_values(["Substrate_uM", "Replicate"])
    return analyzed_traces, trace_df

def apply_blank_correction(trace_df, use_blank_correction=False, blank_substrate_value=0.0):
    df = trace_df.copy()
    df["Velocity_Corrected"] = df["Velocity_Raw"]
    blank_summary = None
    blank_mean = np.nan
    blank_sd = np.nan
    blank_n = 0
    if use_blank_correction:
        blank_df = df[df["Substrate_uM"] == blank_substrate_value].copy()
        if blank_df.empty:
            raise ValueError(f"Blank correction was selected, but no {blank_substrate_value:g} uM columns were found.")
        blank_n = len(blank_df)
        blank_mean = blank_df["Velocity_Raw"].mean()
        blank_sd = blank_df["Velocity_Raw"].std(ddof=1) if blank_n > 1 else np.nan
        df.loc[df["Substrate_uM"] != blank_substrate_value, "Velocity_Corrected"] = df.loc[df["Substrate_uM"] != blank_substrate_value, "Velocity_Raw"] - blank_mean
        blank_summary = pd.DataFrame({"Blank_Substrate_uM": [blank_substrate_value], "Blank_N": [blank_n], "Blank_Mean_Velocity": [blank_mean], "Blank_SD_Velocity": [blank_sd], "Velocity_units": [df["Velocity_units"].iloc[0] if len(df) else ""]})
    return df, blank_summary, blank_mean, blank_sd, blank_n

def summarize_by_substrate(trace_df, use_corrected=False, exclude_zero_for_fits=True):
    velocity_col = "Velocity_Corrected" if use_corrected else "Velocity_Raw"
    grouped_summary = trace_df.groupby("Substrate_uM", as_index=False).agg(
        N=(velocity_col, "count"),
        Mean_Velocity=(velocity_col, "mean"),
        SD_Velocity=(velocity_col, lambda x: np.std(x, ddof=1) if len(x) > 1 else np.nan),
        Mean_R2=("R2_initial_fit", "mean"),
    ).sort_values("Substrate_uM")
    grouped_summary["SEM_Velocity"] = grouped_summary["SD_Velocity"] / np.sqrt(grouped_summary["N"])
    grouped_summary["Velocity_Source"] = velocity_col
    grouped_summary["Velocity_units"] = trace_df["Velocity_units"].iloc[0] if len(trace_df) else ""
    grouped_summary["Used_For_MM_LWB"] = "Yes"
    if exclude_zero_for_fits:
        grouped_summary.loc[grouped_summary["Substrate_uM"] == 0, "Used_For_MM_LWB"] = "No"
    return grouped_summary

def fit_mm_and_lwb(summary_df):
    fit_df = summary_df[summary_df["Used_For_MM_LWB"] == "Yes"].copy()
    S = fit_df["Substrate_uM"].to_numpy(dtype=float)
    v = fit_df["Mean_Velocity"].to_numpy(dtype=float)
    mask = (~np.isnan(S)) & (~np.isnan(v)) & (S > 0) & (v > 0)
    S = S[mask]
    v = v[mask]
    if len(S) < 3:
        raise ValueError("Need at least 3 positive substrate concentrations with positive mean velocities for kinetic fitting.")
    vmax_guess = np.max(v)
    km_guess = np.median(S)
    popt, pcov = curve_fit(michaelis_menten, S, v, p0=[vmax_guess, km_guess], maxfev=10000)
    Vmax_mm, Km_mm = popt
    try:
        mm_sd = np.sqrt(np.diag(pcov))
        Vmax_mm_sd = float(mm_sd[0])
        Km_mm_sd = float(mm_sd[1])
    except Exception:
        Vmax_mm_sd = np.nan
        Km_mm_sd = np.nan
    vhat_mm = michaelis_menten(S, Vmax_mm, Km_mm)
    mm_residuals = v - vhat_mm
    mm_sse = float(np.sum(mm_residuals ** 2))
    mm_mse = float(np.mean(mm_residuals ** 2))
    ss_tot_mm = float(np.sum((v - np.mean(v)) ** 2))
    mm_r2 = np.nan if ss_tot_mm == 0 else float(1 - (mm_sse / ss_tot_mm))

    x_eh = v / S
    y_eh = v
    eh = linregress(x_eh, y_eh)
    eh_slope = eh.slope
    eh_intercept = eh.intercept
    Vmax_eh = eh_intercept
    Km_eh = -eh_slope
    eh_r2 = eh.rvalue ** 2
    Vmax_eh_sd = getattr(eh, "intercept_stderr", np.nan)
    Km_eh_sd = getattr(eh, "stderr", np.nan)

    invS = 1.0 / S
    invv = 1.0 / v
    lb = linregress(invS, invv)
    slope_lb = lb.slope
    intercept_lb = lb.intercept
    if intercept_lb == 0:
        raise ValueError("Lineweaver-Burk intercept was zero; cannot compute Vmax.")
    Vmax_lb = 1.0 / intercept_lb
    Km_lb = slope_lb * Vmax_lb

    slope_lb_sd = getattr(lb, "stderr", np.nan)
    intercept_lb_sd = getattr(lb, "intercept_stderr", np.nan)
    if np.isnan(intercept_lb_sd):
        Vmax_lb_sd = np.nan
        Km_lb_sd = np.nan
    else:
        Vmax_lb_sd = abs((-1.0 / (intercept_lb ** 2)) * intercept_lb_sd)
        if np.isnan(slope_lb_sd):
            Km_lb_sd = np.nan
        else:
            # error propagation assuming independence
            Km_lb_sd = float(np.sqrt((Vmax_lb * slope_lb_sd) ** 2 + (slope_lb * Vmax_lb_sd) ** 2))

    return {
        "S": S, "v": v,
        "Vmax_mm": Vmax_mm, "Km_mm": Km_mm, "Vmax_mm_sd": Vmax_mm_sd, "Km_mm_sd": Km_mm_sd,
        "MM_SSE": mm_sse, "MM_MSE": mm_mse, "MM_R2": mm_r2,
        "Vmax_eh": Vmax_eh, "Km_eh": Km_eh, "Vmax_eh_sd": Vmax_eh_sd, "Km_eh_sd": Km_eh_sd,
        "eh_slope": eh_slope, "eh_intercept": eh_intercept, "eh_r2": eh_r2,
        "Vmax_lb": Vmax_lb, "Km_lb": Km_lb, "Vmax_lb_sd": Vmax_lb_sd, "Km_lb_sd": Km_lb_sd,
        "lb_slope": slope_lb, "lb_intercept": intercept_lb, "lb_r2": lb.rvalue ** 2
    }

def calculate_final_dimer_concentration(stock_dimer_uM, enzyme_volume_uL, total_volume_uL):
    if stock_dimer_uM <= 0 or enzyme_volume_uL <= 0 or total_volume_uL <= 0:
        raise ValueError("Stock dimer concentration, enzyme volume, and total volume must all be > 0.")
    if enzyme_volume_uL > total_volume_uL:
        raise ValueError("Enzyme volume cannot exceed total assay volume.")
    return stock_dimer_uM * (enzyme_volume_uL / total_volume_uL)

def convert_dimer_to_active_site_concentration(final_dimer_uM, active_sites_per_dimer=2):
    if final_dimer_uM <= 0 or active_sites_per_dimer <= 0:
        raise ValueError("Final dimer concentration and active sites per dimer must be > 0.")
    return final_dimer_uM * active_sites_per_dimer

def compute_kcat(vmax_value, active_site_uM):
    if active_site_uM <= 0:
        raise ValueError("Active-site concentration must be > 0.")
    return vmax_value / active_site_uM

def generate_qc_flags(trace_df, summary_df, use_blank_correction=False, blank_mean=np.nan, blank_sd=np.nan, blank_n=0, blank_large_fraction_threshold=0.2, low_r2_threshold=0.98):
    flags = []
    if len(trace_df):
        low_r2_df = trace_df[trace_df["R2_initial_fit"] < low_r2_threshold]
        for _, row in low_r2_df.iterrows():
            flags.append({"Level": "Warning", "Category": "Initial velocity fit", "Detail": f"Substrate {row['Substrate_uM']:.6g} uM, {row['Replicate']}: R^2 = {row['R2_initial_fit']:.4f} < {low_r2_threshold:.2f}"})
    if use_blank_correction and not np.isnan(blank_mean):
        nonzero = summary_df[summary_df["Substrate_uM"] > 0].copy()
        if len(nonzero):
            max_mean = nonzero["Mean_Velocity"].max()
            if max_mean > 0:
                frac = abs(blank_mean) / max_mean
                if frac >= blank_large_fraction_threshold:
                    flags.append({"Level": "Warning", "Category": "Blank drift", "Detail": f"Mean blank velocity is {frac:.1%} of the largest nonzero mean velocity, which is relatively large."})
        neg_df = trace_df[(trace_df["Substrate_uM"] > 0) & (trace_df["Velocity_Corrected"] < 0)]
        for _, row in neg_df.iterrows():
            flags.append({"Level": "Warning", "Category": "Negative corrected velocity", "Detail": f"Substrate {row['Substrate_uM']:.6g} uM, {row['Replicate']}: corrected velocity = {row['Velocity_Corrected']:.6g}"})
        if blank_n == 1:
            flags.append({"Level": "Note", "Category": "Blank replicates", "Detail": "Only one blank replicate was found. Multiple blank replicates would provide a better estimate of drift."})
    if not flags:
        flags.append({"Level": "OK", "Category": "QC", "Detail": "No automatic QC issues were flagged with the current thresholds."})
    return pd.DataFrame(flags)

def run_dataset_analysis(file_path, label, settings):
    grouped_traces, preview_df = load_plate_reader_file(file_path, ignored_substrates=settings["ignored"])
    analyzed_traces, trace_df = analyze_grouped_traces(grouped_traces, method=settings["method"], first_n=settings["first_n"], auto_min_points=settings["auto_min"], auto_max_fraction=settings["auto_frac"], convert_to_conc=settings["convert"], epsilon=settings["epsilon"], pathlength_cm=settings["pathlength"])
    trace_df, blank_summary_df, blank_mean, blank_sd, blank_n = apply_blank_correction(trace_df, use_blank_correction=settings["do_blank"], blank_substrate_value=settings["blank_sub"])
    summary_df = summarize_by_substrate(trace_df, use_corrected=settings["do_blank"], exclude_zero_for_fits=True)
    fitres = fit_mm_and_lwb(summary_df)
    qc_df = generate_qc_flags(trace_df, summary_df, use_blank_correction=settings["do_blank"], blank_mean=blank_mean, blank_sd=blank_sd, blank_n=blank_n)

    final_active_site_uM = None
    kcat_mm = None
    kcat_over_km = None
    final_dimer_uM = None
    if settings["compute_kcat"]:
        if summary_df["Velocity_units"].iloc[0] != "uM/s":
            raise ValueError("kcat requires concentration-based velocities. Enable Beer-Lambert conversion first.")
        final_dimer_uM = calculate_final_dimer_concentration(settings["stock_dimer"], settings["enzyme_vol"], settings["total_vol"])
        final_active_site_uM = convert_dimer_to_active_site_concentration(final_dimer_uM, settings["sites_per_dimer"])
        kcat_mm = compute_kcat(fitres["Vmax_mm"], final_active_site_uM)
        kcat_over_km = kcat_mm / fitres["Km_mm"]

    return {"label": label, "preview_df": preview_df, "analyzed_traces": analyzed_traces, "trace_df": trace_df, "summary_df": summary_df, "fitres": fitres, "qc_df": qc_df, "blank_summary_df": blank_summary_df, "blank_mean": blank_mean, "blank_sd": blank_sd, "blank_n": blank_n, "final_dimer_uM": final_dimer_uM, "final_active_site_uM": final_active_site_uM, "kcat_mm": kcat_mm, "kcat_over_km": kcat_over_km}



def make_single_panel(result, title_prefix="", show_plots=True, save_base=None):
    units = result["summary_df"]["Velocity_units"].iloc[0] if len(result["summary_df"]) else "Velocity"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Figure 1. Progress curves
    substrates = sorted({x["substrate_uM"] for x in result["analyzed_traces"]})
    for substrate in substrates:
        traces = [x for x in result["analyzed_traces"] if x["substrate_uM"] == substrate]
        if not traces:
            continue
        velocities = [tr.get("velocity_raw", np.nan) for tr in traces]
        mean_v = np.nanmean(velocities)
        tr = traces[0]
        ax1.plot(
            tr["time_s"], tr["absorbance"],
            marker=WT_MARKER, linestyle=WT_LINE, color=WT_COLOR,
            markersize=MARKER_SIZE, linewidth=1.5,
            label=f"{substrate:.0f} uM ({mean_v:.2f})"
        )
    style_axes(ax1, "Figure 1. Progress Curves", "Time (s)", "Absorbance")
    style_legend(ax1)

    # Figure 2. Michaelis-Menten
    S = result["fitres"]["S"]
    v = result["fitres"]["v"]
    Sfit = np.linspace(0, max(S) * 1.1, 300)
    vfit = michaelis_menten(Sfit, result["fitres"]["Vmax_mm"], result["fitres"]["Km_mm"])
    summary = result["summary_df"][result["summary_df"]["Substrate_uM"].isin(S)].copy()

    fit_label = (
        f"MM: Km={result['fitres']['Km_mm']:.2f}±{result['fitres']['Km_mm_sd']:.2f} uM\n"
        f"Vmax={result['fitres']['Vmax_mm']:.2f}±{result['fitres']['Vmax_mm_sd']:.2f} {units}"
    )

    ax2.errorbar(
        summary["Substrate_uM"], summary["Mean_Velocity"], yerr=summary["SD_Velocity"],
        fmt=WT_MARKER, color=WT_COLOR, markersize=MARKER_SIZE, capsize=4, label="_nolegend_"
    )
    ax2.plot(Sfit, vfit, color=WT_COLOR, linestyle=WT_LINE, linewidth=LINE_WIDTH, label=fit_label)
    style_axes(ax2, "Figure 2. Michaelis-Menten", "Substrate (uM)", f"Initial velocity ({units})")
    style_legend(ax2)
    leg = ax2.get_legend()
    if leg:
        leg.set_bbox_to_anchor((1, 0))
        leg._loc = 4

    # Figure 3. Eadie-Hofstee
    x_eh = v / S
    y_eh = v
    xline_eh = np.linspace(0, max(x_eh) * 1.1, 300)
    yline_mm_eh = result["fitres"]["Vmax_mm"] - result["fitres"]["Km_mm"] * xline_eh
    yline_fit_eh = result["fitres"]["eh_intercept"] + result["fitres"]["eh_slope"] * xline_eh

    label_mm = (
        f"From MM: Km={result['fitres']['Km_mm']:.2f}±{result['fitres']['Km_mm_sd']:.2f} uM\n"
        f"Vmax={result['fitres']['Vmax_mm']:.2f}±{result['fitres']['Vmax_mm_sd']:.2f} {units}"
    )
    label_eh = (
        f"EH fit: Km={result['fitres']['Km_eh']:.2f}±{result['fitres']['Km_eh_sd']:.2f} uM\n"
        f"Vmax={result['fitres']['Vmax_eh']:.2f}±{result['fitres']['Vmax_eh_sd']:.2f} {units}"
    )

    ax3.scatter(x_eh, y_eh, color=WT_COLOR, marker=WT_MARKER, s=MARKER_SIZE**2, label="_nolegend_")
    ax3.plot(xline_eh, yline_mm_eh, color=EH_MM_COLOR, linestyle="-", linewidth=LINE_WIDTH, label=label_mm)
    ax3.plot(xline_eh, yline_fit_eh, color=WT_COLOR, linestyle="--", linewidth=LINE_WIDTH, label=label_eh)
    style_axes(ax3, "Figure 3. Eadie-Hofstee", f"v/[S] ({units}/uM)", f"v ({units})")
    style_legend(ax3)

    # Figure 4. Lineweaver-Burk
    invS = 1.0 / S
    invv = 1.0 / v
    xline = np.linspace(0, max(invS) * 1.1, 300)
    yline = result["fitres"]["lb_intercept"] + result["fitres"]["lb_slope"] * xline

    lwb_label = (
        f"LWB: Km={result['fitres']['Km_lb']:.2f}±{result['fitres']['Km_lb_sd']:.2f} uM\n"
        f"Vmax={result['fitres']['Vmax_lb']:.2f}±{result['fitres']['Vmax_lb_sd']:.2f} {units}"
    )

    ax4.scatter(invS, invv, color=WT_COLOR, marker=WT_MARKER, s=MARKER_SIZE**2, label="_nolegend_")
    ax4.plot(xline, yline, color=WT_COLOR, linestyle=WT_LINE, linewidth=LINE_WIDTH, label=lwb_label)
    style_axes(ax4, "Figure 4. Lineweaver-Burk", "1/[S] (1/uM)", f"1/v (1/({units}))")
    style_legend(ax4)

    fig.suptitle(f"{title_prefix} Kinetic Analysis", fontsize=15, fontweight="bold", color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_base:
        fig.savefig(save_base + "_kinetics_panel.png", dpi=300, bbox_inches="tight")
        fig.savefig(save_base + "_kinetics_panel.pdf", bbox_inches="tight")
    if show_plots:
        plt.show()

def make_comparison_panel(wt_res, mut_res, title_prefix="", show_plots=True, save_base=None):
    units = wt_res["summary_df"]["Velocity_units"].iloc[0] if len(wt_res["summary_df"]) else "Velocity"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Figure 1
    for dataset, color, marker, linestyle, label_prefix in [(wt_res, WT_COLOR, WT_MARKER, WT_LINE, "WT"), (mut_res, MUT_COLOR, MUT_MARKER, MUT_LINE, "Mut")]:
        substrates = sorted({x["substrate_uM"] for x in dataset["analyzed_traces"]})
        for substrate in substrates:
            traces = [x for x in dataset["analyzed_traces"] if x["substrate_uM"] == substrate]
            if traces:
                velocities = [tr.get("velocity_raw", np.nan) for tr in traces]
                mean_v = np.nanmean(velocities)
                tr = traces[0]
                ax1.plot(tr["time_s"], tr["absorbance"], marker=marker, linestyle=linestyle, color=color,
                         markersize=MARKER_SIZE, linewidth=1.5,
                         label=f"{label_prefix} {substrate:.0f} uM ({mean_v:.2f})")
    style_axes(ax1, "Figure 1. Progress Curves Overlay", "Time (s)", "Absorbance")
    style_legend(ax1)

    # Figure 2
    for dataset, color, marker, linestyle, name in [(wt_res, WT_COLOR, WT_MARKER, WT_LINE, "WT"), (mut_res, MUT_COLOR, MUT_MARKER, MUT_LINE, "Mutant")]:
        S = dataset["fitres"]["S"]
        v = dataset["fitres"]["v"]
        Sfit = np.linspace(0, max(S) * 1.1, 300)
        vfit = michaelis_menten(Sfit, dataset["fitres"]["Vmax_mm"], dataset["fitres"]["Km_mm"])
        summary = dataset["summary_df"][dataset["summary_df"]["Substrate_uM"].isin(S)].copy()
        fit_label = (
            f"{name} MM: Km={dataset['fitres']['Km_mm']:.2f}±{dataset['fitres']['Km_mm_sd']:.2f} uM\n"
            f"Vmax={dataset['fitres']['Vmax_mm']:.2f}±{dataset['fitres']['Vmax_mm_sd']:.2f} {units}"
        )
        ax2.errorbar(summary["Substrate_uM"], summary["Mean_Velocity"], yerr=summary["SD_Velocity"],
                     fmt=marker, color=color, markersize=MARKER_SIZE, capsize=4, label="_nolegend_")
        ax2.plot(Sfit, vfit, color=color, linestyle=linestyle, linewidth=LINE_WIDTH, label=fit_label)
    style_axes(ax2, "Figure 2. Michaelis-Menten Overlay", "Substrate (uM)", f"Initial velocity ({units})")
    style_legend(ax2)
    leg = ax2.get_legend()
    if leg:
        leg.set_bbox_to_anchor((1, 0))
        leg._loc = 4

    # Figure 3
    S = wt_res["fitres"]["S"]; v = wt_res["fitres"]["v"]; x_eh = v / S; y_eh = v; xline = np.linspace(0, max(x_eh) * 1.1, 300)
    ax3.scatter(x_eh, y_eh, color=WT_COLOR, marker=WT_MARKER, s=MARKER_SIZE**2, label="_nolegend_")
    ax3.plot(xline, wt_res["fitres"]["Vmax_mm"] - wt_res["fitres"]["Km_mm"] * xline, color=EH_MM_COLOR, linestyle="-", linewidth=LINE_WIDTH,
             label=f"WT from MM: Km={wt_res['fitres']['Km_mm']:.2f}±{wt_res['fitres']['Km_mm_sd']:.2f} uM\nVmax={wt_res['fitres']['Vmax_mm']:.2f}±{wt_res['fitres']['Vmax_mm_sd']:.2f} {units}")
    ax3.plot(xline, wt_res["fitres"]["eh_intercept"] + wt_res["fitres"]["eh_slope"] * xline, color=WT_COLOR, linestyle="--", linewidth=LINE_WIDTH,
             label=f"WT EH: Km={wt_res['fitres']['Km_eh']:.2f}±{wt_res['fitres']['Km_eh_sd']:.2f} uM\nVmax={wt_res['fitres']['Vmax_eh']:.2f}±{wt_res['fitres']['Vmax_eh_sd']:.2f} {units}")
    S = mut_res["fitres"]["S"]; v = mut_res["fitres"]["v"]; x_eh = v / S; y_eh = v; xline = np.linspace(0, max(x_eh) * 1.1, 300)
    ax3.scatter(x_eh, y_eh, color=MUT_COLOR, marker=MUT_MARKER, s=MARKER_SIZE**2, label="_nolegend_")
    ax3.plot(xline, mut_res["fitres"]["Vmax_mm"] - mut_res["fitres"]["Km_mm"] * xline, color=EH_MM_COLOR, linestyle=":", linewidth=LINE_WIDTH,
             label=f"Mut from MM: Km={mut_res['fitres']['Km_mm']:.2f}±{mut_res['fitres']['Km_mm_sd']:.2f} uM\nVmax={mut_res['fitres']['Vmax_mm']:.2f}±{mut_res['fitres']['Vmax_mm_sd']:.2f} {units}")
    ax3.plot(xline, mut_res["fitres"]["eh_intercept"] + mut_res["fitres"]["eh_slope"] * xline, color=MUT_COLOR, linestyle="--", linewidth=LINE_WIDTH,
             label=f"Mut EH: Km={mut_res['fitres']['Km_eh']:.2f}±{mut_res['fitres']['Km_eh_sd']:.2f} uM\nVmax={mut_res['fitres']['Vmax_eh']:.2f}±{mut_res['fitres']['Vmax_eh_sd']:.2f} {units}")
    style_axes(ax3, "Figure 3. Eadie-Hofstee Overlay", f"v/[S] ({units}/uM)", f"v ({units})")
    style_legend(ax3)

    # Figure 4
    for dataset, color, marker, linestyle, name in [(wt_res, WT_COLOR, WT_MARKER, WT_LINE, "WT"), (mut_res, MUT_COLOR, MUT_MARKER, MUT_LINE, "Mutant")]:
        S = dataset["fitres"]["S"]; v = dataset["fitres"]["v"]
        invS = 1.0 / S; invv = 1.0 / v
        xline = np.linspace(0, max(invS) * 1.1, 300)
        yline = dataset["fitres"]["lb_intercept"] + dataset["fitres"]["lb_slope"] * xline
        fit_label = (
            f"{name} LWB: Km={dataset['fitres']['Km_lb']:.2f}±{dataset['fitres']['Km_lb_sd']:.2f} uM\n"
            f"Vmax={dataset['fitres']['Vmax_lb']:.2f}±{dataset['fitres']['Vmax_lb_sd']:.2f} {units}"
        )
        ax4.scatter(invS, invv, color=color, marker=marker, s=MARKER_SIZE**2, label="_nolegend_")
        ax4.plot(xline, yline, color=color, linestyle=linestyle, linewidth=LINE_WIDTH, label=fit_label)
    style_axes(ax4, "Figure 4. Lineweaver-Burk Overlay", "1/[S] (1/uM)", f"1/v (1/({units}))")
    style_legend(ax4)

    fig.suptitle(f"{title_prefix} WT vs Mutant Comparison", fontsize=15, fontweight="bold", color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_base:
        fig.savefig(save_base + "_comparison_panel.png", dpi=300, bbox_inches="tight")
        fig.savefig(save_base + "_comparison_panel.pdf", bbox_inches="tight")
    if show_plots:
        plt.show()

def build_comparison_table(wt_res, mut_res):
    def pct_change(mut, wt):
        if wt is None or pd.isna(wt) or wt == 0:
            return np.nan
        return ((mut - wt) / wt) * 100.0
    rows = []
    params = [("Km (MM)", wt_res["fitres"]["Km_mm"], mut_res["fitres"]["Km_mm"], "uM"), ("Vmax (MM)", wt_res["fitres"]["Vmax_mm"], mut_res["fitres"]["Vmax_mm"], wt_res["summary_df"]["Velocity_units"].iloc[0]), ("Km (EH)", wt_res["fitres"]["Km_eh"], mut_res["fitres"]["Km_eh"], "uM"), ("Vmax (EH)", wt_res["fitres"]["Vmax_eh"], mut_res["fitres"]["Vmax_eh"], wt_res["summary_df"]["Velocity_units"].iloc[0]), ("Km (LWB)", wt_res["fitres"]["Km_lb"], mut_res["fitres"]["Km_lb"], "uM"), ("Vmax (LWB)", wt_res["fitres"]["Vmax_lb"], mut_res["fitres"]["Vmax_lb"], wt_res["summary_df"]["Velocity_units"].iloc[0])]
    if wt_res["kcat_mm"] is not None and mut_res["kcat_mm"] is not None:
        params.extend([("kcat", wt_res["kcat_mm"], mut_res["kcat_mm"], "s^-1"), ("kcat/Km", wt_res["kcat_over_km"], mut_res["kcat_over_km"], "s^-1 uM^-1")])
    for name, wt, mut, unit in params:
        wt_sd = np.nan
        mut_sd = np.nan
        if name == "Km (MM)":
            wt_sd = wt_res["fitres"]["Km_mm_sd"]; mut_sd = mut_res["fitres"]["Km_mm_sd"]
        elif name == "Vmax (MM)":
            wt_sd = wt_res["fitres"]["Vmax_mm_sd"]; mut_sd = mut_res["fitres"]["Vmax_mm_sd"]
        elif name == "Km (EH)":
            wt_sd = wt_res["fitres"]["Km_eh_sd"]; mut_sd = mut_res["fitres"]["Km_eh_sd"]
        elif name == "Vmax (EH)":
            wt_sd = wt_res["fitres"]["Vmax_eh_sd"]; mut_sd = mut_res["fitres"]["Vmax_eh_sd"]
        elif name == "Km (LWB)":
            wt_sd = wt_res["fitres"]["Km_lb_sd"]; mut_sd = mut_res["fitres"]["Km_lb_sd"]
        elif name == "Vmax (LWB)":
            wt_sd = wt_res["fitres"]["Vmax_lb_sd"]; mut_sd = mut_res["fitres"]["Vmax_lb_sd"]
        rows.append({"Parameter": name, "WT": wt, "WT_SD": wt_sd, "Mutant": mut, "Mutant_SD": mut_sd, "Mut - WT": mut - wt, "%(Delta) Mut vs WT": pct_change(mut, wt), "Units": unit})
    return pd.DataFrame(rows)


def render_summary_table_png(df, title, outpath):
    fig_h = max(2.5, 0.45 * (len(df) + 2))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", color="black", pad=12)

    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.3g}")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor("#EDEDED")
        else:
            cell.set_text_props(color="black")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def export_single_results(output_path, result):
    fit = result["fitres"]
    fit_table = pd.DataFrame({
        "Parameter": ["Vmax_MM", "Vmax_MM_SD", "Km_MM_uM", "Km_MM_SD", "MM_SSE", "MM_MSE", "MM_R2",
                      "Vmax_EH", "Vmax_EH_SD", "Km_EH_uM", "Km_EH_SD", "EH_R2",
                      "Vmax_LWB", "Vmax_LWB_SD", "Km_LWB_uM", "Km_LWB_SD", "LWB_R2"],
        "Value": [fit["Vmax_mm"], fit["Vmax_mm_sd"], fit["Km_mm"], fit["Km_mm_sd"], fit["MM_SSE"], fit["MM_MSE"], fit["MM_R2"],
                  fit["Vmax_eh"], fit["Vmax_eh_sd"], fit["Km_eh"], fit["Km_eh_sd"], fit["eh_r2"],
                  fit["Vmax_lb"], fit["Vmax_lb_sd"], fit["Km_lb"], fit["Km_lb_sd"], fit["lb_r2"]]
    })
    if result["kcat_mm"] is not None:
        fit_table = pd.concat([fit_table, pd.DataFrame({"Parameter": ["kcat_MM_per_active_site_s^-1", "kcat_over_Km_s^-1_uM^-1"], "Value": [result["kcat_mm"], result["kcat_over_km"]]})], ignore_index=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result["preview_df"].to_excel(writer, sheet_name="Column_Preview", index=False)
        result["trace_df"].to_excel(writer, sheet_name="Per_Trace_Velocities", index=False)
        result["summary_df"].to_excel(writer, sheet_name="Grouped_By_Substrate", index=False)
        fit_table.to_excel(writer, sheet_name="Kinetic_Fits", index=False)
        result["qc_df"].to_excel(writer, sheet_name="QC_Flags", index=False)
        if result["blank_summary_df"] is not None:
            result["blank_summary_df"].to_excel(writer, sheet_name="Blank_Correction", index=False)

def export_comparison_results(output_path, wt_res, mut_res, comp_df):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        wt_res["preview_df"].to_excel(writer, sheet_name="WT_Column_Preview", index=False)
        wt_res["trace_df"].to_excel(writer, sheet_name="WT_Per_Trace", index=False)
        wt_res["summary_df"].to_excel(writer, sheet_name="WT_Grouped", index=False)
        mut_res["preview_df"].to_excel(writer, sheet_name="Mut_Column_Preview", index=False)
        mut_res["trace_df"].to_excel(writer, sheet_name="Mut_Per_Trace", index=False)
        mut_res["summary_df"].to_excel(writer, sheet_name="Mut_Grouped", index=False)
        comp_df.to_excel(writer, sheet_name="WT_vs_Mutant", index=False)

class UnifiedKineticsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enzyme Kinetics Analyzer")
        self.mode = tk.StringVar(value="single")
        self.single_file = tk.StringVar()
        self.wt_file = tk.StringVar()
        self.mut_file = tk.StringVar()
        self.ignore_substrates = tk.StringVar(value="1000")
        self.fit_method = tk.StringVar(value="first_n")
        self.first_n_points = tk.StringVar(value="5")
        self.auto_min_points = tk.StringVar(value="4")
        self.auto_max_fraction = tk.StringVar(value="0.5")
        self.convert_to_conc = tk.BooleanVar(value=True)
        self.epsilon = tk.StringVar(value="6220")
        self.pathlength = tk.StringVar(value="1.0")
        self.use_blank_correction = tk.BooleanVar(value=True)
        self.blank_substrate_uM = tk.StringVar(value="0")
        self.compute_kcat_var = tk.BooleanVar(value=False)
        self.stock_dimer_uM = tk.StringVar(value="")
        self.enzyme_volume_uL = tk.StringVar(value="10")
        self.total_volume_uL = tk.StringVar(value="1000")
        self.active_sites_per_dimer = tk.StringVar(value="2")
        self.status_text = tk.StringVar(value="Choose a file or two files, then click Analyze.")
        self.build_gui()

    def build_gui(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        mode_frame = ttk.Frame(frm)
        mode_frame.grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Radiobutton(mode_frame, text="Single dataset analysis", variable=self.mode, value="single").grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(mode_frame, text="WT vs Mutant comparison", variable=self.mode, value="dual").grid(row=0, column=1)

        ttk.Label(frm, text="Single dataset file:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.single_file, width=70).grid(row=2, column=0, columnspan=3, sticky="ew", pady=2)
        ttk.Button(frm, text="Browse", command=lambda: self.browse_file(self.single_file)).grid(row=2, column=3, padx=4)

        ttk.Label(frm, text="Wild-type file:").grid(row=3, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.wt_file, width=70).grid(row=4, column=0, columnspan=3, sticky="ew", pady=2)
        ttk.Button(frm, text="Browse WT", command=lambda: self.browse_file(self.wt_file)).grid(row=4, column=3, padx=4)

        ttk.Label(frm, text="Mutant file:").grid(row=5, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.mut_file, width=70).grid(row=6, column=0, columnspan=3, sticky="ew", pady=2)
        ttk.Button(frm, text="Browse Mutant", command=lambda: self.browse_file(self.mut_file)).grid(row=6, column=3, padx=4)

        ttk.Label(frm, text="Ignore substrate concentrations (comma-separated, uM):").grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Entry(frm, textvariable=self.ignore_substrates, width=30).grid(row=8, column=0, sticky="w")

        ttk.Label(frm, text="Initial velocity fitting method:").grid(row=9, column=0, sticky="w", pady=(10, 0))
        method_frame = ttk.Frame(frm)
        method_frame.grid(row=10, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(method_frame, text="First N points", variable=self.fit_method, value="first_n").grid(row=0, column=0)
        ttk.Radiobutton(method_frame, text="Auto-search early linear region", variable=self.fit_method, value="auto").grid(row=0, column=1, padx=10)

        ttk.Label(frm, text="First N points:").grid(row=11, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.first_n_points, width=10).grid(row=11, column=1, sticky="w")
        ttk.Label(frm, text="Auto min points:").grid(row=12, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.auto_min_points, width=10).grid(row=12, column=1, sticky="w")
        ttk.Label(frm, text="Auto max fraction of trace:").grid(row=13, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.auto_max_fraction, width=10).grid(row=13, column=1, sticky="w")

        ttk.Checkbutton(frm, text="Convert Abs/s to concentration rate using Beer-Lambert", variable=self.convert_to_conc).grid(row=14, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Label(frm, text="ε (M^-1 cm^-1):").grid(row=15, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.epsilon, width=12).grid(row=15, column=1, sticky="w")
        ttk.Label(frm, text="Path length (cm):").grid(row=16, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.pathlength, width=12).grid(row=16, column=1, sticky="w")

        ttk.Checkbutton(frm, text="Use blank correction", variable=self.use_blank_correction).grid(row=17, column=0, sticky="w", pady=(10, 0))
        ttk.Label(frm, text="Blank substrate value (uM):").grid(row=17, column=1, sticky="e")
        ttk.Entry(frm, textvariable=self.blank_substrate_uM, width=12).grid(row=17, column=2, sticky="w")

        ttk.Checkbutton(frm, text="Compute kcat and kcat/Km", variable=self.compute_kcat_var).grid(row=18, column=0, sticky="w", pady=(10, 0))
        ttk.Label(frm, text="BSA stock dimer (uM):").grid(row=19, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.stock_dimer_uM, width=12).grid(row=19, column=1, sticky="w")
        ttk.Label(frm, text="Enzyme volume (uL):").grid(row=20, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.enzyme_volume_uL, width=12).grid(row=20, column=1, sticky="w")
        ttk.Label(frm, text="Total volume (uL):").grid(row=21, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.total_volume_uL, width=12).grid(row=21, column=1, sticky="w")
        ttk.Label(frm, text="Active sites per dimer:").grid(row=22, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.active_sites_per_dimer, width=12).grid(row=22, column=1, sticky="w")

        ttk.Button(frm, text="Analyze", command=self.analyze).grid(row=23, column=0, pady=(12, 0), sticky="w")
        self.output = tk.Text(frm, width=120, height=18)
        self.output.grid(row=24, column=0, columnspan=4, sticky="nsew", pady=(12, 0))
        frm.rowconfigure(24, weight=1)
        frm.columnconfigure(0, weight=1)
        ttk.Label(frm, textvariable=self.status_text).grid(row=25, column=0, columnspan=4, sticky="w", pady=(8, 0))

    def browse_file(self, var):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            var.set(path)

    def parse_ignored(self):
        text = self.ignore_substrates.get().strip()
        if not text:
            return []
        return [float(x.strip()) for x in text.split(",") if x.strip()]

    def get_settings(self):
        stock = None
        if self.stock_dimer_uM.get().strip():
            stock = float(self.stock_dimer_uM.get())
        return {"ignored": self.parse_ignored(), "method": self.fit_method.get(), "first_n": int(self.first_n_points.get()), "auto_min": int(self.auto_min_points.get()), "auto_frac": float(self.auto_max_fraction.get()), "convert": self.convert_to_conc.get(), "epsilon": float(self.epsilon.get()), "pathlength": float(self.pathlength.get()), "do_blank": self.use_blank_correction.get(), "blank_sub": float(self.blank_substrate_uM.get()), "compute_kcat": self.compute_kcat_var.get(), "stock_dimer": stock, "enzyme_vol": float(self.enzyme_volume_uL.get()), "total_vol": float(self.total_volume_uL.get()), "sites_per_dimer": float(self.active_sites_per_dimer.get())}

    def analyze(self):
        self.output.delete("1.0", tk.END)
        try:
            settings = self.get_settings()
            if settings["compute_kcat"] and settings["stock_dimer"] is None:
                raise ValueError("Enter the BSA-derived stock dimer concentration to compute kcat.")
            if self.mode.get() == "single":
                file_path = self.single_file.get().strip()
                if not file_path:
                    raise ValueError("Please choose a file for single-dataset analysis.")
                result = run_dataset_analysis(file_path, "Dataset", settings)
                base = os.path.splitext(file_path)[0]
                export_single_results(base + "_kinetics_results.xlsx", result)
                summary_table_df = pd.DataFrame({
                    "Method": ["MM", "EH", "LWB"],
                    "Km (uM)": [result["fitres"]["Km_mm"], result["fitres"]["Km_eh"], result["fitres"]["Km_lb"]],
                    "SD(Km)": [result["fitres"]["Km_mm_sd"], result["fitres"]["Km_eh_sd"], result["fitres"]["Km_lb_sd"]],
                    "Vmax": [result["fitres"]["Vmax_mm"], result["fitres"]["Vmax_eh"], result["fitres"]["Vmax_lb"]],
                    "SD(Vmax)": [result["fitres"]["Vmax_mm_sd"], result["fitres"]["Vmax_eh_sd"], result["fitres"]["Vmax_lb_sd"]],
                    "R^2": [result["fitres"]["MM_R2"], result["fitres"]["eh_r2"], result["fitres"]["lb_r2"]],
                })
                if result["kcat_mm"] is not None:
                    summary_table_df["kcat"] = [result["kcat_mm"], np.nan, np.nan]
                    summary_table_df["kcat/Km"] = [result["kcat_over_km"], np.nan, np.nan]
                render_summary_table_png(summary_table_df, os.path.basename(base) + " Summary", base + "_summary_table.png")
                make_single_panel(result, title_prefix=os.path.basename(base), show_plots=True, save_base=base)
                units = result["summary_df"]["Velocity_units"].iloc[0]
                lines = ["Single dataset analysis", "-" * 80, "Grouped by substrate concentration", result["summary_df"].to_string(index=False), "", "QC flags", result["qc_df"].to_string(index=False), "", "Kinetic parameters", f"MM:  Km = {result['fitres']['Km_mm']:.6g} uM, Vmax = {result['fitres']['Vmax_mm']:.6g} {units}, R^2 = {result['fitres']['MM_R2']:.6f}", f"EH:  Km = {result['fitres']['Km_eh']:.6g} uM, Vmax = {result['fitres']['Vmax_eh']:.6g} {units}, R^2 = {result['fitres']['eh_r2']:.6f}", f"LWB: Km = {result['fitres']['Km_lb']:.6g} uM, Vmax = {result['fitres']['Vmax_lb']:.6g} {units}, R^2 = {result['fitres']['lb_r2']:.6f}"]
                if result["kcat_mm"] is not None:
                    lines += [f"kcat = {result['kcat_mm']:.6g} s^-1", f"kcat/Km = {result['kcat_over_km']:.6g} s^-1 uM^-1"]
                lines += ["", "Saved files", base + "_kinetics_results.xlsx", base + "_kinetics_panel.png", base + "_kinetics_panel.pdf", base + "_summary_table.png"]
                self.output.insert(tk.END, "\n".join(lines))
                self.status_text.set("Single-dataset analysis complete.")
            else:
                wt_path = self.wt_file.get().strip()
                mut_path = self.mut_file.get().strip()
                if not wt_path or not mut_path:
                    raise ValueError("Please choose both WT and mutant files.")
                wt_res = run_dataset_analysis(wt_path, "WT", settings)
                mut_res = run_dataset_analysis(mut_path, "Mutant", settings)
                comp_df = build_comparison_table(wt_res, mut_res)
                base = os.path.splitext(wt_path)[0] + "_vs_" + os.path.splitext(os.path.basename(mut_path))[0]
                export_comparison_results(base + "_comparison_results.xlsx", wt_res, mut_res, comp_df)
                render_summary_table_png(comp_df, os.path.basename(base) + " WT vs Mutant Summary", base + "_comparison_table.png")
                make_comparison_panel(wt_res, mut_res, title_prefix=os.path.basename(base), show_plots=True, save_base=base)
                lines = ["WT vs Mutant parameter comparison", "-" * 80, comp_df.to_string(index=False), "", "Saved files", base + "_comparison_results.xlsx", base + "_comparison_panel.png", base + "_comparison_panel.pdf", base + "_comparison_table.png"]
                self.output.insert(tk.END, "\n".join(lines))
                self.status_text.set("WT vs Mutant comparison complete.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_text.set("Analysis failed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedKineticsGUI(root)
    root.mainloop()
