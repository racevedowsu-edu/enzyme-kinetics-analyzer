
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

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


def style_legend(ax, loc="best", bbox_to_anchor=None):
    leg = ax.legend(fontsize=9, frameon=True, loc=loc, bbox_to_anchor=bbox_to_anchor)
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

    if str(file_path).lower().endswith(".csv"):
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
        substrate = safe_float(str(raw_substrate).replace(",", ""))

        if substrate is None:
            preview_rows.append({
                "Column_Index": col_offset,
                "Header_Value": raw_substrate,
                "Parsed_Substrate_uM": np.nan,
                "Replicate_Name": "",
                "Used": "No",
                "Reason": "Non-numeric or blank header",
                "Valid_Points": 0,
            })
            continue

        if substrate in ignored_substrates:
            preview_rows.append({
                "Column_Index": col_offset,
                "Header_Value": raw_substrate,
                "Parsed_Substrate_uM": substrate,
                "Replicate_Name": "",
                "Used": "No",
                "Reason": "Ignored by user",
                "Valid_Points": 0,
            })
            continue

        absorbance = pd.to_numeric(df.iloc[2:, col_offset], errors="coerce").values
        mask = (~np.isnan(time)) & (~np.isnan(absorbance))
        t = time[mask]
        a = absorbance[mask]

        if len(t) < 3:
            preview_rows.append({
                "Column_Index": col_offset,
                "Header_Value": raw_substrate,
                "Parsed_Substrate_uM": substrate,
                "Replicate_Name": "",
                "Used": "No",
                "Reason": "Fewer than 3 valid points",
                "Valid_Points": int(len(t)),
            })
            continue

        trace_counter[substrate] = trace_counter.get(substrate, 0) + 1
        replicate_name = f"rep{trace_counter[substrate]}"
        grouped.setdefault(substrate, []).append({
            "substrate_uM": substrate,
            "replicate_name": replicate_name,
            "column_index": col_offset,
            "time_s": t,
            "absorbance": a,
        })
        preview_rows.append({
            "Column_Index": col_offset,
            "Header_Value": raw_substrate,
            "Parsed_Substrate_uM": substrate,
            "Replicate_Name": replicate_name,
            "Used": "Yes",
            "Reason": "Used",
            "Valid_Points": int(len(t)),
        })

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
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "fit_idx": fit_idx,
        "velocity_abs_per_s": velocity_abs_per_s,
    }


def convert_absorbance_rate_to_concentration_rate(rate_abs_per_s, epsilon_M_cm=6220.0, pathlength_cm=1.0):
    return (rate_abs_per_s / (epsilon_M_cm * pathlength_cm)) * 1e6


def analyze_grouped_traces(grouped_traces, method, first_n, auto_min_points, auto_max_fraction,
                           convert_to_conc=False, epsilon=6220.0, pathlength_cm=1.0):
    analyzed_traces = []
    for substrate, trace_list in grouped_traces.items():
        for trace in trace_list:
            fitres = fit_initial_velocity(
                trace["time_s"], trace["absorbance"],
                method=method, first_n=first_n,
                auto_min_points=auto_min_points,
                auto_max_fraction=auto_max_fraction
            )
            if convert_to_conc:
                velocity = convert_absorbance_rate_to_concentration_rate(
                    fitres["velocity_abs_per_s"], epsilon, pathlength_cm
                )
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

        df.loc[df["Substrate_uM"] != blank_substrate_value, "Velocity_Corrected"] = (
            df.loc[df["Substrate_uM"] != blank_substrate_value, "Velocity_Raw"] - blank_mean
        )

        blank_summary = pd.DataFrame({
            "Blank_Substrate_uM": [blank_substrate_value],
            "Blank_N": [blank_n],
            "Blank_Mean_Velocity": [blank_mean],
            "Blank_SD_Velocity": [blank_sd],
            "Velocity_units": [df["Velocity_units"].iloc[0] if len(df) else ""],
        })

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


def convert_mg_per_ml_to_uM(stock_mg_per_mL, mw_kda):
    if mw_kda == 0:
        raise ValueError("Molecular weight (kDa) cannot be zero for concentration conversion.")
    return (stock_mg_per_mL * 1000.0) / mw_kda


def run_dataset_analysis(file_path, settings, label="Dataset"):
    grouped_traces, preview_df = load_plate_reader_file(file_path, ignored_substrates=settings["ignored"])
    analyzed_traces, trace_df = analyze_grouped_traces(
        grouped_traces,
        method=settings["method"],
        first_n=settings["first_n"],
        auto_min_points=settings["auto_min"],
        auto_max_fraction=settings["auto_frac"],
        convert_to_conc=settings["convert"],
        epsilon=settings["epsilon"],
        pathlength_cm=settings["pathlength"]
    )
    trace_df, blank_summary_df, blank_mean, blank_sd, blank_n = apply_blank_correction(
        trace_df,
        use_blank_correction=settings["do_blank"],
        blank_substrate_value=settings["blank_sub"]
    )

    summary_df_original_vel = summarize_by_substrate(
        trace_df, use_corrected=settings["do_blank"], exclude_zero_for_fits=True
    )
    fitres_original_vel = fit_mm_and_lwb(summary_df_original_vel)

    final_dimer_uM = np.nan
    final_active_site_uM = np.nan
    kcat_mm = np.nan
    kcat_over_km = np.nan
    mg_protein_in_assay = np.nan

    if settings["compute_kcat"]:
        if not settings["convert"]:
            raise ValueError("kcat requires Beer-Lambert conversion to concentration-based velocities.")
        if settings["enzyme_vol"] <= 0 or settings["stock_dimer_mg_per_mL"] <= 0:
            raise ValueError("Enzyme volume and stock dimer concentration must be positive for kcat calculation.")

        mg_protein_in_assay = (settings["stock_dimer_mg_per_mL"] / 1000.0) * settings["enzyme_vol"]

        stock_dimer_uM = convert_mg_per_ml_to_uM(
            settings["stock_dimer_mg_per_mL"], settings["mw_dimer_kda"]
        )
        final_dimer_uM = calculate_final_dimer_concentration(
            stock_dimer_uM, settings["enzyme_vol"], settings["total_vol"]
        )
        final_active_site_uM = convert_dimer_to_active_site_concentration(
            final_dimer_uM, settings["sites_per_dimer"]
        )

        kcat_mm = compute_kcat(fitres_original_vel["Vmax_mm"], final_active_site_uM)
        kcat_over_km = kcat_mm / fitres_original_vel["Km_mm"]

    summary_df_display = summary_df_original_vel.copy()
    fitres_display = fitres_original_vel.copy()

    if settings["compute_kcat"] and not np.isnan(mg_protein_in_assay) and mg_protein_in_assay > 0:
        scale_factor = 60.0 / mg_protein_in_assay
        summary_df_display["Mean_Velocity"] *= scale_factor
        summary_df_display["SD_Velocity"] *= scale_factor
        summary_df_display["SEM_Velocity"] *= scale_factor
        summary_df_display["Velocity_units"] = "µmol NADH/min/mg protein"
        fitres_display = fit_mm_and_lwb(summary_df_display)

    return {
        "label": label,
        "preview_df": preview_df,
        "analyzed_traces": analyzed_traces,
        "trace_df": trace_df,
        "summary_df": summary_df_display,
        "fitres": fitres_display,
        "blank_summary_df": blank_summary_df,
        "blank_mean": blank_mean,
        "blank_sd": blank_sd,
        "blank_n": blank_n,
        "final_dimer_uM": final_dimer_uM,
        "final_active_site_uM": final_active_site_uM,
        "kcat_mm": kcat_mm,
        "kcat_over_km": kcat_over_km,
    }


def make_single_panel(result, title_prefix="", show_plots=True, save_base=None):
    units = result["summary_df"]["Velocity_units"].iloc[0] if len(result["summary_df"]) else "Velocity"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    concentration_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*', 'X', '+']
    colorblind_colors = ['#E69F00', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#000000']

    substrates = sorted({x["substrate_uM"] for x in result["analyzed_traces"]})
    substrate_to_marker_idx = {sub: i % len(concentration_markers) for i, sub in enumerate(substrates)}
    substrate_to_color_idx = {sub: i % len(colorblind_colors) for i, sub in enumerate(substrates)}

    for substrate in substrates:
        traces = [x for x in result["analyzed_traces"] if x["substrate_uM"] == substrate]
        mean_v = np.nanmean([tr.get("velocity_raw", np.nan) for tr in traces])
        tr = traces[0]

        t = tr["time_s"]
        a = tr["absorbance"]
        fit_idx = tr["fit_idx"]

        current_marker = concentration_markers[substrate_to_marker_idx[substrate]]
        current_color = colorblind_colors[substrate_to_color_idx[substrate]]

        fit_mask = np.zeros(len(t), dtype=bool)
        fit_mask[fit_idx] = True

        ax1.plot(t, a, marker=current_marker, linestyle='None', color=current_color,
                 markersize=MARKER_SIZE, linewidth=1.5, alpha=0.25, label=None)
        ax1.plot(t[fit_mask], a[fit_mask], marker=current_marker, linestyle='None',
                 color=current_color, markersize=MARKER_SIZE, alpha=1.0,
                 label=f"{substrate:.0f} uM ({mean_v:.2f})")

        xfit = t[fit_idx]
        yfit = tr["intercept"] + tr["slope"] * xfit
        ax1.plot(xfit, yfit, linestyle='--', color=current_color, linewidth=2, alpha=1.0)

    style_axes(ax1, "Figure 1. Progress Curves", "Time (s)", "Absorbance")
    style_legend(ax1)

    all_summary = result["summary_df"]
    if not all_summary.empty:
        excluded_summary = all_summary[all_summary["Used_For_MM_LWB"] == "No"]
        if not excluded_summary.empty:
            ax2.errorbar(excluded_summary["Substrate_uM"], excluded_summary["Mean_Velocity"],
                         yerr=excluded_summary["SD_Velocity"], fmt=WT_MARKER, color=WT_COLOR,
                         markersize=MARKER_SIZE, capsize=4, alpha=0.5, fillstyle='none',
                         label="_nolegend_")
        used_for_fit_summary = all_summary[all_summary["Used_For_MM_LWB"] == "Yes"]
        if not used_for_fit_summary.empty:
            ax2.errorbar(used_for_fit_summary["Substrate_uM"], used_for_fit_summary["Mean_Velocity"],
                         yerr=used_for_fit_summary["SD_Velocity"], fmt=WT_MARKER, color=WT_COLOR,
                         markersize=MARKER_SIZE, capsize=4, alpha=1.0, fillstyle='full',
                         label="_nolegend_")

    S_for_curve = result["fitres"]["S"]
    if len(S_for_curve) > 0:
        Sfit = np.linspace(0, max(S_for_curve) * 1.1, 300)
        ax2.plot(Sfit, michaelis_menten(Sfit, result["fitres"]["Vmax_mm"], result["fitres"]["Km_mm"]),
                 color=WT_COLOR, linestyle=WT_LINE, linewidth=LINE_WIDTH,
                 label=f"MM: Km={result['fitres']['Km_mm']:.2f}±{result['fitres']['Km_mm_sd']:.2f} uM\n"
                       f"Vmax={result['fitres']['Vmax_mm']:.2f}±{result['fitres']['Vmax_mm_sd']:.2f} {units}")
    style_axes(ax2, "Figure 2. Michaelis-Menten", "Substrate (uM)", f"Initial velocity ({units})")
    style_legend(ax2, loc="lower right")

    x_eh, y_eh = result["fitres"]["v"] / result["fitres"]["S"], result["fitres"]["v"]
    xline_eh = np.linspace(0, max(x_eh) * 1.1, 300)
    ax3.scatter(x_eh, y_eh, color=WT_COLOR, marker=WT_MARKER, s=MARKER_SIZE ** 2, label="_nolegend_")
    ax3.plot(xline_eh, result["fitres"]["Vmax_mm"] - result["fitres"]["Km_mm"] * xline_eh,
             color=WT_COLOR, linestyle=WT_LINE, linewidth=LINE_WIDTH,
             label=f"From MM: Km={result['fitres']['Km_mm']:.2f}±{result['fitres']['Km_mm_sd']:.2f} uM\n"
                   f"Vmax={result['fitres']['Vmax_mm']:.2f}±{result['fitres']['Vmax_mm_sd']:.2f} {units}")
    ax3.plot(xline_eh, result["fitres"]["eh_intercept"] + result["fitres"]["eh_slope"] * xline_eh,
             color="black", linestyle="--", linewidth=LINE_WIDTH,
             label=f"EH fit: Km={result['fitres']['Km_eh']:.2f}±{result['fitres']['Km_eh_sd']:.2f} uM\n"
                   f"Vmax={result['fitres']['Vmax_eh']:.2f}±{result['fitres']['Vmax_eh_sd']:.2f} {units}")
    style_axes(ax3, "Figure 3. Eadie-Hofstee", f"v/[S] ({units}/uM)", f"v ({units})")
    style_legend(ax3)

    invS, invv = 1.0 / result["fitres"]["S"], 1.0 / result["fitres"]["v"]
    xline = np.linspace(0, max(invS) * 1.1, 300)
    yline = result["fitres"]["lb_intercept"] + result["fitres"]["lb_slope"] * xline
    ax4.scatter(invS, invv, color=WT_COLOR, marker=WT_MARKER, s=MARKER_SIZE ** 2, label="_nolegend_")
    ax4.plot(xline, yline, color=WT_COLOR, linestyle=WT_LINE, linewidth=LINE_WIDTH,
             label=f"LWB: Km={result['fitres']['Km_lb']:.2f}±{result['fitres']['Km_lb_sd']:.2f} uM\n"
                   f"Vmax={result['fitres']['Vmax_lb']:.2f}±{result['fitres']['Vmax_lb_sd']:.2f} {units}")
    style_axes(ax4, "Figure 4. Lineweaver-Burk", "1/[S] (1/uM)", f"1/v (1/({units}))")
    style_legend(ax4)

    fig.suptitle(f"{title_prefix} Kinetic Analysis", fontsize=15, fontweight="bold", color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_base:
        fig.savefig(save_base + "_kinetics_panel.png", dpi=300, bbox_inches="tight")
        fig.savefig(save_base + "_kinetics_panel.pdf", bbox_inches="tight")
    if show_plots:
        plt.show()


def make_multi_panel(all_results, title_prefix="", show_plots=True, save_base=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()

    concentration_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*', 'X', '+']
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    num_colors = len(colors)
    units = None

    for i, (label, result) in enumerate(all_results.items()):
        is_wt = label == 'WT'
        color = WT_COLOR if is_wt else colors[i % num_colors]
        dataset_marker = WT_MARKER if is_wt else concentration_markers[i % len(concentration_markers)]
        linestyle = WT_LINE if is_wt else MUT_LINE
        progress_curve_color_base = colors[i % num_colors]
        units = result["summary_df"]["Velocity_units"].iloc[0] if len(result["summary_df"]) else "Velocity"

        substrates_in_result = sorted({x["substrate_uM"] for x in result["analyzed_traces"]})
        substrate_to_marker_idx = {sub: j % len(concentration_markers) for j, sub in enumerate(substrates_in_result)}

        for substrate in substrates_in_result:
            traces_at_sub = [x for x in result["analyzed_traces"] if x["substrate_uM"] == substrate]
            if traces_at_sub:
                tr = traces_at_sub[0]
                t = tr["time_s"]
                a = tr["absorbance"]
                fit_idx = tr["fit_idx"]
                fit_mask = np.zeros(len(t), dtype=bool)
                fit_mask[fit_idx] = True
                current_concentration_marker = concentration_markers[substrate_to_marker_idx[substrate]]

                ax1.plot(t[fit_mask], a[fit_mask], marker=current_concentration_marker,
                         linestyle='None', color=progress_curve_color_base,
                         markersize=MARKER_SIZE, alpha=1.0,
                         label=f"{label} {substrate:.0f} uM")

                xfit = t[fit_idx]
                yfit = tr["intercept"] + tr["slope"] * xfit
                ax1.plot(xfit, yfit, linestyle=linestyle, color=progress_curve_color_base,
                         linewidth=LINE_WIDTH, alpha=1.0)

        all_summary = result["summary_df"]
        if not all_summary.empty:
            used_for_fit_summary = all_summary[all_summary["Used_For_MM_LWB"] == "Yes"]
            if not used_for_fit_summary.empty:
                ax2.errorbar(used_for_fit_summary["Substrate_uM"], used_for_fit_summary["Mean_Velocity"],
                             yerr=used_for_fit_summary["SD_Velocity"], fmt=dataset_marker, color=color,
                             markersize=MARKER_SIZE, capsize=4, alpha=1.0, fillstyle='full',
                             label=f'{label} Data')

        S_for_curve = result["fitres"]["S"]
        if len(S_for_curve) > 0:
            Sfit = np.linspace(0, max(S_for_curve) * 1.1, 300)
            ax2.plot(Sfit, michaelis_menten(Sfit, result["fitres"]["Vmax_mm"], result["fitres"]["Km_mm"]),
                     color=color, linestyle=linestyle, linewidth=LINE_WIDTH,
                     label=f"{label} MM Fit: Km={result['fitres']['Km_mm']:.2g}, Vmax={result['fitres']['Vmax_mm']:.2g}")

        if len(result["fitres"]["S"]) > 0 and len(result["fitres"]["v"]) > 0:
            x_eh, y_eh = result["fitres"]["v"] / result["fitres"]["S"], result["fitres"]["v"]
            xline_eh = np.linspace(0, max(x_eh) * 1.1, 300)
            ax3.scatter(x_eh, y_eh, color=color, marker=dataset_marker, s=MARKER_SIZE**2, label=f'{label} Data')
            ax3.plot(xline_eh, result["fitres"]["Vmax_mm"] - result["fitres"]["Km_mm"] * xline_eh,
                     color=color, linestyle=linestyle, linewidth=LINE_WIDTH,
                     label=f"{label} MM-derived line")

            invS, invv = 1.0 / result["fitres"]["S"], 1.0 / result["fitres"]["v"]
            xline_lb = np.linspace(0, max(invS) * 1.1, 300)
            yline_lb = result["fitres"]["lb_intercept"] + result["fitres"]["lb_slope"] * xline_lb
            ax4.scatter(invS, invv, color=color, marker=dataset_marker, s=MARKER_SIZE**2, label=f'{label} Data')
            ax4.plot(xline_lb, yline_lb, color=color, linestyle=linestyle, linewidth=LINE_WIDTH,
                     label=f"{label} LWB Fit: Km={result['fitres']['Km_lb']:.2g}, Vmax={result['fitres']['Vmax_lb']:.2g}")

    style_axes(ax1, "Figure 1. Progress Curves", "Time (s)", "Absorbance")
    style_legend(ax1, loc="upper left", bbox_to_anchor=(1, 1))
    style_axes(ax2, "Figure 2. Michaelis-Menten", "Substrate (uM)", f"Initial velocity ({units})")
    style_legend(ax2, loc="lower right")
    style_axes(ax3, "Figure 3. Eadie-Hofstee", f"v/[S] ({units}/uM)", f"v ({units})")
    style_legend(ax3)
    style_axes(ax4, "Figure 4. Lineweaver-Burk", "1/[S] (1/uM)", f"1/v (1/({units}))")
    style_legend(ax4)

    fig.suptitle(f"{title_prefix} Kinetic Analysis", fontsize=16, fontweight="bold", color="black")
    fig.tight_layout(rect=[0, 0, 0.95, 0.97])

    if save_base:
        fig.savefig(save_base + "_comparison_panel.png", dpi=300, bbox_inches="tight")
        fig.savefig(save_base + "_comparison_panel.pdf", bbox_inches="tight")
    if show_plots:
        plt.show()


def render_summary_table_png(df, title, outpath):
    fig_h = max(2.5, 0.45 * (len(df) + 2))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", color="black", pad=12)

    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.3g}")

    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                     loc="center", cellLoc="center", colLoc="center")
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
    if not np.isnan(result["kcat_mm"]):
        fit_table = pd.concat([fit_table, pd.DataFrame({
            "Parameter": ["kcat_MM_per_active_site_s^-1", "kcat_over_Km_s^-1_uM^-1"],
            "Value": [result["kcat_mm"], result["kcat_over_km"]]
        })], ignore_index=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result["preview_df"].to_excel(writer, sheet_name="Column_Preview", index=False)
        result["trace_df"].to_excel(writer, sheet_name="Per_Trace_Velocities", index=False)
        result["summary_df"].to_excel(writer, sheet_name="Grouped_By_Substrate", index=False)
        fit_table.to_excel(writer, sheet_name="Kinetic_Fits", index=False)
        if result["blank_summary_df"] is not None:
            result["blank_summary_df"].to_excel(writer, sheet_name="Blank_Correction", index=False)


def export_multi_results(output_path, all_results, comparison_df):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for label, result in all_results.items():
            safe_label = label.replace(" ", "_")[:20]
            result["preview_df"].to_excel(writer, sheet_name=f"{safe_label}_Preview", index=False)
            result["trace_df"].to_excel(writer, sheet_name=f"{safe_label}_Trace", index=False)
            result["summary_df"].to_excel(writer, sheet_name=f"{safe_label}_Grouped", index=False)
        comparison_df.to_excel(writer, sheet_name="Comparison", index=False)


class MutantInputFrame(ttk.LabelFrame):
    def __init__(self, master, idx, remove_callback):
        super().__init__(master, text=f"Mutant {idx}", padding=8)
        self.idx = idx
        self.remove_callback = remove_callback

        self.file_var = tk.StringVar()
        self.exclude_var = tk.StringVar()
        self.stock_var = tk.StringVar(value="1.0")
        self.mw_var = tk.StringVar(value="60.0")
        self.enzyme_vol_var = tk.StringVar(value="10.0")
        self.total_vol_var = tk.StringVar(value="1000.0")
        self.sites_var = tk.StringVar(value="2.0")

        ttk.Label(self, text="File:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.file_var, width=55).grid(row=0, column=1, columnspan=3, sticky="ew")
        ttk.Button(self, text="Browse", command=self.browse).grid(row=0, column=4, padx=4)
        ttk.Label(self, text="Exclude uM:").grid(row=1, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.exclude_var, width=18).grid(row=1, column=1, sticky="w")
        ttk.Button(self, text="Remove", command=self.remove_self).grid(row=1, column=4, sticky="e", padx=4)

        ttk.Label(self, text="Stock (mg/mL):").grid(row=2, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.stock_var, width=12).grid(row=2, column=1, sticky="w")
        ttk.Label(self, text="MW dimer (kDa):").grid(row=2, column=2, sticky="w")
        ttk.Entry(self, textvariable=self.mw_var, width=12).grid(row=2, column=3, sticky="w")

        ttk.Label(self, text="Enzyme (uL):").grid(row=3, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.enzyme_vol_var, width=12).grid(row=3, column=1, sticky="w")
        ttk.Label(self, text="Total (uL):").grid(row=3, column=2, sticky="w")
        ttk.Entry(self, textvariable=self.total_vol_var, width=12).grid(row=3, column=3, sticky="w")

        ttk.Label(self, text="Sites/dimer:").grid(row=4, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.sites_var, width=12).grid(row=4, column=1, sticky="w")

        for c in range(5):
            self.columnconfigure(c, weight=1 if c in (1, 3) else 0)

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.file_var.set(path)

    def remove_self(self):
        self.remove_callback(self)

    def update_idx(self, idx):
        self.idx = idx
        self.config(text=f"Mutant {idx}")

    def set_kcat_visibility(self, visible):
        state = "normal" if visible else "disabled"
        for child in self.winfo_children():
            txt = getattr(child, "cget", lambda *a, **k: "")("text") if hasattr(child, "cget") else ""
            # keep file and exclude fields enabled
            if txt in {"File:", "Exclude uM:"}:
                continue
            if isinstance(child, ttk.Entry):
                # entries mapped by variable names; disable kcat entries only
                var = str(child.cget("textvariable"))
                if var in {str(self.stock_var), str(self.mw_var), str(self.enzyme_vol_var), str(self.total_vol_var), str(self.sites_var)}:
                    child.state(["!disabled"] if visible else ["disabled"])


class EnzymeKineticsDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enzyme Kinetics Analyzer - Desktop Port")
        self.mode = tk.StringVar(value="single")

        self.single_file = tk.StringVar()
        self.single_exclude = tk.StringVar()

        self.wt_file = tk.StringVar()
        self.wt_exclude = tk.StringVar()

        self.fit_method = tk.StringVar(value="first_n")
        self.first_n = tk.StringVar(value="5")
        self.auto_min = tk.StringVar(value="4")
        self.auto_frac = tk.StringVar(value="0.5")

        self.convert_to_conc = tk.BooleanVar(value=True)
        self.epsilon = tk.StringVar(value="6220")
        self.pathlength = tk.StringVar(value="1.0")

        self.use_blank = tk.BooleanVar(value=True)
        self.blank_sub = tk.StringVar(value="0")

        self.compute_kcat = tk.BooleanVar(value=False)

        self.single_stock = tk.StringVar(value="1.0")
        self.single_mw = tk.StringVar(value="60.0")
        self.single_enzyme_vol = tk.StringVar(value="10.0")
        self.single_total_vol = tk.StringVar(value="1000.0")
        self.single_sites = tk.StringVar(value="2.0")

        self.wt_stock = tk.StringVar(value="1.0")
        self.wt_mw = tk.StringVar(value="60.0")
        self.wt_enzyme_vol = tk.StringVar(value="10.0")
        self.wt_total_vol = tk.StringVar(value="1000.0")
        self.wt_sites = tk.StringVar(value="2.0")

        self.mutant_frames = []

        self.status_text = tk.StringVar(value="Choose files, adjust settings, and click Analyze.")
        self.build_gui()
        self.update_mode_visibility()
        self.update_kcat_visibility()

    def configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        style.configure("Subtle.TLabel", foreground="#444444")
        style.configure("Section.TLabelframe", padding=10)
        style.configure("Section.TLabelframe.Label", font=("Arial", 11, "bold"))
        style.configure("Action.TButton", padding=(10, 6))

    def bind_mousewheel(self, widget):
        def _on_mousewheel(event):
            delta = 0
            if hasattr(event, 'delta') and event.delta:
                delta = int(-1 * (event.delta / 120))
            elif getattr(event, 'num', None) == 4:
                delta = -1
            elif getattr(event, 'num', None) == 5:
                delta = 1
            if delta:
                self.canvas.yview_scroll(delta, 'units')

        widget.bind_all("<MouseWheel>", _on_mousewheel)
        widget.bind_all("<Button-4>", _on_mousewheel)
        widget.bind_all("<Button-5>", _on_mousewheel)

    def build_gui(self):
        self.configure_styles()

        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Enzyme Kinetics Analyzer", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Desktop version with single-dataset and WT + multiple-mutant workflows.", style="Subtle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))

        paned = ttk.Panedwindow(outer, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self.canvas = tk.Canvas(left, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.canvas.yview)
        self.scrollable = ttk.Frame(self.canvas)
        self.scrollable.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.bind_mousewheel(self.canvas)

        frm = self.scrollable
        frm.columnconfigure(0, weight=1)

        mode_card = ttk.LabelFrame(frm, text="Mode", style="Section.TLabelframe")
        mode_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(mode_card, text="Single dataset", variable=self.mode, value="single", command=self.update_mode_visibility).grid(row=0, column=0, padx=(0, 12), sticky="w")
        ttk.Radiobutton(mode_card, text="Multi-dataset (WT + mutants)", variable=self.mode, value="multi", command=self.update_mode_visibility).grid(row=0, column=1, sticky="w")

        files_nb = ttk.Notebook(frm)
        files_nb.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        single_tab = ttk.Frame(files_nb, padding=8)
        multi_tab = ttk.Frame(files_nb, padding=8)
        files_nb.add(single_tab, text="Single dataset")
        files_nb.add(multi_tab, text="Multi-dataset")
        single_tab.columnconfigure(1, weight=1)
        multi_tab.columnconfigure(1, weight=1)

        self.single_box = ttk.LabelFrame(single_tab, text="Single dataset file", style="Section.TLabelframe")
        self.single_box.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.single_box.columnconfigure(1, weight=1)
        ttk.Label(self.single_box, text="File:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(self.single_box, textvariable=self.single_file, width=72).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(self.single_box, text="Browse", command=lambda: self.browse_file(self.single_file)).grid(row=0, column=2, padx=4)
        ttk.Label(self.single_box, text="Exclude uM:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(self.single_box, textvariable=self.single_exclude, width=20).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(self.single_box, text="Comma-separated values to omit from fitting and plotting.", style="Subtle.TLabel").grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))

        self.multi_box = ttk.LabelFrame(multi_tab, text="WT and mutant files", style="Section.TLabelframe")
        self.multi_box.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.multi_box.columnconfigure(1, weight=1)
        ttk.Label(self.multi_box, text="WT file:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.multi_box, textvariable=self.wt_file, width=72).grid(row=0, column=1, sticky="ew")
        ttk.Button(self.multi_box, text="Browse WT", command=lambda: self.browse_file(self.wt_file)).grid(row=0, column=2, padx=4)
        ttk.Label(self.multi_box, text="WT Exclude uM:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(self.multi_box, textvariable=self.wt_exclude, width=20).grid(row=1, column=1, sticky="w", pady=(4, 0))

        self.wt_kcat_frame = ttk.LabelFrame(self.multi_box, text="WT kcat inputs", style="Section.TLabelframe")
        self.wt_kcat_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        for i in range(4):
            self.wt_kcat_frame.columnconfigure(i, weight=1)
        ttk.Label(self.wt_kcat_frame, text="Stock (mg/mL):").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.wt_kcat_frame, textvariable=self.wt_stock, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(self.wt_kcat_frame, text="MW dimer (kDa):").grid(row=0, column=2, sticky="w")
        ttk.Entry(self.wt_kcat_frame, textvariable=self.wt_mw, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(self.wt_kcat_frame, text="Enzyme (uL):").grid(row=1, column=0, sticky="w")
        ttk.Entry(self.wt_kcat_frame, textvariable=self.wt_enzyme_vol, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(self.wt_kcat_frame, text="Total (uL):").grid(row=1, column=2, sticky="w")
        ttk.Entry(self.wt_kcat_frame, textvariable=self.wt_total_vol, width=10).grid(row=1, column=3, sticky="w")
        ttk.Label(self.wt_kcat_frame, text="Sites/dimer:").grid(row=2, column=0, sticky="w")
        ttk.Entry(self.wt_kcat_frame, textvariable=self.wt_sites, width=10).grid(row=2, column=1, sticky="w")

        ttk.Button(self.multi_box, text="Add Mutant", style="Action.TButton", command=self.add_mutant_frame).grid(row=3, column=0, sticky="w", pady=(6, 6))
        ttk.Label(self.multi_box, text="Add as many mutant datasets as you need.", style="Subtle.TLabel").grid(row=3, column=1, columnspan=2, sticky="w", pady=(6, 6))
        self.mutant_container = ttk.Frame(self.multi_box)
        self.mutant_container.grid(row=4, column=0, columnspan=3, sticky="ew")
        self.mutant_container.columnconfigure(0, weight=1)

        settings_nb = ttk.Notebook(frm)
        settings_nb.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        fit_tab = ttk.Frame(settings_nb, padding=8)
        conversion_tab = ttk.Frame(settings_nb, padding=8)
        kcat_tab = ttk.Frame(settings_nb, padding=8)
        settings_nb.add(fit_tab, text="Fit settings")
        settings_nb.add(conversion_tab, text="Conversion / blank")
        settings_nb.add(kcat_tab, text="kcat")

        fit_card = ttk.LabelFrame(fit_tab, text="Initial velocity settings", style="Section.TLabelframe")
        fit_card.grid(row=0, column=0, sticky="ew")
        ttk.Label(fit_card, text="Fit method:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(fit_card, text="First N points", variable=self.fit_method, value="first_n").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(fit_card, text="Auto-search early linear region", variable=self.fit_method, value="auto").grid(row=0, column=2, sticky="w")
        ttk.Label(fit_card, text="First N:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(fit_card, textvariable=self.first_n, width=10).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(fit_card, text="Auto min points:").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Entry(fit_card, textvariable=self.auto_min, width=10).grid(row=1, column=3, sticky="w", pady=(6, 0))
        ttk.Label(fit_card, text="Auto max fraction:").grid(row=1, column=4, sticky="w", pady=(6, 0))
        ttk.Entry(fit_card, textvariable=self.auto_frac, width=10).grid(row=1, column=5, sticky="w", pady=(6, 0))

        conv_card = ttk.LabelFrame(conversion_tab, text="Units and blank correction", style="Section.TLabelframe")
        conv_card.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(conv_card, text="Convert Abs/s to uM/s with Beer-Lambert", variable=self.convert_to_conc).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(conv_card, text="Epsilon:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(conv_card, textvariable=self.epsilon, width=10).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(conv_card, text="Path length (cm):").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Entry(conv_card, textvariable=self.pathlength, width=10).grid(row=1, column=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(conv_card, text="Apply blank correction", variable=self.use_blank).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(conv_card, text="Blank substrate (uM):").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(conv_card, textvariable=self.blank_sub, width=10).grid(row=2, column=3, sticky="w", pady=(8, 0))

        kcat_card = ttk.LabelFrame(kcat_tab, text="Turnover settings", style="Section.TLabelframe")
        kcat_card.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(kcat_card, text="Compute kcat and kcat/Km", variable=self.compute_kcat, command=self.update_kcat_visibility).grid(row=0, column=0, sticky="w")
        ttk.Label(kcat_card, text="When enabled, desktop outputs also display normalized velocities in µmol NADH/min/mg protein.", style="Subtle.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.single_kcat_frame = ttk.LabelFrame(kcat_tab, text="Single-dataset kcat inputs", style="Section.TLabelframe")
        self.single_kcat_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for i in range(4):
            self.single_kcat_frame.columnconfigure(i, weight=1)
        ttk.Label(self.single_kcat_frame, text="Stock (mg/mL):").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.single_kcat_frame, textvariable=self.single_stock, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(self.single_kcat_frame, text="MW dimer (kDa):").grid(row=0, column=2, sticky="w")
        ttk.Entry(self.single_kcat_frame, textvariable=self.single_mw, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(self.single_kcat_frame, text="Enzyme (uL):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.single_kcat_frame, textvariable=self.single_enzyme_vol, width=10).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(self.single_kcat_frame, text="Total (uL):").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Entry(self.single_kcat_frame, textvariable=self.single_total_vol, width=10).grid(row=1, column=3, sticky="w", pady=(6, 0))
        ttk.Label(self.single_kcat_frame, text="Sites/dimer:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.single_kcat_frame, textvariable=self.single_sites, width=10).grid(row=2, column=1, sticky="w", pady=(6, 0))

        action_bar = ttk.Frame(frm)
        action_bar.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        action_bar.columnconfigure(2, weight=1)
        ttk.Button(action_bar, text="Analyze", style="Action.TButton", command=self.analyze).grid(row=0, column=0, sticky="w")
        ttk.Button(action_bar, text="Clear Results", command=lambda: self.output.delete("1.0", tk.END)).grid(row=0, column=1, padx=(8, 0), sticky="w")
        ttk.Label(action_bar, textvariable=self.status_text, style="Subtle.TLabel").grid(row=0, column=2, sticky="e")

        results_header = ttk.Label(right, text="Results", style="Header.TLabel")
        results_header.grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.output = scrolledtext.ScrolledText(right, width=70, height=32, wrap=tk.WORD, font=("Consolas", 10))
        self.output.grid(row=1, column=0, sticky="nsew")

    def browse_file(self, var):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            var.set(path)

    def add_mutant_frame(self):
        idx = len(self.mutant_frames) + 1
        frame = MutantInputFrame(self.mutant_container, idx, self.remove_mutant_frame)
        frame.grid(row=len(self.mutant_frames), column=0, sticky="ew", pady=4)
        self.mutant_frames.append(frame)
        self.update_kcat_visibility()

    def remove_mutant_frame(self, frame):
        frame.destroy()
        self.mutant_frames = [m for m in self.mutant_frames if m is not frame]
        for i, m in enumerate(self.mutant_frames, start=1):
            m.grid_configure(row=i-1)
            m.update_idx(i)

    def update_mode_visibility(self):
        if self.mode.get() == "single":
            self.single_box.grid()
            self.multi_box.grid_remove()
            self.single_kcat_frame.grid()
        else:
            self.single_box.grid_remove()
            self.multi_box.grid()
            self.single_kcat_frame.grid_remove()
            if not self.mutant_frames:
                self.add_mutant_frame()
        self.update_kcat_visibility()

    def update_kcat_visibility(self):
        single_show = self.compute_kcat.get() and self.mode.get() == "single"
        multi_show = self.compute_kcat.get() and self.mode.get() == "multi"

        if single_show:
            self.single_kcat_frame.grid()
        else:
            self.single_kcat_frame.grid_remove()

        for child in self.wt_kcat_frame.winfo_children():
            if isinstance(child, ttk.Entry):
                child.state(["!disabled"] if multi_show else ["disabled"])
        for frame in self.mutant_frames:
            frame.set_kcat_visibility(multi_show)

    def parse_excluded(self, text):
        text = text.strip()
        if not text:
            return []
        return [float(x.strip()) for x in text.split(",") if x.strip()]

    def common_settings(self):
        return {
            "method": self.fit_method.get(),
            "first_n": int(self.first_n.get()),
            "auto_min": int(self.auto_min.get()),
            "auto_frac": float(self.auto_frac.get()),
            "convert": self.convert_to_conc.get(),
            "epsilon": float(self.epsilon.get()),
            "pathlength": float(self.pathlength.get()),
            "do_blank": self.use_blank.get(),
            "blank_sub": float(self.blank_sub.get()),
            "compute_kcat": self.compute_kcat.get(),
        }

    def build_single_settings(self):
        s = self.common_settings()
        s.update({
            "ignored": self.parse_excluded(self.single_exclude.get()),
            "stock_dimer_mg_per_mL": float(self.single_stock.get()),
            "mw_dimer_kda": float(self.single_mw.get()),
            "enzyme_vol": float(self.single_enzyme_vol.get()),
            "total_vol": float(self.single_total_vol.get()),
            "sites_per_dimer": float(self.single_sites.get()),
        })
        return s

    def build_wt_settings(self):
        s = self.common_settings()
        s.update({
            "ignored": self.parse_excluded(self.wt_exclude.get()),
            "stock_dimer_mg_per_mL": float(self.wt_stock.get()),
            "mw_dimer_kda": float(self.wt_mw.get()),
            "enzyme_vol": float(self.wt_enzyme_vol.get()),
            "total_vol": float(self.wt_total_vol.get()),
            "sites_per_dimer": float(self.wt_sites.get()),
        })
        return s

    def build_mut_settings(self, frame):
        s = self.common_settings()
        s.update({
            "ignored": self.parse_excluded(frame.exclude_var.get()),
            "stock_dimer_mg_per_mL": float(frame.stock_var.get()),
            "mw_dimer_kda": float(frame.mw_var.get()),
            "enzyme_vol": float(frame.enzyme_vol_var.get()),
            "total_vol": float(frame.total_vol_var.get()),
            "sites_per_dimer": float(frame.sites_var.get()),
        })
        return s

    def analyze(self):
        self.output.delete("1.0", tk.END)
        try:
            output_dir = Path.cwd() / "desktop_widget_outputs"
            output_dir.mkdir(exist_ok=True)

            if self.mode.get() == "single":
                file_path = self.single_file.get().strip()
                if not file_path:
                    raise ValueError("Please choose a single dataset file.")

                result = run_dataset_analysis(file_path, self.build_single_settings(), label="Dataset")
                base = str(output_dir / Path(file_path).stem)
                export_single_results(base + "_results.xlsx", result)

                summary_table_df = pd.DataFrame({
                    "Method": ["MM", "EH", "LWB"],
                    "Km (uM)": [result["fitres"]["Km_mm"], result["fitres"]["Km_eh"], result["fitres"]["Km_lb"]],
                    "SD(Km)": [result["fitres"]["Km_mm_sd"], result["fitres"]["Km_eh_sd"], result["fitres"]["Km_lb_sd"]],
                    "Vmax": [result["fitres"]["Vmax_mm"], result["fitres"]["Vmax_eh"], result["fitres"]["Vmax_lb"]],
                    "SD(Vmax)": [result["fitres"]["Vmax_mm_sd"], result["fitres"]["Vmax_eh_sd"], result["fitres"]["Vmax_lb_sd"]],
                    "R^2": [result["fitres"]["MM_R2"], result["fitres"]["eh_r2"], result["fitres"]["lb_r2"]],
                    "Vmax_Units": [result["summary_df"]["Velocity_units"].iloc[0]] * 3,
                })
                if not np.isnan(result["kcat_mm"]):
                    summary_table_df["kcat (s^-1)"] = [result["kcat_mm"], np.nan, np.nan]
                    summary_table_df["kcat/Km (s^-1 uM^-1)"] = [result["kcat_over_km"], np.nan, np.nan]

                render_summary_table_png(summary_table_df, Path(base).name + " Summary", base + "_summary_table.png")
                make_single_panel(result, title_prefix=Path(base).name, show_plots=True, save_base=base)

                lines = [
                    "Single dataset analysis",
                    "-" * 90,
                    f"Loaded file: {file_path}",
                    "",
                    "Grouped by substrate concentration",
                    result["summary_df"].to_string(index=False),
                    "",
                    "Kinetic summary",
                    summary_table_df.to_string(index=False),
                    "",
                    "Saved files",
                    base + "_results.xlsx",
                    base + "_summary_table.png",
                    base + "_kinetics_panel.png",
                    base + "_kinetics_panel.pdf",
                ]
                self.output.insert(tk.END, "\n".join(lines))
                self.status_text.set("Single-dataset analysis complete.")

            else:
                wt_path = self.wt_file.get().strip()
                if not wt_path:
                    raise ValueError("Please choose a WT file.")
                if not self.mutant_frames:
                    raise ValueError("Add at least one mutant dataset.")

                all_results = {"WT": run_dataset_analysis(wt_path, self.build_wt_settings(), label="WT")}

                for i, frame in enumerate(self.mutant_frames, start=1):
                    mut_path = frame.file_var.get().strip()
                    if not mut_path:
                        raise ValueError(f"Please choose a file for Mutant {i}.")
                    all_results[f"Mutant {i}"] = run_dataset_analysis(mut_path, self.build_mut_settings(frame), label=f"Mutant {i}")

                rows = []
                wt_res = all_results["WT"]
                wt_km = wt_res["fitres"]["Km_mm"]
                wt_vmax = wt_res["fitres"]["Vmax_mm"]
                wt_kcat = wt_res["kcat_mm"]
                wt_kcatkm = wt_res["kcat_over_km"]

                for label, res in all_results.items():
                    row = {
                        "Dataset": label,
                        "Km (uM)": res["fitres"]["Km_mm"],
                        "Vmax": res["fitres"]["Vmax_mm"],
                        "Vmax_Units": res["summary_df"]["Velocity_units"].iloc[0],
                        "kcat (s^-1)": res["kcat_mm"],
                        "kcat/Km (s^-1 uM^-1)": res["kcat_over_km"],
                    }
                    if label != "WT":
                        row["Km Difference (Mut - WT)"] = res["fitres"]["Km_mm"] - wt_km
                        row["Km % Change"] = ((res["fitres"]["Km_mm"] - wt_km) / wt_km) * 100 if wt_km else np.nan
                        row["Vmax Difference (Mut - WT)"] = res["fitres"]["Vmax_mm"] - wt_vmax
                        row["Vmax % Change"] = ((res["fitres"]["Vmax_mm"] - wt_vmax) / wt_vmax) * 100 if wt_vmax else np.nan
                        if not np.isnan(res["kcat_mm"]) and not np.isnan(wt_kcat) and wt_kcat != 0:
                            row["kcat Difference (Mut - WT)"] = res["kcat_mm"] - wt_kcat
                            row["kcat % Change"] = ((res["kcat_mm"] - wt_kcat) / wt_kcat) * 100
                        if not np.isnan(res["kcat_over_km"]) and not np.isnan(wt_kcatkm) and wt_kcatkm != 0:
                            row["kcat/Km Difference (Mut - WT)"] = res["kcat_over_km"] - wt_kcatkm
                            row["kcat/Km % Change"] = ((res["kcat_over_km"] - wt_kcatkm) / wt_kcatkm) * 100
                    rows.append(row)

                comp_df = pd.DataFrame(rows)
                base_parts = [Path(wt_path).stem] + [Path(f.file_var.get().strip()).stem for f in self.mutant_frames]
                base = str(output_dir / "_vs_".join(base_parts))
                export_multi_results(base + "_comparison_results.xlsx", all_results, comp_df)
                render_summary_table_png(comp_df, Path(base).name + " Comparison", base + "_comparison_table.png")
                make_multi_panel(all_results, title_prefix=Path(base).name, show_plots=True, save_base=base)

                lines = [
                    "Multi-dataset analysis",
                    "-" * 90,
                    comp_df.to_string(index=False),
                    "",
                    "Saved files",
                    base + "_comparison_results.xlsx",
                    base + "_comparison_table.png",
                    base + "_comparison_panel.png",
                    base + "_comparison_panel.pdf",
                ]
                self.output.insert(tk.END, "\n".join(lines))
                self.status_text.set("Multi-dataset analysis complete.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_text.set("Analysis failed.")


if __name__ == "__main__":
    root = tk.Tk()
    app = EnzymeKineticsDesktopApp(root)
    root.geometry("1280x900")
    root.mainloop()
