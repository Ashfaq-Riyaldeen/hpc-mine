"""
generate_charts.py
==================
Generates all performance charts for the HPC Analysis Report.
EC7207 / EE7218  High Performance Computing  –  Group 11
Topic: Pearson Correlation-Based Recommendation System

Charts produced (saved to  analysis_diagrams/charts/):
  1. speedup_comparison.png       – Speedup vs workers (OpenMP/Pthreads/MPI)
  2. execution_time_bar.png       – Wall-clock time comparison (all versions)
  3. efficiency_plot.png          – Parallel efficiency vs workers
  4. sim_pred_breakdown.png       – Sim + Pred time stacked per version
  5. mae_comparison.png           – MAE correctness across all versions
  6. hybrid_configs.png           – Hybrid P×T configuration comparison
  7. amdahls_law.png              – Theoretical vs actual speedup
  8. summary_dashboard.png        – All 4 key charts in one figure

HOW TO USE
----------
1. Run your benchmark:      bash results/run_benchmarks.sh
2. Fill in the timing numbers in the  ── YOUR DATA ──  section below.
3. Run:  python analysis_diagrams/generate_charts.py
4. Charts appear in  analysis_diagrams/charts/

Dependencies:
  pip install matplotlib numpy
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(SCRIPT_DIR, "charts")
os.makedirs(OUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# ── YOUR DATA ────────────────────────────────────────────────────────────────
# Replace the placeholder values below with numbers from your benchmark run.
# All times are in SECONDS.  Leave CUDA values as-is if not available.
# ─────────────────────────────────────────────────────────────────────────────

# Problem size used in benchmarks
N_USERS = 1000
N_ITEMS = 1000

# ── Serial baseline ───────────────────────────────────────────────────────────
SERIAL_SIM  = 1.1165     # [Timing] Similarity matrix  (seconds)
SERIAL_PRED = 5.2862     # [Timing] Prediction phase   (seconds)
SERIAL_TOT  = SERIAL_SIM + SERIAL_PRED
SERIAL_MAE  = 1.2574     # [Eval]   MAE on test set

# ── OpenMP (sim+pred total at each thread count) ──────────────────────────────
#   workers:       1       2       4       8
OPENMP_SIM  = [ 1.1470,  0.5768,  0.2855,  0.1474 ]
OPENMP_PRED = [ 5.5064,  2.7558,  1.3791,  0.6973 ]
OPENMP_MAE  = [ 1.2574,  1.2574,  1.2574,  1.2574 ]

# ── Pthreads (sim+pred total at each thread count) ───────────────────────────
#   workers:       1       2       4       8
PTHREADS_SIM  = [ 0.9404,  0.6929,  0.4066,  0.2177 ]
PTHREADS_PRED = [ 5.5292,  2.7646,  1.3816,  0.7047 ]
PTHREADS_MAE  = [ 1.2574,  1.2574,  1.2574,  1.2574 ]

# ── MPI (sim+pred total at each process count) ───────────────────────────────
#   workers:       1       2       4       8
MPI_SIM  = [ 2.3949,  1.1931,  0.5972,  0.3007 ]
MPI_PRED = [ 5.5561,  2.7599,  1.3848,  0.7083 ]
MPI_MAE  = [ 1.2574,  1.2574,  1.2574,  1.2574 ]

# ── CUDA (single GPU run) ────────────────────────────────────────────────────
CUDA_SIM   = None        # CUDA not available on this system
CUDA_PRED  = None
CUDA_MAE   = None

# ── Hybrid (P MPI ranks × T OpenMP threads, 8 total workers) ─────────────────
# Config:           2×4     4×2     8×1     1×8
HYBRID_LABELS  = ["2×4", "4×2", "8×1", "1×8"]
HYBRID_SIM     = [ 0.7651,  0.3802,  0.2333,  1.5311 ]
HYBRID_PRED    = [ 1.7681,  0.8893,  0.7008,  3.5192 ]
HYBRID_MAE     = [ 1.2574,  1.2574,  1.2574,  1.2574 ]

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED QUANTITIES
# ─────────────────────────────────────────────────────────────────────────────
WORKERS = [1, 2, 4, 8]

def total(sim_list, pred_list):
    return [s + p for s, p in zip(sim_list, pred_list)]

OPENMP_TOT   = total(OPENMP_SIM,   OPENMP_PRED)
PTHREADS_TOT = total(PTHREADS_SIM, PTHREADS_PRED)
MPI_TOT      = total(MPI_SIM,      MPI_PRED)
HYBRID_TOT   = total(HYBRID_SIM,   HYBRID_PRED)

def speedups(tot_list):
    return [SERIAL_TOT / t for t in tot_list]

def efficiency(su_list, workers):
    return [s / w for s, w in zip(su_list, workers)]

OMP_SU  = speedups(OPENMP_TOT)
PTH_SU  = speedups(PTHREADS_TOT)
MPI_SU  = speedups(MPI_TOT)

OMP_EFF  = efficiency(OMP_SU,  WORKERS)
PTH_EFF  = efficiency(PTH_SU,  WORKERS)
MPI_EFF  = efficiency(MPI_SU,  WORKERS)

CUDA_TOT = CUDA_SIM + CUDA_PRED if CUDA_SIM else None
CUDA_SU  = SERIAL_TOT / CUDA_TOT if CUDA_TOT else None

# ─────────────────────────────────────────────────────────────────────────────
# STYLE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "serial":   "#2c7bb6",
    "openmp":   "#1a9641",
    "pthreads": "#b8860b",
    "mpi":      "#d7191c",
    "cuda":     "#7600a3",
    "hybrid":   "#d45f00",
    "ideal":    "#aaaaaa",
}
MARKERS = {
    "openmp":   "o",
    "pthreads": "s",
    "mpi":      "^",
    "ideal":    "--",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
})

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Speedup vs Workers
# ─────────────────────────────────────────────────────────────────────────────
def chart_speedup():
    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Ideal linear speedup
    ax.plot(WORKERS, WORKERS, linestyle="--", color=COLORS["ideal"],
            linewidth=1.5, label="Ideal (linear)", zorder=1)

    ax.plot(WORKERS, OMP_SU,  marker="o", color=COLORS["openmp"],
            linewidth=2, markersize=7, label="OpenMP")
    ax.plot(WORKERS, PTH_SU,  marker="s", color=COLORS["pthreads"],
            linewidth=2, markersize=7, label="Pthreads")
    ax.plot(WORKERS, MPI_SU,  marker="^", color=COLORS["mpi"],
            linewidth=2, markersize=7, label="MPI")

    # CUDA single point
    if CUDA_SU:
        ax.scatter([8], [CUDA_SU], marker="*", s=200, color=COLORS["cuda"],
                   zorder=5, label=f"CUDA (×{CUDA_SU:.1f})")

    # Hybrid best config marker
    best_hyb_su = SERIAL_TOT / min(HYBRID_TOT)
    ax.scatter([8], [best_hyb_su], marker="D", s=100, color=COLORS["hybrid"],
               zorder=5, label=f"Hybrid best (×{best_hyb_su:.1f})")

    ax.set_xlabel("Number of Workers (threads / processes)")
    ax.set_ylabel("Speedup  $S(p) = T_{\\mathrm{serial}} / T_{\\mathrm{parallel}}$")
    ax.set_title(f"Speedup vs. Workers\n"
                 f"(Problem: {N_USERS}×{N_ITEMS}, "
                 f"$T_{{\\mathrm{{serial}}}}={SERIAL_TOT:.2f}$ s)")
    ax.set_xticks(WORKERS)
    ax.legend(loc="upper left")
    ax.set_xlim(0.5, 9)
    ax.set_ylim(0)

    fig.tight_layout()
    save(fig, "speedup_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Execution Time Bar Chart (best config per version)
# ─────────────────────────────────────────────────────────────────────────────
def chart_execution_time():
    labels = ["Serial\n(1 core)", "OpenMP\n(8T)", "Pthreads\n(8T)",
              "MPI\n(8P)", "Hybrid\n(2×4)", "CUDA\n(GPU)"]
    sim_times  = [SERIAL_SIM,  OPENMP_SIM[-1],  PTHREADS_SIM[-1],
                  MPI_SIM[-1], HYBRID_SIM[0],   CUDA_SIM if CUDA_SIM else 0]
    pred_times = [SERIAL_PRED, OPENMP_PRED[-1], PTHREADS_PRED[-1],
                  MPI_PRED[-1],HYBRID_PRED[0],  CUDA_PRED if CUDA_SIM else 0]

    x   = np.arange(len(labels))
    w   = 0.55
    col = [COLORS["serial"], COLORS["openmp"], COLORS["pthreads"],
           COLORS["mpi"], COLORS["hybrid"], COLORS["cuda"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars1 = ax.bar(x, sim_times,  w, label="Similarity phase",
                   color=col, alpha=0.85, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x, pred_times, w, bottom=sim_times,
                   label="Prediction phase",
                   color=col, alpha=0.45, edgecolor="white",
                   linewidth=0.8, hatch="///")

    # Value annotations
    for i, (s, p) in enumerate(zip(sim_times, pred_times)):
        total_t = s + p
        if total_t > 0:
            ax.text(x[i], total_t + 0.3, f"{total_t:.2f}s",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=col[i])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-Clock Time (seconds)")
    ax.set_title(f"Execution Time Comparison — Best Configuration per Version\n"
                 f"(Problem: {N_USERS}×{N_ITEMS})")
    ax.legend(loc="upper right")

    # Speedup annotations below bars
    serials = [SERIAL_TOT] + [OPENMP_TOT[-1], PTHREADS_TOT[-1],
                               MPI_TOT[-1], HYBRID_TOT[0],
                               CUDA_TOT if CUDA_TOT else 0]
    for i, t in enumerate(serials):
        if i == 0 or t == 0:
            continue
        su = SERIAL_TOT / t
        ax.text(x[i], -1.5, f"×{su:.1f}", ha="center",
                fontsize=9, color=col[i], fontweight="bold")
    ax.text(x[0], -1.5, "baseline", ha="center", fontsize=9, color=col[0])
    ax.annotate("speedup:", xy=(x[0]-0.65, -1.5), fontsize=8,
                color="gray", va="center")

    fig.tight_layout()
    save(fig, "execution_time_bar.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Parallel Efficiency
# ─────────────────────────────────────────────────────────────────────────────
def chart_efficiency():
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.axhline(1.0, linestyle="--", color=COLORS["ideal"],
               linewidth=1.5, label="Ideal efficiency (1.0)")
    ax.plot(WORKERS, OMP_EFF,  marker="o", color=COLORS["openmp"],
            linewidth=2, markersize=7, label="OpenMP")
    ax.plot(WORKERS, PTH_EFF,  marker="s", color=COLORS["pthreads"],
            linewidth=2, markersize=7, label="Pthreads")
    ax.plot(WORKERS, MPI_EFF,  marker="^", color=COLORS["mpi"],
            linewidth=2, markersize=7, label="MPI")

    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Efficiency  $E(p) = S(p) / p$")
    ax.set_title("Parallel Efficiency vs. Workers\n"
                 "(1.0 = perfect linear speedup)")
    ax.set_xticks(WORKERS)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")

    # Shade ideal region
    ax.axhspan(0.7, 1.15, alpha=0.05, color="green", label="_nolegend_")
    ax.text(8.1, 1.02, "ideal zone", fontsize=8, color="green", alpha=0.7)

    fig.tight_layout()
    save(fig, "efficiency_plot.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — Sim vs Pred Time Breakdown (Stacked)
# ─────────────────────────────────────────────────────────────────────────────
def chart_sim_pred_breakdown():
    versions = (
        ["Ser"] +
        [f"OMP\n{w}T" for w in WORKERS] +
        [f"PTH\n{w}T" for w in WORKERS] +
        [f"MPI\n{w}P" for w in WORKERS] +
        (["CUDA"] if CUDA_SIM else []) +
        [f"Hyb\n{l}" for l in HYBRID_LABELS]
    )
    sims = (
        [SERIAL_SIM] + OPENMP_SIM + PTHREADS_SIM + MPI_SIM +
        ([CUDA_SIM] if CUDA_SIM else []) + HYBRID_SIM
    )
    preds = (
        [SERIAL_PRED] + OPENMP_PRED + PTHREADS_PRED + MPI_PRED +
        ([CUDA_PRED] if CUDA_SIM else []) + HYBRID_PRED
    )

    x  = np.arange(len(versions))
    w  = 0.65

    # Colour each group
    def group_colors(n, base):
        return [base] * n

    bar_colors = (
        [COLORS["serial"]] +
        group_colors(4, COLORS["openmp"]) +
        group_colors(4, COLORS["pthreads"]) +
        group_colors(4, COLORS["mpi"]) +
        ([COLORS["cuda"]] if CUDA_SIM else []) +
        group_colors(4, COLORS["hybrid"])
    )

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x, sims,  w, color=bar_colors, alpha=0.9,
           label="Similarity phase", edgecolor="white", linewidth=0.6)
    ax.bar(x, preds, w, bottom=sims, color=bar_colors, alpha=0.45,
           label="Prediction phase", hatch="///",
           edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=8)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Similarity + Prediction Time Breakdown Across All Versions and Configurations")

    # Group separators
    separators = [0.5, 4.5, 8.5, 12.5]
    if CUDA_SIM:
        separators += [13.5]
    for sep in separators:
        ax.axvline(sep, color="gray", linewidth=0.8, linestyle=":")

    # Group labels
    group_x     = [2.0, 6.0, 10.0, 14.0 if not CUDA_SIM else 14.5]
    group_names = ["OpenMP", "Pthreads", "MPI",
                   "Hybrid" if not CUDA_SIM else "CUDA / Hybrid"]
    for gx, gn in zip(group_x, group_names):
        ax.text(gx, -2.0, gn, ha="center", fontsize=9,
                fontweight="bold", color="gray")

    # Legend patches
    p1 = mpatches.Patch(color="dimgray", alpha=0.9, label="Similarity phase")
    p2 = mpatches.Patch(color="dimgray", alpha=0.45, hatch="///", label="Prediction phase")
    ax.legend(handles=[p1, p2], loc="upper right")

    fig.tight_layout()
    save(fig, "sim_pred_breakdown.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — MAE Correctness Comparison
# ─────────────────────────────────────────────────────────────────────────────
def chart_mae():
    labels = ["Serial", "OpenMP\n(4T)", "Pthreads\n(4T)",
              "MPI\n(4P)", "Hybrid\n(2×4)"]
    maes   = [SERIAL_MAE, OPENMP_MAE[2], PTHREADS_MAE[2],
              MPI_MAE[2],  HYBRID_MAE[0]]
    cols   = [COLORS["serial"], COLORS["openmp"], COLORS["pthreads"],
              COLORS["mpi"], COLORS["hybrid"]]

    if CUDA_SIM:
        labels.append("CUDA")
        maes.append(CUDA_MAE)
        cols.append(COLORS["cuda"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, maes, 0.55, color=cols, alpha=0.85,
                  edgecolor="white", linewidth=0.8)

    # Tolerance band around serial MAE
    ax.axhline(SERIAL_MAE,        color=COLORS["serial"],
               linewidth=1.5, linestyle="-",  label="Serial MAE baseline")
    ax.axhline(SERIAL_MAE + 0.001, color="red",
               linewidth=1,   linestyle="--", label="±0.001 tolerance")
    ax.axhline(SERIAL_MAE - 0.001, color="red", linewidth=1, linestyle="--")
    ax.axhspan(SERIAL_MAE - 0.001, SERIAL_MAE + 0.001,
               alpha=0.10, color="green")

    # Value labels
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("MAE Correctness — All Parallel Versions vs. Serial Baseline\n"
                 "(must be within ±0.001 of serial)")

    # Set y-axis to zoom in
    y_center = SERIAL_MAE
    ax.set_ylim(y_center - 0.005, y_center + 0.008)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    save(fig, "mae_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 6 — Hybrid Configuration Comparison
# ─────────────────────────────────────────────────────────────────────────────
def chart_hybrid():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: timing bars
    ax = axes[0]
    x  = np.arange(len(HYBRID_LABELS))
    w  = 0.55
    hcols = [COLORS["hybrid"]] * 4
    ax.bar(x, HYBRID_SIM,  w, color=hcols, alpha=0.9,
           label="Similarity", edgecolor="white")
    ax.bar(x, HYBRID_PRED, w, bottom=HYBRID_SIM, color=hcols, alpha=0.45,
           label="Prediction", hatch="///", edgecolor="white")
    for i, (s, p) in enumerate(zip(HYBRID_SIM, HYBRID_PRED)):
        ax.text(x[i], s + p + 0.05, f"{s+p:.2f}s",
                ha="center", fontsize=9, fontweight="bold",
                color=COLORS["hybrid"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"P={l[0]}×T={l[2]}" for l in HYBRID_LABELS])
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Hybrid: Time per P×T Configuration\n(all = 8 total workers)")
    ax.legend()

    # Right: speedup bars
    ax2 = axes[1]
    hyb_su = [SERIAL_TOT / t for t in HYBRID_TOT]
    bars = ax2.bar(x, hyb_su, w, color=hcols, alpha=0.85, edgecolor="white")
    for bar, su in zip(bars, hyb_su):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 su + 0.05, f"×{su:.2f}",
                 ha="center", fontsize=9, fontweight="bold",
                 color=COLORS["hybrid"])
    # Reference lines
    ax2.axhline(OMP_SU[-1],  color=COLORS["openmp"],   linestyle="--",
                linewidth=1.5, label=f"OpenMP 8T (×{OMP_SU[-1]:.1f})")
    ax2.axhline(MPI_SU[-1],  color=COLORS["mpi"],      linestyle="-.",
                linewidth=1.5, label=f"MPI 8P (×{MPI_SU[-1]:.1f})")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"P={l[0]}×T={l[2]}" for l in HYBRID_LABELS])
    ax2.set_ylabel("Speedup")
    ax2.set_title("Hybrid: Speedup per P×T Configuration")
    ax2.legend(fontsize=9)

    fig.suptitle("Hybrid MPI+OpenMP — Configuration Analysis (8 total workers)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "hybrid_configs.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 7 — Amdahl's Law: Theoretical vs Actual
# ─────────────────────────────────────────────────────────────────────────────
def chart_amdahl():
    # Estimate serial fraction from timing:
    # f ≈ (Phase1 + Phase5) / Total ≈ 2-3%
    f_serial = 0.03
    p_range  = np.linspace(1, 8, 200)
    amdahl   = 1.0 / (f_serial + (1 - f_serial) / p_range)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(p_range, amdahl, color=COLORS["ideal"], linewidth=2,
            linestyle="--", label=f"Amdahl's Law (serial fraction={f_serial*100:.0f}%)")
    ax.plot(p_range, p_range, color="lightgray", linewidth=1.5,
            linestyle=":", label="Ideal linear")

    ax.plot(WORKERS, OMP_SU,  "o-", color=COLORS["openmp"],
            linewidth=2, markersize=7, label="OpenMP (actual)")
    ax.plot(WORKERS, PTH_SU,  "s-", color=COLORS["pthreads"],
            linewidth=2, markersize=7, label="Pthreads (actual)")
    ax.plot(WORKERS, MPI_SU,  "^-", color=COLORS["mpi"],
            linewidth=2, markersize=7, label="MPI (actual)")

    ax.set_xlabel("Number of Workers $p$")
    ax.set_ylabel("Speedup $S(p)$")
    ax.set_title("Amdahl's Law: Theoretical Limit vs. Actual Speedup\n"
                 f"(serial fraction $f \\approx {f_serial*100:.0f}\\%$, "
                 f"$S_{{\\max}} \\approx {1/f_serial:.0f}\\times$)")
    ax.set_xticks(WORKERS)
    ax.legend(loc="upper left")
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0)

    fig.tight_layout()
    save(fig, "amdahls_law.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 8 — Summary Dashboard (2×2 grid)
# ─────────────────────────────────────────────────────────────────────────────
def chart_dashboard():
    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Top-left: Speedup ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(WORKERS, WORKERS,  "--", color=COLORS["ideal"],
             linewidth=1.5, label="Ideal")
    ax1.plot(WORKERS, OMP_SU,  "o-", color=COLORS["openmp"],
             linewidth=2, markersize=6, label="OpenMP")
    ax1.plot(WORKERS, PTH_SU,  "s-", color=COLORS["pthreads"],
             linewidth=2, markersize=6, label="Pthreads")
    ax1.plot(WORKERS, MPI_SU,  "^-", color=COLORS["mpi"],
             linewidth=2, markersize=6, label="MPI")
    if CUDA_SU:
        ax1.scatter([8], [CUDA_SU], marker="*", s=150,
                    color=COLORS["cuda"], zorder=5, label=f"CUDA ×{CUDA_SU:.1f}")
    ax1.set_title("(a) Speedup vs Workers", fontweight="bold")
    ax1.set_xlabel("Workers"); ax1.set_ylabel("Speedup")
    ax1.set_xticks(WORKERS); ax1.legend(fontsize=8)

    # ── Top-right: Efficiency ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axhline(1.0, linestyle="--", color=COLORS["ideal"], linewidth=1.5)
    ax2.plot(WORKERS, OMP_EFF,  "o-", color=COLORS["openmp"],
             linewidth=2, markersize=6, label="OpenMP")
    ax2.plot(WORKERS, PTH_EFF,  "s-", color=COLORS["pthreads"],
             linewidth=2, markersize=6, label="Pthreads")
    ax2.plot(WORKERS, MPI_EFF,  "^-", color=COLORS["mpi"],
             linewidth=2, markersize=6, label="MPI")
    ax2.set_title("(b) Parallel Efficiency", fontweight="bold")
    ax2.set_xlabel("Workers"); ax2.set_ylabel("Efficiency E(p)")
    ax2.set_xticks(WORKERS); ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=8)

    # ── Bottom-left: Execution time ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    labels_dash = ["Serial", "OMP\n8T", "PTH\n8T", "MPI\n8P", "Hyb\n2×4"]
    times_dash  = [SERIAL_TOT, OPENMP_TOT[-1], PTHREADS_TOT[-1],
                   MPI_TOT[-1], HYBRID_TOT[0]]
    cols_dash   = [COLORS["serial"], COLORS["openmp"], COLORS["pthreads"],
                   COLORS["mpi"], COLORS["hybrid"]]
    if CUDA_TOT:
        labels_dash.append("CUDA")
        times_dash.append(CUDA_TOT)
        cols_dash.append(COLORS["cuda"])
    bars3 = ax3.bar(labels_dash, times_dash, color=cols_dash, alpha=0.85,
                    edgecolor="white", linewidth=0.8)
    for bar, t in zip(bars3, times_dash):
        ax3.text(bar.get_x() + bar.get_width() / 2, t + 0.2,
                 f"{t:.1f}s", ha="center", fontsize=8, fontweight="bold")
    ax3.set_title("(c) Best-Config Execution Time", fontweight="bold")
    ax3.set_ylabel("Time (s)")

    # ── Bottom-right: MAE ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    mae_labels = ["Serial", "OMP\n4T", "PTH\n4T", "MPI\n4P", "Hyb\n2×4"]
    mae_vals   = [SERIAL_MAE, OPENMP_MAE[2], PTHREADS_MAE[2],
                  MPI_MAE[2],  HYBRID_MAE[0]]
    mae_cols   = [COLORS["serial"], COLORS["openmp"], COLORS["pthreads"],
                  COLORS["mpi"], COLORS["hybrid"]]
    if CUDA_SIM:
        mae_labels.append("CUDA")
        mae_vals.append(CUDA_MAE)
        mae_cols.append(COLORS["cuda"])
    ax4.bar(mae_labels, mae_vals, color=mae_cols, alpha=0.85,
            edgecolor="white", linewidth=0.8)
    ax4.axhline(SERIAL_MAE, linestyle="-",  color=COLORS["serial"],
                linewidth=1.5, label="Serial baseline")
    ax4.axhspan(SERIAL_MAE - 0.001, SERIAL_MAE + 0.001,
                alpha=0.12, color="green", label="±0.001 tolerance")
    ax4.set_title("(d) MAE Correctness", fontweight="bold")
    ax4.set_ylabel("MAE")
    y_c = SERIAL_MAE
    ax4.set_ylim(y_c - 0.006, y_c + 0.01)
    ax4.legend(fontsize=8)

    fig.suptitle(
        f"HPC Performance Summary — Pearson Correlation Recommender\n"
        f"Group 11  |  Problem: {N_USERS}×{N_ITEMS}  |  "
        f"Serial baseline: {SERIAL_TOT:.2f} s",
        fontsize=13, fontweight="bold"
    )
    save(fig, "summary_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  HPC Chart Generator — Group 11")
    print(f"  Output: {OUT_DIR}")
    print("=" * 55)

    print("\n[1/8] Speedup comparison ...")
    chart_speedup()

    print("[2/8] Execution time bar chart ...")
    chart_execution_time()

    print("[3/8] Parallel efficiency ...")
    chart_efficiency()

    print("[4/8] Sim + Pred breakdown ...")
    chart_sim_pred_breakdown()

    print("[5/8] MAE correctness ...")
    chart_mae()

    print("[6/8] Hybrid configuration analysis ...")
    chart_hybrid()

    print("[7/8] Amdahl's Law ...")
    chart_amdahl()

    print("[8/8] Summary dashboard ...")
    chart_dashboard()

    print("\nAll 8 charts saved to:", OUT_DIR)
    print("\nFiles:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")
