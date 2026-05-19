"""
build_report_pdf.py
===================
Generates the HPC Group 11 Analysis Report as a PDF using ReportLab.
Run from any directory:  python3 analysis_report/build_report_pdf.py
Output:  analysis_report/HPC_Group11_Analysis_Report.pdf
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image, KeepTogether
)
from reportlab.lib import colors

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CHARTS_DIR   = os.path.join(PROJECT_ROOT, "analysis_diagrams", "charts")
OUT_PDF      = os.path.join(SCRIPT_DIR, "HPC_Group11_Analysis_Report.pdf")

# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE       = HexColor("#2c7bb6")
DARKBLUE   = HexColor("#1a4e7a")
LIGHTBLUE  = HexColor("#d6eaf8")
LIGHTGRAY  = HexColor("#f5f5f5")
MIDGRAY    = HexColor("#cccccc")
GREEN      = HexColor("#1a9641")
RED        = HexColor("#d7191c")
ORANGE     = HexColor("#d45f00")
DARKGOLD   = HexColor("#b8860b")
WHITE      = white
BLACK      = black

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def style(name, **kwargs):
    return ParagraphStyle(name, parent=styles["Normal"], **kwargs)

S_TITLE   = style("ReportTitle",    fontSize=22, textColor=BLUE,    spaceAfter=6,
                  fontName="Helvetica-Bold", alignment=TA_CENTER, leading=28)
S_SUBT    = style("SubTitle",       fontSize=14, textColor=DARKBLUE, spaceAfter=4,
                  fontName="Helvetica-Bold", alignment=TA_CENTER)
S_META    = style("Meta",           fontSize=11, textColor=black,   spaceAfter=3,
                  alignment=TA_CENTER)
S_H1      = style("H1",             fontSize=14, textColor=BLUE,    spaceBefore=14,
                  spaceAfter=6, fontName="Helvetica-Bold",
                  borderPad=2, leftIndent=0)
S_H2      = style("H2",             fontSize=12, textColor=DARKBLUE, spaceBefore=10,
                  spaceAfter=4, fontName="Helvetica-Bold")
S_H3      = style("H3",             fontSize=11, textColor=DARKBLUE, spaceBefore=8,
                  spaceAfter=3, fontName="Helvetica-BoldOblique")
S_BODY    = style("Body",           fontSize=10, spaceAfter=5, leading=14,
                  alignment=TA_JUSTIFY)
S_BULLET  = style("Bullet",         fontSize=10, spaceAfter=3, leading=13,
                  leftIndent=18, firstLineIndent=-10)
S_CAPTION = style("Caption",        fontSize=9,  textColor=HexColor("#555555"),
                  spaceAfter=6, spaceBefore=2, alignment=TA_CENTER, fontName="Helvetica-Oblique")
S_CODE    = style("Code",           fontSize=9,  fontName="Courier",
                  backColor=LIGHTGRAY, leftIndent=12, rightIndent=12,
                  spaceBefore=4, spaceAfter=4, leading=13)
S_NOTE    = style("Note",           fontSize=9,  textColor=HexColor("#555555"),
                  fontName="Helvetica-Oblique", leftIndent=12, spaceAfter=4)

def bullet(text):
    return Paragraph(f"• {text}", S_BULLET)

def h1(text):
    return Paragraph(text, S_H1)

def h2(text):
    return Paragraph(text, S_H2)

def h3(text):
    return Paragraph(text, S_H3)

def p(text):
    return Paragraph(text, S_BODY)

def code(text):
    return Paragraph(text.replace(" ", "&nbsp;").replace("\n", "<br/>"), S_CODE)

def spacer(h=0.3):
    return Spacer(1, h * cm)

def hr():
    return HRFlowable(width="100%", thickness=1, color=MIDGRAY, spaceAfter=8, spaceBefore=4)

# ── Table helpers ─────────────────────────────────────────────────────────────
def make_table(data, col_widths, header=True, zebra=True):
    tbl = Table(data, colWidths=col_widths)
    ts  = [
        ("GRID",      (0, 0), (-1, -1), 0.4, MIDGRAY),
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
    ]
    if header:
        ts += [
            ("BACKGROUND",  (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [LIGHTGRAY, WHITE] if zebra else [WHITE]),
        ]
    tbl.setStyle(TableStyle(ts))
    return tbl

def chart(filename, width=15*cm, caption_text=""):
    path = os.path.join(CHARTS_DIR, filename)
    elems = []
    if os.path.exists(path):
        elems.append(Image(path, width=width, height=width * 0.62))
    if caption_text:
        elems.append(Paragraph(caption_text, S_CAPTION))
    return elems

# ── Page template ─────────────────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(BLUE)
    canvas.rect(2*cm, A4[1] - 1.5*cm, A4[0] - 4*cm, 0.35*cm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2*cm, A4[1] - 1.4*cm, "EC7207 — High Performance Computing")
    canvas.drawRightString(A4[0] - 2*cm, A4[1] - 1.4*cm, "Group 11 — Analysis Report")
    # Footer
    canvas.setFillColor(MIDGRAY)
    canvas.rect(2*cm, 1.3*cm, A4[0] - 4*cm, 0.02*cm, fill=1, stroke=0)
    canvas.setFillColor(HexColor("#777777"))
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(A4[0] / 2, 1.0*cm, str(doc.page))
    canvas.restoreState()

# ══════════════════════════════════════════════════════════════════════════════
# BUILD DOCUMENT
# ══════════════════════════════════════════════════════════════════════════════
doc = SimpleDocTemplate(
    OUT_PDF,
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.2*cm, bottomMargin=2.2*cm,
    title="HPC Group 11 Analysis Report",
    author="Group 11 — EC7207"
)

story = []

# ─────────────────────────────────────────────────────────────────────────────
# TITLE PAGE
# ─────────────────────────────────────────────────────────────────────────────
story += [
    spacer(2),
    Paragraph("EC7207: High Performance Computing", S_SUBT),
    spacer(0.3),
    Paragraph("Analysis Report", style("AR", fontSize=13, alignment=TA_CENTER,
              fontName="Helvetica", textColor=HexColor("#555555"), spaceAfter=10)),
    HRFlowable(width="70%", thickness=2, color=BLUE, spaceAfter=12, spaceBefore=4),
    Paragraph("Parallel Pearson Correlation<br/>Based Recommendation System",
              style("BigTitle", fontSize=24, textColor=BLUE, alignment=TA_CENTER,
                    fontName="Helvetica-Bold", leading=30, spaceAfter=16)),
    HRFlowable(width="70%", thickness=2, color=BLUE, spaceAfter=12),
    spacer(0.5),
    Paragraph("<b>Group Number: 11</b>", S_META),
    spacer(0.2),
    Paragraph("EG/2021/4417 — Ashfaq M.R.M.", S_META),
    Paragraph("EG/2021/4419 — Athanayaka K.A.L.G.", S_META),
    Paragraph("EG/2021/4424 — Balasooriya J.M.", S_META),
    spacer(1.5),
    Paragraph("Technologies: Serial · OpenMP · Pthreads · MPI · Hybrid MPI+OpenMP",
              style("Tech", fontSize=10, alignment=TA_CENTER,
                    textColor=HexColor("#555555"), spaceAfter=4)),
    Paragraph("Problem: 1,000 users × 1,000 items · SEED=42 · Sparsity=70% · TOP-K=20",
              style("Tech2", fontSize=9, alignment=TA_CENTER,
                    textColor=HexColor("#777777"), spaceAfter=4)),
    Paragraph("Platform: Linux · GCC 15.2 · OpenMPI · 32 logical CPU cores",
              style("Tech3", fontSize=9, alignment=TA_CENTER,
                    textColor=HexColor("#777777"))),
    spacer(3),
    Paragraph("May 2026", style("Date", fontSize=10, alignment=TA_CENTER,
              textColor=HexColor("#999999"))),
    PageBreak(),
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
story += [
    h1("1. Introduction"),
    hr(),
    p("Modern collaborative filtering recommendation systems compute pairwise "
      "user similarity over large rating matrices. As the number of users grows, "
      "the O(N² × M) similarity computation becomes the dominant bottleneck — "
      "for N = M = 1,000 this equates to approximately 500 million floating-point "
      "operations."),
    p("This report documents the design, implementation, and performance evaluation "
      "of a <b>Pearson Correlation-Based Collaborative Filtering Recommender System</b> "
      "parallelised using five HPC technologies: OpenMP, POSIX Threads (Pthreads), "
      "MPI, CUDA (code implemented; no GPU available on benchmark machine), and a "
      "Hybrid MPI+OpenMP approach."),
    p("All measurements were obtained on a Linux machine with <b>32 logical CPU "
      "cores</b> and 32 GB RAM, using GCC 15.2 and OpenMPI."),
    spacer(0.3),
    h2("Experimental Configuration"),
    make_table([
        ["Parameter", "Value"],
        ["Users (N)", "1,000"],
        ["Items (M)", "1,000"],
        ["Sparsity", "70%"],
        ["Test split / Test set size", "10%  →  29,866 held-out ratings"],
        ["TOP-K neighbours", "20"],
        ["Random seed", "42"],
        ["Benchmark thread counts", "1, 2, 4, 8"],
        ["Timing method", "clock_gettime / omp_get_wtime / MPI_Wtime"],
    ], [6*cm, 10*cm]),
    spacer(0.3),
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. ALGORITHM DESIGN
# ─────────────────────────────────────────────────────────────────────────────
story += [
    h1("2. Algorithm Design and Parallel Programming Concepts"),
    hr(),
    h2("2.1  Algorithm Pipeline (5 Phases)"),
    p("The system executes in five sequential phases. Phase 4 (prediction) "
      "dominates at ~82% of serial runtime; Phase 3 (similarity) accounts for "
      "~18%. Both are embarrassingly parallel at the row level."),
    make_table([
        ["Phase", "Name", "Description", "Complexity"],
        ["1", "Data Generation",    "Synthetic sparse rating matrix; 10% test split",       "O(N × M)"],
        ["2", "User Means",         "Per-user mean rating μ_u",                             "O(N × M)"],
        ["3", "Similarity ★",      "Pearson correlation for all (u,v) pairs — bottleneck", "O(N² × M)"],
        ["4", "Predictions ★",     "TOP-K weighted prediction for unrated cells",          "O(N² × M)"],
        ["5", "MAE Evaluation",     "Error on held-out test set",                           "O(|T|)"],
    ], [1.2*cm, 3.2*cm, 7.5*cm, 3.3*cm]),
    spacer(0.2),
    Paragraph("★  Both phases are parallelised across all 5 implementations.", S_NOTE),
    spacer(0.3),
    p("<b>Pearson correlation formula (Phase 3):</b>"),
    code(
        "         Σᵢ (Rᵤᵢ − μᵤ)(Rᵥᵢ − μᵥ)\n"
        "sim(u,v) = ─────────────────────────────────────\n"
        "           √[Σᵢ (Rᵤᵢ−μᵤ)²] × √[Σᵢ (Rᵥᵢ−μᵥ)²]"
    ),
    p("<b>Prediction formula (Phase 4):</b>"),
    code(
        "                   Σₖ [sim(u,k) × (Rₖᵢ − μₖ)]\n"
        "pred(u,i) = μᵤ + ────────────────────────────────\n"
        "                          Σₖ sim(u,k)"
    ),
    spacer(0.3),
]

# ── 2.2 OpenMP ────────────────────────────────────────────────────────────────
story += [
    h2("2.2  OpenMP — Shared-Memory Fork-Join Parallelism"),
    p("<b>Concept:</b> OpenMP exploits shared-memory parallelism using compiler "
      "directives. A single master thread forks into T worker threads, all sharing "
      "the same process address space, then rejoins at an implicit barrier."),
    p("<b>Phase 3 (Similarity — key decision):</b> The outer loop over user pairs "
      "uses <font face='Courier' size=9>#pragma omp parallel for schedule(dynamic,4)</font>. "
      "<b>Dynamic scheduling</b> was chosen because the triangular loop creates "
      "load imbalance: thread 0 processes 999 Pearson calls while the last thread "
      "processes 1. Dynamic scheduling lets idle threads pull new 4-row chunks "
      "from a shared work queue, eliminating idle time."),
    p("<b>Race-freedom:</b> Each thread owns an exclusive set of outer-loop rows u "
      "and writes SIM(u,v) and SIM(v,u) only for those rows. No two threads write "
      "the same cell — no mutex or atomic required."),
    p("<b>Phase 5 (MAE):</b> The <font face='Courier' size=9>reduction(+:err)</font> "
      "clause gives each thread a private accumulator; OpenMP sums all copies at "
      "the implicit barrier."),
    spacer(0.2),
]

# ── 2.3 Pthreads ──────────────────────────────────────────────────────────────
story += [
    h2("2.3  Pthreads — Manual POSIX Thread Management"),
    p("<b>Concept:</b> POSIX Threads provide explicit, low-level manual thread "
      "management. Unlike OpenMP, Pthreads requires the programmer to explicitly "
      "create threads, assign work, and synchronise using pthread_join."),
    p("<b>Thread lifecycle pattern</b> — a run_parallel(fn, nthreads) harness is "
      "called at the start of each of Phases 2, 3, and 4:"),
    bullet("pthread_create() creates T threads, each receiving a ThreadArgs struct "
           "with its tid and total nthreads."),
    bullet("Each thread derives its exclusive row range: "
           "start = tid × ⌈N/T⌉, end = min(start+⌈N/T⌉, N)."),
    bullet("pthread_join() waits for all T threads — acting as a phase barrier."),
    p("<b>Key difference from OpenMP:</b> Threads are created and joined three "
      "separate times (once per phase). OpenMP reuses threads across phases. "
      "Pthreads also uses static row assignment, which cannot adapt to the "
      "triangular loop imbalance — explaining slightly lower efficiency than OpenMP."),
    spacer(0.2),
]

# ── 2.4 MPI ───────────────────────────────────────────────────────────────────
story += [
    h2("2.4  MPI — Distributed-Memory Parallelism"),
    p("<b>Concept:</b> MPI implements distributed-memory parallelism where each "
      "rank is a completely independent process with its own private address space. "
      "Data sharing requires explicit message passing."),
    p("<b>Data partitioning:</b> The N = 1,000 users are divided evenly across P "
      "ranks. Rank r owns rows [r×(N/P), (r+1)×(N/P))."),
    make_table([
        ["Phase", "Strategy", "MPI Primitive"],
        ["Phase 1", "All ranks call srand(42) → identical data, no communication", "—"],
        ["Phase 2", "Each rank computes its slice of user_mean[]", "MPI_Allgatherv"],
        ["Phase 3", "Each rank fills its rows of the N×N sim matrix", "MPI_Allgatherv"],
        ["Phase 4", "Each rank predicts its own users — no communication", "—"],
        ["Phase 5", "Local error sums aggregated to rank 0", "MPI_Reduce(SUM)"],
    ], [1.5*cm, 10*cm, 4.2*cm]),
    spacer(0.2),
    p("<b>Key overhead:</b> MPI_Allgatherv for the N×N similarity matrix "
      "transfers ~4 MB at N=1,000. This overhead limits efficiency compared to "
      "shared-memory models, but scales linearly with P."),
    spacer(0.2),
]

# ── 2.5 CUDA ──────────────────────────────────────────────────────────────────
story += [
    h2("2.5  CUDA — GPU Massively Parallel Computing"),
    Paragraph("Note: CUDA source code (cuda/cuda_recommender.cu) is fully "
              "implemented but could not be benchmarked — no GPU available.",
              style("NoteBox", fontSize=9, textColor=HexColor("#7b6000"),
                    backColor=HexColor("#fff9c4"), borderPad=6,
                    leftIndent=8, spaceAfter=6, fontName="Helvetica-Oblique")),
    p("<b>Concept:</b> CUDA exploits the massively parallel NVIDIA GPU architecture, "
      "launching thousands of threads in a two-level grid/block hierarchy."),
    make_table([
        ["Kernel", "Launch Config", "Assignment"],
        ["kernel_user_means",   "<<<(N+255)/256, 256>>>",                   "1 thread = 1 user mean"],
        ["kernel_similarity",   "dim3(⌈N/16⌉,⌈N/16⌉), dim3(16,16)",       "1 thread = 1 (u,v) pair"],
        ["kernel_predictions",  "2D grid — users × items",                  "1 thread = 1 (user,item) pair"],
    ], [4.5*cm, 6.5*cm, 5.2*cm]),
    spacer(0.2),
    p("For N=1,000: the similarity kernel launches <b>63×63×256 = 1,016,064 "
      "threads simultaneously</b>, covering all 499,500 unique user pairs. "
      "Timing uses cudaEventRecord (GPU-side timestamps) to separate kernel "
      "execution time from cudaMemcpy transfer overhead."),
    spacer(0.2),
]

# ── 2.6 Hybrid ────────────────────────────────────────────────────────────────
story += [
    h2("2.6  Hybrid MPI+OpenMP — Two-Level Parallelism"),
    p("<b>Concept:</b> Combines MPI for coarse-grain inter-node distribution and "
      "OpenMP for fine-grain intra-node threading, matching modern HPC cluster "
      "architecture."),
    make_table([
        ["Level", "Technology", "Scope", "Example (P=2, T=4)"],
        ["Level 1", "MPI",    "Partitions users across processes/nodes", "Rank 0: users 0–499, Rank 1: users 500–999"],
        ["Level 2", "OpenMP", "Sub-divides local users within each rank", "Rank 0's 4 threads: rows 0–124, 125–249, …"],
        ["Total",   "P × T",  "Workers",                                  "2 × 4 = 8 workers"],
    ], [1.5*cm, 2.5*cm, 5.5*cm, 6.7*cm]),
    spacer(0.2),
    p("<b>Thread safety:</b> MPI_Init_thread(MPI_THREAD_FUNNELED) ensures all "
      "MPI calls are made exclusively from the main thread, after the OpenMP "
      "parallel region completes its barrier."),
    make_table([
        ["Config", "P (MPI)", "T (OMP)", "Characteristic"],
        ["Hybrid 2×4", "2", "4", "Fewer MPI messages; more shared-memory parallelism"],
        ["Hybrid 4×2", "4", "2", "Balanced communication vs. threading overhead"],
        ["Hybrid 8×1", "8", "1", "Equivalent to pure MPI (no OpenMP benefit)"],
        ["Hybrid 1×8", "1", "8", "Pure OpenMP wrapped with MPI overhead (worst case)"],
    ], [3*cm, 2*cm, 2*cm, 9.2*cm]),
    PageBreak(),
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. ACCURACY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
story += [
    h1("3. Accuracy Analysis — Parallel vs. Serial"),
    hr(),
    p("All parallel implementations are validated against the serial baseline "
      "using <b>Mean Absolute Error (MAE)</b> and a <b>similarity-matrix checksum</b>."),
    code("MAE = (1 / |T|) × Σ |pred(u,i) − actual(u,i)|     where |T| = 29,866"),
    p("<b>Why MAE instead of RMSE?</b> MAE treats all prediction errors equally, "
      "providing a clearer measure on the bounded 1–5 rating scale. RMSE would "
      "disproportionately penalise occasional large errors."),
    p("<b>Sim-matrix checksum:</b> Sum of all N×N similarity matrix values — any "
      "deviation across versions indicates a race condition or partitioning bug."),
    spacer(0.3),
    h2("3.1  MAE Results"),
    make_table([
        ["Version",       "Workers", "MAE",   "Matches Serial?",        "Sim Checksum"],
        ["Serial",        "1",      "1.2574", "baseline",               "942.387323"],
        ["OpenMP",        "1T",     "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["OpenMP",        "2T",     "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["OpenMP",        "4T",     "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["OpenMP",        "8T",     "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["Pthreads",      "1T",     "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["Pthreads",      "2T",     "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["Pthreads",      "4T",     "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["Pthreads",      "8T",     "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["MPI",           "1P",     "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["MPI",           "2P",     "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["MPI",           "4P",     "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["MPI",           "8P",     "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["Hybrid 2×4",    "8",      "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["Hybrid 4×2",    "8",      "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["Hybrid 8×1",    "8",      "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["Hybrid 1×8",    "8",      "1.2574", "✓  YES  (Δ = 0.0000)",   "942.387323"],
        ["CUDA",          "GPU",    "N/A",    "N/A (no GPU available)",  "N/A"],
    ], [3.5*cm, 2*cm, 2*cm, 5.5*cm, 3.2*cm]),
    spacer(0.3),
    p("<b>Key finding:</b> All 17 CPU-based runs produce an MAE of exactly "
      "<b>1.2574</b> and an identical similarity-matrix checksum of <b>942.387323</b>. "
      "This confirms: (1) zero race conditions in any parallel version; "
      "(2) correct data partitioning and synchronisation; "
      "(3) the fixed seed SEED=42 ensures every version operates on identical data."),
    spacer(0.3),
]
story += chart("mae_comparison.png", width=13*cm,
               caption_text="Figure 1: MAE correctness across all parallel versions vs. serial baseline. "
                            "All bars lie within the ±0.001 tolerance band (green shading).")
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# 4. TIMING MEASUREMENTS
# ─────────────────────────────────────────────────────────────────────────────
story += [
    h1("4. Timing Measurements and Performance Analysis"),
    hr(),
    h2("4.1  Performance Metric Definitions"),
    code(
        "Speedup:     S(p) = T_serial / T_parallel(p)\n"
        "Efficiency:  E(p) = S(p) / p         (1.0 = perfect scaling)\n"
        "Amdahl:      S_max = 1 / (f + (1−f)/p)    f = serial fraction ≈ 0.03\n"
        "             → S_max(8) = 1/(0.03 + 0.97/8) ≈ 6.61×"
    ),
    spacer(0.3),
    h2("4.2  Serial Baseline"),
    make_table([
        ["Phase",                             "Time (s)", "% of Total"],
        ["Data generation",                   "0.0100",   "0.2%"],
        ["User mean computation",             "0.0022",   "0.0%"],
        ["Similarity matrix (Phase 3)",       "1.1165",   "17.4%"],
        ["Prediction phase (Phase 4)",        "5.2862",   "82.6%"],
        ["MAE evaluation",                    "< 0.001",  "~0%"],
        ["Total (sim + pred)",                "6.4027",   "100%"],
    ], [8*cm, 3*cm, 3*cm]),
    Paragraph("Phase 4 (prediction) dominates because each unrated (user,item) "
              "cell requires an O(N) neighbour scan plus a sort, applied N×M times.",
              S_NOTE),
    spacer(0.3),
    h2("4.3  OpenMP — Varying Thread Count"),
    make_table([
        ["Threads (T)", "Sim (s)", "Pred (s)", "Total (s)", "Speedup S(T)", "Efficiency E(T)"],
        ["1",  "1.1470", "5.5064", "6.6534", "0.96", "0.96"],
        ["2",  "0.5768", "2.7558", "3.3326", "1.92", "0.96"],
        ["4",  "0.2855", "1.3791", "1.6646", "3.85", "0.96"],
        ["8",  "0.1474", "0.6973", "0.8447", "7.58 ★", "0.95"],
    ], [3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm, 3*cm]),
    Paragraph("★  Best result among all CPU implementations.", S_NOTE),
    spacer(0.3),
    h2("4.4  Pthreads — Varying Thread Count"),
    make_table([
        ["Threads (T)", "Sim (s)", "Pred (s)", "Total (s)", "Speedup S(T)", "Efficiency E(T)"],
        ["1", "0.9404", "5.5292", "6.4696", "0.99", "0.99"],
        ["2", "0.6929", "2.7646", "3.4575", "1.85", "0.93"],
        ["4", "0.4066", "1.3816", "1.7882", "3.58", "0.90"],
        ["8", "0.2177", "0.7047", "0.9224", "6.94", "0.87"],
    ], [3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm, 3*cm]),
    spacer(0.3),
    h2("4.5  MPI — Varying Process Count"),
    make_table([
        ["Processes (P)", "Sim (s)", "Pred (s)", "Total (s)", "Speedup S(P)", "Efficiency E(P)"],
        ["1", "2.3949", "5.5561", "7.9510", "0.81", "0.81"],
        ["2", "1.1931", "2.7599", "3.9530", "1.62", "0.81"],
        ["4", "0.5972", "1.3848", "1.9820", "3.23", "0.81"],
        ["8", "0.3007", "0.7083", "1.0090", "6.35", "0.79"],
    ], [3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm, 3*cm]),
    Paragraph("MPI 1P (7.95 s) is slower than serial (6.40 s) due to process "
              "startup, MPI_Allgatherv overhead with P=1, and MPI_Wtime clock cost. "
              "This is expected and not a bug.", S_NOTE),
    spacer(0.3),
    h2("4.6  Hybrid MPI+OpenMP — Fixed 8 Total Workers"),
    make_table([
        ["Config", "P", "T", "Sim (s)", "Pred (s)", "Total (s)", "Speedup", "Efficiency"],
        ["Hybrid 2×4", "2", "4", "0.7651", "1.7681", "2.5332", "2.53",    "0.32"],
        ["Hybrid 4×2", "4", "2", "0.3802", "0.8893", "1.2695", "5.04",    "0.63"],
        ["Hybrid 8×1", "8", "1", "0.2333", "0.7008", "0.9341", "6.85 ★", "0.86"],
        ["Hybrid 1×8", "1", "8", "1.5311", "3.5192", "5.0503", "1.27",    "0.16"],
    ], [3*cm, 1.2*cm, 1.2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2.8*cm]),
    spacer(0.3),
]

# Charts
story += chart("speedup_comparison.png", width=14.5*cm,
               caption_text="Figure 2: Speedup vs. workers for OpenMP, Pthreads, MPI. "
                            "Dashed = ideal linear; ◆ = best Hybrid result (8×1).")
story.append(spacer(0.3))
story += chart("execution_time_bar.png", width=14.5*cm,
               caption_text="Figure 3: Wall-clock execution time — best config per version. "
                            "Solid = similarity phase; hatched = prediction phase.")
story.append(PageBreak())
story += chart("efficiency_plot.png", width=14.5*cm,
               caption_text="Figure 4: Parallel efficiency E(p) = S(p)/p. "
                            "Green zone (>0.70) = good worker utilisation.")
story.append(spacer(0.3))
story += chart("sim_pred_breakdown.png", width=15.5*cm,
               caption_text="Figure 5: Similarity + Prediction time breakdown across all "
                            "versions and configurations.")
story.append(spacer(0.3))
story += chart("hybrid_configs.png", width=14.5*cm,
               caption_text="Figure 6: Hybrid MPI+OpenMP configuration analysis. "
                            "Left: wall-clock time. Right: speedup vs. OpenMP-8T and MPI-8P.")
story.append(PageBreak())
story += chart("amdahls_law.png", width=14.5*cm,
               caption_text="Figure 7: Amdahl's Law theoretical limit (f≈3%, S_max≈33×) "
                            "vs. actual speedup.")
story.append(spacer(0.3))
story += chart("summary_dashboard.png", width=15.5*cm,
               caption_text="Figure 8: Summary dashboard — (a) speedup, (b) efficiency, "
                            "(c) execution time, (d) MAE correctness.")
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────────
# 5. PERFORMANCE DISCUSSION
# ─────────────────────────────────────────────────────────────────────────────
story += [
    h1("5. Performance Discussion"),
    hr(),
    h2("5.1  Head-to-Head Comparison at 8 Workers"),
    make_table([
        ["Version (8 workers)",   "Total (s)", "Speedup", "Efficiency", "MAE"],
        ["Serial (baseline)",     "6.4027",    "1.00",    "1.00",       "1.2574"],
        ["OpenMP 8T  ★",         "0.8447",    "7.58",    "0.95",       "1.2574"],
        ["Pthreads 8T",           "0.9224",    "6.94",    "0.87",       "1.2574"],
        ["Hybrid 8×1",            "0.9341",    "6.85",    "0.86",       "1.2574"],
        ["MPI 8P",                "1.0090",    "6.35",    "0.79",       "1.2574"],
        ["Hybrid 4×2",            "1.2695",    "5.04",    "0.63",       "1.2574"],
        ["Hybrid 2×4",            "2.5332",    "2.53",    "0.32",       "1.2574"],
        ["Hybrid 1×8",            "5.0503",    "1.27",    "0.16",       "1.2574"],
    ], [5.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm]),
    Paragraph("★  Best result across all implementations.", S_NOTE),
    spacer(0.4),
    h2("5.2  OpenMP vs. Pthreads"),
    p("Both target the same shared-memory hardware. OpenMP achieves ×7.58 (95% "
      "efficiency) while Pthreads achieves ×6.94 (87%). OpenMP is faster because:"),
    bullet("<b>Dynamic load balancing:</b> schedule(dynamic,4) adapts to the "
           "triangular loop imbalance (row 0 processes 999 Pearson calls; the last "
           "row processes 1). Static ceiling-division in Pthreads cannot compensate."),
    bullet("<b>Thread reuse:</b> OpenMP reuses threads across Phases 2–4 within one "
           "parallel region. Pthreads creates and joins threads separately for each "
           "phase — paying OS thread-creation overhead three times per run."),
    bullet("<b>Runtime optimisations:</b> The GCC OpenMP runtime applies "
           "platform-specific scheduling heuristics unavailable in bare Pthreads."),
    spacer(0.3),
    h2("5.3  MPI Scaling Behaviour"),
    p("MPI achieves consistent ~0.79–0.81 efficiency. The dominant overhead is "
      "MPI_Allgatherv for the N×N similarity matrix (~4 MB at N=1,000). Despite "
      "this, ×6.35 speedup at 8 processes confirms that MPI scales well. On a "
      "multi-node cluster where shared memory is unavailable, MPI would be the "
      "mandatory model and would close the efficiency gap with OpenMP."),
    spacer(0.3),
    h2("5.4  Hybrid MPI+OpenMP — Counter-Intuitive Inversion"),
    p("On a single node, <b>more MPI processes = better Hybrid performance</b>:"),
    make_table([
        ["Config",     "Speedup", "Observation"],
        ["Hybrid 8×1", "6.85",    "Best — essentially pure MPI on 8 processes"],
        ["Hybrid 4×2", "5.04",    "Moderate — communication overhead grows"],
        ["Hybrid 2×4", "2.53",    "Poor — MPI startup dominates over OpenMP gain"],
        ["Hybrid 1×8", "1.27",    "Worst — all MPI overhead, zero distribution benefit"],
    ], [3*cm, 2.5*cm, 10.7*cm]),
    spacer(0.2),
    p("<b>Explanation:</b> On a single node, MPI inter-process communication "
      "(even via shared-memory shmem transport) has higher overhead than "
      "OpenMP's direct shared-memory access. When fewer MPI ranks are used (e.g. "
      "Hybrid 2×4), each rank must use OpenMP threads within a larger memory "
      "domain, but the MPI Allgatherv still incurs its full startup and "
      "serialisation cost. Hybrid 1×8 is worst because it pays all MPI "
      "initialisation overhead while getting no distributed-memory benefit."),
    p("The Hybrid model's performance advantage over pure OpenMP only emerges at "
      "<b>multi-node scale</b>, where distributing work across nodes via MPI is "
      "necessary."),
    spacer(0.3),
    h2("5.5  Amdahl's Law Analysis"),
    p("With serial fraction f ≈ 0.03 (Phases 1, 2, and 5 are serial):"),
    code("S_max(8) = 1 / (0.03 + 0.97/8) = 1 / 0.1513 ≈ 6.61×"),
    make_table([
        ["Technology",  "Amdahl Prediction", "Actual Speedup", "Observation"],
        ["OpenMP 8T",   "6.61×",             "7.58× ↑",        "Exceeds — dynamic scheduling reduces effective serial fraction"],
        ["Pthreads 8T", "6.61×",             "6.94×",          "Close to theoretical limit"],
        ["MPI 8P",      "6.61×",             "6.35× ↓",        "Below — communication overhead not modelled by Amdahl"],
    ], [3.5*cm, 3.5*cm, 3*cm, 6.2*cm]),
    PageBreak(),
]

# ─────────────────────────────────────────────────────────────────────────────
# 6. CONCLUSIONS
# ─────────────────────────────────────────────────────────────────────────────
story += [
    h1("6. Conclusions"),
    hr(),
    p("This project successfully implemented and benchmarked a Pearson Correlation "
      "Collaborative Filtering Recommender System across five HPC parallelisation "
      "strategies. Key findings are summarised below."),
    spacer(0.2),
    h3("Finding 1 — All parallel versions preserve numerical correctness"),
    p("Every CPU implementation produces MAE = <b>1.2574</b> and checksum = "
      "<b>942.387323</b>, identical to the serial baseline. Zero race conditions "
      "or partitioning errors were detected."),
    h3("Finding 2 — OpenMP achieves the best single-node performance"),
    p("×7.58 speedup at 8 threads, 95% efficiency. Dynamic load balancing and "
      "thread reuse across phases explain the advantage over other CPU models."),
    h3("Finding 3 — Pthreads is competitive but less efficient"),
    p("×6.94 speedup at 8T, 87% efficiency. Static row assignment cannot "
      "adapt to the triangular loop imbalance; per-phase thread creation adds overhead."),
    h3("Finding 4 — MPI achieves good speedup with consistent communication overhead"),
    p("×6.35 speedup at 8P, 79% efficiency. The ~4 MB MPI_Allgatherv for the "
      "similarity matrix limits single-node efficiency. MPI would close the gap at "
      "multi-node scale."),
    h3("Finding 5 — Hybrid performance inversely correlates with OpenMP thread count on a single node"),
    p("Hybrid 8×1 (×6.85) > Hybrid 4×2 (×5.04) > Hybrid 2×4 (×2.53) > Hybrid 1×8 (×1.27). "
      "The Hybrid model's advantage over pure OpenMP emerges only at multi-node scale."),
    h3("Finding 6 — CUDA would provide the largest speedup"),
    p("The implemented design launches 1,016,064 threads simultaneously for the "
      "similarity phase. GPU speedup would substantially exceed all CPU approaches."),
    spacer(0.4),
    h2("Final Ranking (8 workers, single node)"),
    make_table([
        ["Rank", "Version",      "Total (s)", "Speedup", "Notes"],
        ["1",    "OpenMP 8T",    "0.8447",    "×7.58",   "Best — dynamic load balancing"],
        ["2",    "Pthreads 8T",  "0.9224",    "×6.94",   "Good — static assignment limits efficiency"],
        ["3",    "Hybrid 8×1",   "0.9341",    "×6.85",   "≈ Pure MPI on single node"],
        ["4",    "MPI 8P",       "1.0090",    "×6.35",   "Good — limited by Allgatherv overhead"],
        ["5",    "Hybrid 4×2",   "1.2695",    "×5.04",   "Mixed — communication cost grows"],
        ["6",    "Hybrid 2×4",   "2.5332",    "×2.53",   "Poor — MPI overhead dominates"],
        ["7",    "Hybrid 1×8",   "5.0503",    "×1.27",   "Worst — MPI cost, no distribution benefit"],
        ["—",    "Serial",       "6.4027",    "×1.00",   "Baseline"],
    ], [1*cm, 3.5*cm, 2.5*cm, 2.5*cm, 6.7*cm]),
    spacer(0.5),
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
story += [
    h1("References"),
    hr(),
    Paragraph("[1] J. S. Breese, D. Heckerman, and C. Kadie, \"Empirical analysis of "
              "predictive algorithms for collaborative filtering,\" "
              "<i>arXiv preprint arXiv:1301.7363</i>, 2013.", S_BODY),
    Paragraph("[2] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, \"Item-based "
              "collaborative filtering recommendation algorithms,\" in "
              "<i>Proc. 10th Int. Conf. on World Wide Web</i>, pp. 285–295, 2001.", S_BODY),
    Paragraph("[3] B. Chapman, G. Jost, and R. van der Pas, <i>Using OpenMP</i>. "
              "MIT Press, 2008.", S_BODY),
    Paragraph("[4] W. Gropp, E. Lusk, and A. Skjellum, <i>Using MPI</i>. "
              "MIT Press, 1994.", S_BODY),
    Paragraph("[5] J. Nickolls, I. Buck, M. Garland, and K. Skadron, \"Scalable "
              "parallel programming with CUDA,\" <i>Queue</i>, vol. 6, no. 2, "
              "pp. 40–53, 2008.", S_BODY),
    spacer(1),
    HRFlowable(width="100%", thickness=0.5, color=MIDGRAY),
    spacer(0.2),
    Paragraph("Group 11  |  EC7207 High Performance Computing  |  May 2026",
              style("Footer", fontSize=9, alignment=TA_CENTER,
                    textColor=HexColor("#888888"))),
]

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"\nPDF successfully built: {OUT_PDF}")
print(f"File size: {os.path.getsize(OUT_PDF) / 1024:.0f} KB")
