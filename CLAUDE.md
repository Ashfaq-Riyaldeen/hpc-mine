# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A university HPC project (EC7207 / EE7218, Group 11): one **User-Based Collaborative Filtering recommender** (Pearson-correlation similarity) implemented **six times** — Serial, OpenMP, Pthreads, MPI, CUDA, and Hybrid MPI+OpenMP — so their performance and accuracy can be compared. The deliverable is the comparison itself (the analysis report + diagrams), not a production system.

## The core invariant (read this before editing any `*_recommender.*`)

**All six versions implement the identical algorithm on identical data, and must produce identical results.** This is the entire point of the project — only the *runtime/parallelisation* differs. Concretely:

- The fixed constants `SEED=42`, `SPARSITY=0.70f`, `TOP_K=20`, `TEST_RATIO=0.10f`, `DEFAULT_USERS/ITEMS=1000` are **duplicated in every source file** and must stay byte-identical across all of them. Changing one without the others invalidates every comparison.
- Phase 1 (data generation) is **always serial**, even in the parallel versions, because C `rand()` is not thread-safe. Every version runs the same `srand(SEED)` + `rand()` sequence to build the same matrix without communication. Never parallelise it.
- Correctness is verified two ways, printed by every run: `[Eval] MAE` (must match serial within ±0.001) and `[Check] Sim-matrix checksum` (must be **bit-identical** across all CPU versions). If you change the algorithm, you must change it the same way in all six files, or these will diverge — which the benchmark flags as a race/partitioning bug.

## Algorithm structure (shared by all versions)

Five sequential phases; the parallelisation targets are Phase 3 and Phase 4:

1. Data generation — serial, sparse ratings matrix + held-out test set
2. User means `μ_u`
3. **Pearson similarity matrix** — O(N²·M), the bottleneck
4. **Predictions** — TOP-K weighted neighbour average
5. MAE/RMSE evaluation on the test set

Matrices are flat 1-D arrays accessed through the macros `R(u,i)`, `SIM(u,v)`, `PRED(u,i)` (row-major). `0.0f` in `ratings[]` means "unrated", not a rating of zero — this sentinel is load-bearing in every phase.

Per-version parallelisation strategy (what actually differs between files):
- **OpenMP** — `#pragma omp parallel for`, `schedule(dynamic,4)` on the triangular similarity loop for load balance; `reduction(+:err)` for MAE; per-thread private `nbrs[]`.
- **Pthreads** — manual `pthread_create`/`pthread_join` per phase (join = barrier); static row ranges via `work_range()`.
- **MPI** — row-partitioned; `MPI_Allgatherv` rebuilds the full matrix on every rank; `MPI_Reduce` for MAE; no broadcast of data (same-seed trick).
- **CUDA** — one thread per (u,v) pair / (user,item) cell; TOP-K kept in registers (no `qsort` on device); `cudaEvent` timing.
- **Hybrid** — MPI row partition across ranks, OpenMP threads within each rank; `MPI_THREAD_FUNNELED` (MPI called only from the main thread, outside OMP regions).

## Build & run

There is **no Makefile** — compile each version directly. The code is written for **Linux** (`clock_gettime`, `mpirun`, POSIX threads); the dev machine here is Windows, so build/run under WSL or a Linux box. Compiled binaries (`*_rec`) are gitignored.

```bash
gcc   -O2 -Wall            -o serial_rec   serial/serial_recommender.c              -lm
gcc   -O2 -Wall -fopenmp   -o openmp_rec   openmp/openmp_recommender.c              -lm
gcc   -O2 -Wall            -o pthreads_rec pthreads/pthreads_recommender.c -lpthread -lm
mpicc -O2 -Wall            -o mpi_rec      mpi/mpi_recommender.c                    -lm
mpicc -O2 -Wall -fopenmp   -o hybrid_rec   hybrid/hybrid_recommender.c              -lm
nvcc  -O2 -arch=sm_75      -o cuda_rec     cuda/cuda_recommender.cu                 -lm   # match -arch to the GPU
```

Run signatures differ — note how the worker count is passed in each case:
```bash
./serial_rec [users] [items]
OMP_NUM_THREADS=8 ./openmp_rec [users] [items]      # threads via env var
./pthreads_rec [users] [items] [threads]            # threads as 3rd arg
mpirun -np 8 ./mpi_rec [users] [items]              # processes via -np
mpirun -np 2 env OMP_NUM_THREADS=4 ./hybrid_rec     # P ranks × T threads
./cuda_rec [users] [items]
```

## Benchmarking

`bash results/run_benchmarks.sh` (run from project root) compiles everything, sweeps thread/process counts (1,2,4,8) and the four hybrid P×T configs, and auto-fills `results/timing_raw.txt`, `results/speedup_table.csv`, and `results/mae_comparison.txt`. It parses the `[Timing]`, `[Eval]`, `[Check]` lines that every binary prints — preserve that output format if you touch the C code, or the script's `grep`/`awk` parsing breaks. Speedup is `Serial_total / Parallel_total` from the `[Timing] Total (sim+pred)` line.

## Report & diagrams

- `analysis_report/ANALYSIS_REPORT.md` is the primary report and must satisfy the three guideline requirements in `Project Guideline.pdf`: (1) a diagram + description of the parallel concepts applied, (2) accuracy vs serial (MAE/checksum), (3) timing across varying threads/parameters. A parallel LaTeX copy (`analysis_report.tex` → `.pdf`) also exists and can drift from the Markdown — keep them in sync when changing report content.
- `analysis_diagrams/drawio/` holds six `.drawio` diagrams (one per concept) plus `all_diagrams.drawio` (all six as tabs) and `diagram_descriptions.md`. Export to PNG for embedding with the installed draw.io desktop CLI:
  ```bash
  "/c/Program Files/draw.io/draw.io.exe" --export --format png --scale 2 --border 10 --output png/<name>.png <name>.drawio
  ```
  Diagrams use orthogonal edges and grid-aligned coordinates — keep arrows orthogonal and avoid long diagonal connectors so they render cleanly.
- `analysis_diagrams/charts/` holds the performance PNGs referenced by the report (Figures 7–14; the diagrams are Figures 1–6).
