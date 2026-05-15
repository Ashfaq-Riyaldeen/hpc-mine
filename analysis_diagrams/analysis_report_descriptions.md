# Analysis Report — Descriptions
## EE7218 / EC7207: High Performance Computing
## Group 11 — Pearson Correlation-Based Collaborative Filtering Recommender System

---

# 1. Diagram and Description — How Parallel Programming Concepts Were Applied

## 1.1 Project Overview and Algorithm

This project implements a **Pearson Correlation-Based Collaborative Filtering Recommender System** that predicts user ratings for unseen items using similarity between users. The system was first implemented as a serial baseline and then parallelised using five HPC technologies: OpenMP, POSIX Threads (Pthreads), MPI, CUDA, and a Hybrid MPI+OpenMP approach.

The algorithm operates on a synthetic ratings matrix of **1,000 users × 1,000 items** (N=1000, M=1000) with the following fixed parameters:

| Parameter | Value |
|-----------|-------|
| Users (N) | 1,000 |
| Items (M) | 1,000 |
| Random Seed | 42 |
| Sparsity | 70% |
| Test Split | 10% |
| TOP-K Neighbours | 20 |

---

## 1.2 Algorithm Pipeline (Diagram 1)

The system executes in five sequential phases:

**Phase 1 — Data Generation** [O(N×M)]
The ratings matrix is populated using a fixed seed (`srand(SEED=42)`) to ensure every parallel version generates the same data. This guarantees that accuracy (MAE) comparisons across all implementations are perfectly fair — every version sees identical input.

**Phase 2 — User Mean Computation** [O(N×M)]
Each user's mean rating (μ_u) is computed across their known entries. This normalises individual rating scales before the Pearson formula is applied.

**Phase 3 — Pearson Similarity Matrix** [O(N²×M)] ← **DOMINANT BOTTLENECK**
The Pearson correlation coefficient between every pair of users (u, v) is computed:

```
sim(u,v) = Σ(R_ui − μ_u)(R_vi − μ_v) / [√Σ(R_ui−μ_u)² × √Σ(R_vi−μ_v)²]
```

With N=1000, this requires 499,500 unique Pearson computations (≈500 million floating-point operations), accounting for approximately **97% of total runtime**. This phase is the primary target of all parallelisation strategies.

**Phase 4 — Rating Predictions** [O(N²×M)]
For each unrated (user, item) pair, the TOP-20 most similar users with positive similarity are selected as neighbours. The weighted prediction formula is applied:

```
pred(u,i) = μ_u + Σ_k [sim(u,k) × (R_ki − μ_k)] / Σ_k sim(u,k)
```

**Phase 5 — MAE Evaluation** [O(|T|)]
Mean Absolute Error is computed over all test-set entries. All parallel versions must produce MAE within **±0.001** of the serial baseline to pass correctness verification.

---

## 1.3 OpenMP — Shared-Memory Fork-Join Parallelism (Diagram 2)

**Concept Applied:**
OpenMP exploits **shared-memory parallelism** using compiler directives. A single master thread forks into T worker threads at a parallel region, all sharing the same process memory space, then joins back into one thread at the barrier.

**Application to Phase 3 (Similarity — bottleneck):**
The outer loop over users is distributed across T threads using `#pragma omp parallel for schedule(dynamic, 4)`. Dynamic scheduling was specifically chosen because the inner loop for user u processes `N−u−1` pairs — Thread 0 performs 999 Pearson calls while Thread 3 (for T=4) performs only 249. Static scheduling would create severe load imbalance; dynamic scheduling lets idle threads pull new chunks of 4 rows from a shared work queue.

**Race-Free Correctness:**
Each thread owns a contiguous set of outer-loop values. It writes both `SIM(u,v)` and `SIM(v,u)` for its owned u. Since no two threads own the same u, no two threads write to the same cell simultaneously — no mutex or atomic operation is required.

**Application to Phase 4 (Predictions):**
Each thread allocates its own private `SimPair *nbrs` buffer on the heap inside the parallel block. This eliminates all shared state during neighbour sorting, preventing race conditions without locking.

**Application to Phase 5 (MAE):**
The `reduction(+:err)` clause gives each thread a private copy of the error accumulator initialised to zero. Threads accumulate independently in parallel; OpenMP automatically sums all private copies at the implicit barrier. Wall-clock time is measured with `omp_get_wtime()` (not `clock()`, which would sum CPU time across all threads).

---

## 1.4 Pthreads — Manual POSIX Thread Management (Diagram 3)

**Concept Applied:**
POSIX Threads provide **explicit, low-level manual thread management**. Unlike OpenMP where the compiler generates thread orchestration from directives, Pthreads requires the programmer to explicitly create threads, assign work, and synchronise using join barriers.

**Thread Lifecycle Pattern:**
The `run_parallel(fn, nthreads)` harness is invoked at the start of each parallel phase (Phases 2, 3, and 4):
1. `pthread_create(&threads[t], NULL, fn, &args[t])` — creates T threads, each receiving a `ThreadArgs` struct containing its thread ID (`tid`) and total thread count (`nthreads`).
2. Each thread independently derives its exclusive row range using ceiling division: `start = tid × chunk`, `end = min(start + chunk, N_USERS)`, where `chunk = ⌈N_USERS / nthreads⌉`.
3. `pthread_join(threads[t], NULL)` — the main thread waits for all T threads to complete, acting as a **phase barrier**. The next phase never begins until all threads have finished the current one.

**Difference from OpenMP:**
In Pthreads, threads are created and joined separately for each phase (three separate `run_parallel()` calls). In OpenMP, a single `parallel` region can host multiple `for` loops with implicit inter-loop barriers. Pthreads offers no such shortcut — `pthread_join` is the only available barrier primitive.

**Thread-Private State:**
In Phase 4, each thread allocates its own `SimPair *nbrs` buffer on the heap inside the thread function and frees it before returning. This eliminates any shared data structure for neighbour sorting, achieving race-freedom without locks.

---

## 1.5 MPI — Distributed-Memory Parallelism (Diagram 4)

**Concept Applied:**
MPI (Message Passing Interface) implements **distributed-memory parallelism** where each rank is a completely independent process with its own private address space. Data sharing between ranks requires explicit message passing using MPI communication primitives.

**Data Partitioning Strategy (Row Partitioning):**
The N=1000 users are divided evenly across P ranks. Rank r owns users in the range `[r×(N/P), (r+1)×(N/P))`. With P=4: Rank 0 owns users 0–249, Rank 1 owns 250–499, and so on.

**Phase-by-Phase Communication Strategy:**

| Phase | Strategy |
|-------|----------|
| Phase 1 | All ranks call `srand(42)` independently → identical data, no communication |
| Phase 2 | Each rank computes its slice of `user_mean[]` → `MPI_Allgatherv` assembles full mean array on every rank |
| Phase 3 | Each rank fills all columns for its owned rows → `MPI_Allgatherv` distributes full N×N `sim_matrix` to every rank |
| Phase 4 | Each rank computes predictions for its own users using the full `sim_matrix` — no MPI needed |
| Phase 5 | Each rank computes local error sum → `MPI_Reduce(MPI_SUM)` delivers global total to Rank 0 |

**Key MPI Primitives Used:**
- `MPI_Allgatherv`: each rank sends a variable-length slice; every rank receives the fully assembled array. Used for both `user_mean[]` and `sim_matrix[]`.
- `MPI_Reduce(MPI_SUM)`: collects partial MAE sums from all ranks into Rank 0.
- `MPI_Barrier`: explicit synchronisation point between phases.
- `MPI_Wtime()`: portable wall-clock timer replacing `clock_gettime()`.

---

## 1.6 CUDA — GPU Massively Parallel Computing (Diagram 5)

**Concept Applied:**
CUDA exploits the **massively parallel architecture** of NVIDIA GPUs. Where CPU parallelism uses tens of threads, CUDA launches thousands of threads simultaneously, organised into a two-level grid/block hierarchy.

**Thread Hierarchy for Phase 3:**
The similarity matrix computation uses a **2D grid** of thread blocks:
- Grid dimension: ⌈N/16⌉ × ⌈N/16⌉ = 63×63 = **3,969 blocks** for N=1000
- Block dimension: 16×16 = **256 threads per block**
- Total threads: 63×63×256 = **1,016,064 threads** — one per (u,v) pair
- Each thread identifies its pair: `u = blockIdx.x×16 + threadIdx.x`, `v = blockIdx.y×16 + threadIdx.y`
- Threads with u ≥ v return immediately (upper triangle only), while threads with u < v compute the full Pearson formula and write both `SIM(u,v)` and `SIM(v,u)` — race-free since no two threads own the same u.

**Three CUDA Kernels:**

| Kernel | Grid | Assignment |
|--------|------|------------|
| `kernel_user_means` | 1D, `<<<(N+255)/256, 256>>>` | 1 thread = 1 user's mean |
| `kernel_similarity` | 2D, `dim3(⌈N/16⌉,⌈N/16⌉), dim3(16,16)` | 1 thread = 1 (u,v) pair |
| `kernel_predictions` | 2D, users × items | 1 thread = 1 (user, item) prediction |

**Memory Hierarchy Utilisation:**
- **Registers**: per-thread scalars (`num`, `den_u`, `den_v`) and the entire TOP-K arrays (`top_sim[20]`, `top_rat[20]`, `top_mu[20]`) — sub-cycle latency.
- **Global Memory (GPU DRAM)**: all large arrays (`d_ratings`, `d_user_mean`, `d_sim`, `d_pred`) — ~300–600 cycle latency.
- **Shared Memory**: not used — the problem is embarrassingly parallel with no data sharing required between threads in the same block.

**Asynchronous Timing:**
CUDA kernel launches return immediately to the CPU (asynchronous). Standard `clock_gettime()` would measure near-zero host time. Instead, `cudaEventRecord()` inserts GPU-side timestamps into the command stream, and `cudaEventSynchronize()` blocks the CPU until the GPU completes before the elapsed time is read.

---

## 1.7 Hybrid MPI+OpenMP — Two-Level Parallelism (Diagram 6)

**Concept Applied:**
The Hybrid approach combines **MPI for coarse-grain inter-node distribution** and **OpenMP for fine-grain intra-node threading**, implementing two-level parallelism that matches modern HPC cluster architecture where each node has multiple CPU cores connected to other nodes via a high-speed network.

**Two-Level Decomposition:**
- **Level 1 (MPI — coarse grain):** P MPI ranks partition users across nodes. Each rank owns a contiguous user range.
- **Level 2 (OpenMP — fine grain):** Within each rank, T OpenMP threads sub-divide the local user range. With P=2, T=4: Rank 0's 4 threads handle rows 0–124, 125–249, 250–374, 375–499 respectively.
- **Total workers:** P × T (e.g., 2×4 = 8 workers)

**Thread Safety — MPI_THREAD_FUNNELED:**
`MPI_Init_thread(MPI_THREAD_FUNNELED)` is called instead of `MPI_Init`. This threading level guarantees that all MPI calls are made exclusively from the main thread. After the OpenMP parallel region completes its implicit barrier, the main thread calls `MPI_Allgatherv` — this ordering satisfies the FUNNELED constraint and avoids undefined behaviour from concurrent MPI calls.

**Two-Level Reduction (Phase 5):**
- **Level 2 (within rank):** `reduction(+:local_err)` merges per-thread error accumulators within each MPI rank.
- **Level 1 (across ranks):** `MPI_Reduce(MPI_SUM)` sums the local errors from all ranks into Rank 0, which computes and prints the global MAE.

**Four Benchmarked Configurations (all 8 total workers):**

| Configuration | MPI Ranks (P) | OMP Threads (T) | Characteristic |
|--------------|--------------|-----------------|----------------|
| Hybrid 2×4 | 2 | 4 | Fewer MPI messages; more shared-memory parallelism |
| Hybrid 4×2 | 4 | 2 | Balanced communication vs. threading overhead |
| Hybrid 8×1 | 8 | 1 | Equivalent to pure MPI (no OpenMP benefit) |
| Hybrid 1×8 | 1 | 8 | Equivalent to pure OpenMP (no MPI distribution) |

---

# 2. Accuracy Comparison — Parallel vs. Serial

All parallel implementations must produce a **Mean Absolute Error (MAE) within ±0.001** of the serial baseline. This tolerance confirms that parallelisation did not alter the computation results — only the execution order and thread assignment.

**MAE Formula:**
```
MAE = (1 / |T|) × Σ |pred(u,i) − actual(u,i)|
```
where |T| is the number of test-set entries (approximately N × M × 10% = 100,000 entries).

**Why MAE rather than RMSE?**
MAE treats all prediction errors equally, making it more robust to occasional large outliers. RMSE would penalise large deviations more heavily, but for this system where all errors are bounded (ratings ∈ {1–5}), MAE provides a clearer measure of average prediction quality.

**Correctness Verification — Sim-Matrix Checksum:**
In addition to MAE, a checksum of the entire similarity matrix (sum of all N×N values) is computed. All CPU-based versions (Serial, OpenMP, Pthreads, MPI, Hybrid) must produce an **identical checksum** since they use the same IEEE 754 floating-point arithmetic with the same input data. CUDA may show minor differences due to GPU floating-point rounding order but must still pass the MAE tolerance check.

**MAE Comparison Table:**

> **Note:** Fill these values after running `bash results/run_benchmarks.sh`

| Version | Workers | MAE | Matches Serial? | Sim-Matrix Checksum |
|---------|---------|-----|-----------------|---------------------|
| Serial | 1 | ___._____ | baseline | _______________ |
| OpenMP | 1 | ___._____ | YES / NO | _______________ |
| OpenMP | 2 | ___._____ | YES / NO | _______________ |
| OpenMP | 4 | ___._____ | YES / NO | _______________ |
| OpenMP | 8 | ___._____ | YES / NO | _______________ |
| Pthreads | 1 | ___._____ | YES / NO | _______________ |
| Pthreads | 2 | ___._____ | YES / NO | _______________ |
| Pthreads | 4 | ___._____ | YES / NO | _______________ |
| Pthreads | 8 | ___._____ | YES / NO | _______________ |
| MPI | 1 | ___._____ | YES / NO | _______________ |
| MPI | 2 | ___._____ | YES / NO | _______________ |
| MPI | 4 | ___._____ | YES / NO | _______________ |
| MPI | 8 | ___._____ | YES / NO | _______________ |
| CUDA | GPU | ___._____ | YES / NO (±0.001) | N/A (GPU rounding) |
| Hybrid 2×4 | 8 | ___._____ | YES / NO | _______________ |
| Hybrid 4×2 | 8 | ___._____ | YES / NO | _______________ |
| Hybrid 8×1 | 8 | ___._____ | YES / NO | _______________ |
| Hybrid 1×8 | 8 | ___._____ | YES / NO | _______________ |

**Expected Result:**
All CPU-based versions should produce **identical MAE** to the serial baseline since they perform the same operations with the same data and the same IEEE 754 arithmetic. Any deviation greater than ±0.001 indicates a race condition, incorrect synchronisation, or a data partitioning error.

---

# 3. Timing Measurements — Serial vs. Parallel

## 3.1 Experimental Setup

All benchmarks were conducted with the following fixed problem size:

| Setting | Value |
|---------|-------|
| Problem size | N_USERS = 1,000, N_ITEMS = 1,000 |
| Random seed | SEED = 42 |
| TOP-K | 20 |
| Timing method | Wall-clock (`omp_get_wtime()` / `MPI_Wtime()` / `cudaEvent`) |
| Reported metrics | Similarity phase time, Prediction phase time, Total time |

## 3.2 Timing Results

> **Note:** Fill these values after running `bash results/run_benchmarks.sh`

### Serial Baseline

| Phase | Time (s) |
|-------|---------|
| Similarity matrix | _______ |
| Prediction phase | _______ |
| **Total** | **_______** |

---

### OpenMP — Varying Thread Count

| Threads (T) | Sim Time (s) | Pred Time (s) | Total (s) | Speedup S(T) | Efficiency E(T) |
|-------------|-------------|--------------|-----------|-------------|-----------------|
| 1 | | | | | |
| 2 | | | | | |
| 4 | | | | | |
| 8 | | | | | |

---

### Pthreads — Varying Thread Count

| Threads (T) | Sim Time (s) | Pred Time (s) | Total (s) | Speedup S(T) | Efficiency E(T) |
|-------------|-------------|--------------|-----------|-------------|-----------------|
| 1 | | | | | |
| 2 | | | | | |
| 4 | | | | | |
| 8 | | | | | |

---

### MPI — Varying Process Count

| Processes (P) | Sim Time (s) | Pred Time (s) | Total (s) | Speedup S(P) | Efficiency E(P) |
|--------------|-------------|--------------|-----------|-------------|-----------------|
| 1 | | | | | |
| 2 | | | | | |
| 4 | | | | | |
| 8 | | | | | |

---

### CUDA — GPU Execution

| Kernel | Time (s) |
|--------|---------|
| `kernel_user_means` | _______ |
| `kernel_similarity` | _______ |
| `kernel_predictions` | _______ |
| cudaMemcpy H→D | _______ |
| cudaMemcpy D→H | _______ |
| **Total (including transfers)** | **_______** |
| **Total (kernels only)** | **_______** |
| Speedup vs. Serial | _______ |

---

### Hybrid MPI+OpenMP — Fixed 8 Total Workers

| Configuration | P | T | Sim Time (s) | Pred Time (s) | Total (s) | Speedup | Efficiency |
|--------------|---|---|-------------|--------------|-----------|---------|------------|
| Hybrid 2×4 | 2 | 4 | | | | | |
| Hybrid 4×2 | 4 | 2 | | | | | |
| Hybrid 8×1 | 8 | 1 | | | | | |
| Hybrid 1×8 | 1 | 8 | | | | | |

---

## 3.3 Performance Metric Definitions

**Speedup:**
```
S(p) = T_serial / T_parallel(p)
```
Measures how many times faster the parallel version is compared to the serial baseline.

**Efficiency:**
```
E(p) = S(p) / p
```
Measures how well each worker is utilised. E(p) = 1.0 means perfect scaling; E(p) < 1.0 means overhead (synchronisation, communication, load imbalance) is reducing effectiveness.

**Amdahl's Law — Theoretical Speedup Limit:**
```
S_max = 1 / (f + (1-f)/p)
```
where f is the serial fraction of the program. Since Phase 3 dominates (~97% of runtime) and Phases 1 and 5 remain serial, the theoretical maximum speedup approaches **1/0.03 ≈ 33×** regardless of how many parallel workers are added.

## 3.4 Analysis Notes

**Expected Observations (to be confirmed with real data):**

1. **OpenMP vs. Pthreads:** Both target the same shared-memory architecture and should produce similar speedups. OpenMP may be marginally faster due to runtime load-balancing optimisations; Pthreads may have lower overhead at low thread counts due to direct system call threading.

2. **MPI Scaling:** With distributed memory, MPI introduces `MPI_Allgatherv` communication overhead (transferring the full N×N similarity matrix ≈ 4 MB at N=1000). This overhead increases with P, limiting efficiency at high process counts compared to shared-memory models.

3. **CUDA Speedup:** CUDA parallelises all 499,500 similarity pairs simultaneously and is expected to give the highest absolute speedup for the similarity phase. However, PCIe transfer time (Host→Device for `d_ratings`, Device→Host for `d_sim`) must be included in the total time for a fair comparison.

4. **Hybrid Configuration Trade-off:** The 2×4 configuration (fewer MPI ranks, more OpenMP threads) is expected to outperform 8×1 (pure MPI) because it reduces `MPI_Allgatherv` communication volume while still using 8 workers. The 4×2 configuration balances both. The 1×8 configuration degenerates to pure OpenMP.

5. **Efficiency Degradation:** Efficiency typically decreases as worker count increases due to:
   - Synchronisation overhead (barriers, joins)
   - Load imbalance (triangular loop structure)
   - Communication cost (MPI Allgatherv grows with P)
   - Memory bandwidth saturation (shared-memory models)

---

## 3.5 How to Run Benchmarks

Run the following command from the project root on a Linux/HPC system:

```bash
bash results/run_benchmarks.sh
```

This will automatically:
- Compile all versions (Serial, OpenMP, Pthreads, MPI, Hybrid)
- Run each version with the specified worker counts
- Parse timing output and write to `results/speedup_table.csv`
- Parse MAE and checksum output and write to `results/mae_comparison.txt`
- Save full terminal output to `results/timing_raw.txt`

After running, the Python chart script will use the real numbers:
```bash
python analysis_diagrams/generate_charts.py
```

Update the `── YOUR DATA ──` section in `generate_charts.py` with values from `results/timing_raw.txt` before generating charts.

---

*Group 11 | EE7218 / EC7207 High Performance Computing*
