# Diagram Descriptions — Parallel Programming Concepts Applied
## EE7218 / EC7207 High Performance Computing — Group 11
### Topic: Pearson Correlation-Based Collaborative Filtering Recommender System

---

## Diagram 1 — 5-Phase Algorithm Pipeline
**File:** `diagram_01_algorithm_pipeline.drawio`

### Description
This diagram illustrates the five sequential phases of the Pearson Correlation Recommender System and identifies which phases are parallelised across all HPC implementations.

The algorithm operates on a 1000×1000 user-item ratings matrix (N_USERS = 1000, N_ITEMS = 1000, SEED = 42, 70% sparsity, 10% test split):

| Phase | Operation | Complexity | Parallelised? |
|-------|-----------|------------|---------------|
| 1 | Data Generation (srand SEED=42) | O(N × M) | Serial only — RNG must be deterministic |
| 2 | User Mean Computation (μ_u) | O(N × M) | OpenMP, Pthreads, MPI, Hybrid |
| 3 | Pearson Similarity Matrix | O(N² × M) | All versions including CUDA |
| 4 | Rating Predictions (TOP-K=20) | O(N² × M) | All versions including CUDA |
| 5 | MAE Evaluation | O(\|T\|) | Serial only — trivially fast |

**Phase 3 is the dominant bottleneck**, consuming ~97% of total runtime with 499,500 unique Pearson correlation computations (≈500 million floating-point operations). This justifies why all parallelisation effort targets Phases 3 and 4.

All parallel versions use SEED=42 to generate identical data, ensuring that accuracy (MAE) comparisons are fair — every version sees exactly the same ratings matrix. The correctness threshold is ±0.001 MAE compared to the serial baseline.

---

## Diagram 2 — OpenMP (Shared-Memory Fork-Join)
**File:** `diagram_02_openmp.drawio`

### Description
OpenMP was applied as a shared-memory parallelisation strategy using the compiler-directive fork-join model. It is the simplest to implement because all threads automatically share the same process memory — no data copying or message passing is needed.

**How parallel programming concepts were applied:**

**Fork-Join Model:**  
`#pragma omp parallel for` causes the master thread to *fork* T worker threads at the start of a parallel section. All threads execute the loop body concurrently and *join* back into one thread at the closing brace (implicit barrier).

**Phase 3 — Similarity (Bottleneck):**  
The outer loop `for (u = 0; u < N_USERS; u++)` is distributed across T threads with `schedule(dynamic, 4)`. Dynamic scheduling is critical here because the inner loop for user u runs `N-u-1` iterations — Thread 0 (u=0) does 999 Pearson calls while Thread 3 (u=750) does only 249. Static scheduling would cause severe load imbalance; dynamic scheduling lets idle threads pull new chunks of 4 rows from a shared work queue.

**Race-Free Writes:**  
Thread T owns exclusive rights to write `SIM(u, v)` for all u in its chunk. It also writes `SIM(v, u)` symmetrically, but since no other thread owns the same u value, no two threads write to the same cell simultaneously — no mutex is needed.

**Phase 4 — Predictions:**  
Each thread allocates its own private `SimPair *nbrs` buffer on the heap inside the parallel region. This avoids any shared state during neighbour sorting.

**Phase 5 — MAE Reduction:**  
`reduction(+:err)` gives each thread a private copy of the error accumulator (initialised to 0). Threads sum independently, and OpenMP automatically combines all private copies at the barrier.

**Timing:**  
`omp_get_wtime()` is used instead of `clock()` because `clock()` accumulates CPU time across all threads — 4 threads running for 1 second would report 4 seconds. `omp_get_wtime()` measures wall-clock elapsed time.

---

## Diagram 3 — Pthreads (POSIX Thread Management)
**File:** `diagram_03_pthreads.drawio`

### Description
POSIX Threads (Pthreads) provide explicit, low-level manual thread management. Unlike OpenMP where the compiler generates thread code from directives, Pthreads requires the programmer to explicitly create, assign work to, and collect threads.

**How parallel programming concepts were applied:**

**Thread Lifecycle:**  
The `run_parallel()` harness is called at the start of each parallel phase (Phases 2, 3, and 4). It creates T threads using `pthread_create(&threads[t], NULL, fn, &args[t])`, waits for all of them using `pthread_join(threads[t], NULL)`, and then returns. The join loop acts as a phase barrier — the next phase never starts until all threads have finished the current one.

**Work Distribution:**  
Each thread receives a `ThreadArgs` struct containing its thread ID (0-based) and total thread count. From these two values it derives its exclusive row range using ceiling division:
```
chunk = (N_USERS + nthreads - 1) / nthreads
start = tid * chunk
end   = min(start + chunk, N_USERS)
```
No global shared counter or mutex is needed for work assignment.

**Thread-Private State:**  
In Phase 4 (predictions), each thread allocates its own `SimPair *nbrs` buffer on the heap inside the thread function and frees it before returning. This pattern eliminates all shared state and prevents race conditions without locks.

**Three Separate Phases:**  
Pthreads threads are created and joined three times — once per parallel phase. This differs from OpenMP where a single `#pragma omp parallel` region can host multiple `#pragma omp for` loops. In Pthreads, `pthread_join` is the only available synchronisation barrier between phases.

---

## Diagram 4 — MPI (Distributed-Memory Row Partitioning)
**File:** `diagram_04_mpi.drawio`

### Description
MPI (Message Passing Interface) was applied as a distributed-memory parallelisation strategy. Unlike shared-memory models (OpenMP, Pthreads), each MPI rank runs as a fully independent process with its own private address space. Communication between ranks requires explicit MPI calls.

**How parallel programming concepts were applied:**

**Data Partitioning:**  
The N_USERS users are divided evenly across P ranks. Rank r owns users `[r×(N/P), (r+1)×(N/P))`. With P=4 and N=1000, Rank 0 owns users 0–249, Rank 1 owns 250–499, etc.

**Phase 1 — No Communication:**  
All ranks independently call `srand(SEED=42)` and generate the full ratings matrix locally. Because the same seed produces the same pseudorandom sequence on every rank, no initial broadcast is needed.

**Phase 2 — MPI_Allgatherv for Means:**  
Each rank computes the mean for its own user slice. `MPI_Allgatherv` then assembles all partial mean arrays into a complete `user_mean[]` array on every rank. Variable-length receives (`recvcounts[]`, `displs[]` arrays) handle unequal partitions.

**Phase 3 — MPI_Allgatherv for Similarity:**  
Each rank computes all similarity values `SIM(u, v)` for its owned rows u (all columns v). After `MPI_Allgatherv`, every rank holds the complete N×N similarity matrix. This eliminates any further communication in Phase 4.

**Phase 4 — No Communication:**  
Since every rank has the full similarity matrix, each rank can independently compute predictions for its own users without any MPI call.

**Phase 5 — MPI_Reduce:**  
Each rank computes the sum of absolute errors for its own test entries. `MPI_Reduce(MPI_SUM)` collects all partial sums into Rank 0, which computes and prints the global MAE.

---

## Diagram 5 — CUDA (GPU Massively Parallel Computing)
**File:** `diagram_05_cuda.drawio`

### Description
CUDA was applied to exploit the massively parallel architecture of NVIDIA GPUs. Unlike CPU parallelism (which uses tens of threads), CUDA launches thousands of threads organised into a two-level grid/block hierarchy.

**How parallel programming concepts were applied:**

**Thread Hierarchy:**  
Threads are grouped into blocks (16×16 = 256 threads per block). Blocks are grouped into a 2D grid (⌈N/16⌉ × ⌈N/16⌉ = 63×63 = 3,969 blocks for N=1000). Each thread identifies itself using `blockIdx` and `threadIdx` and computes its assigned (u, v) pair.

**Three CUDA Kernels:**

1. **`kernel_user_means`** — 1D grid, one thread per user. Each thread scans all N_ITEMS items and computes the mean for its user. Launch: `<<<(N+255)/256, 256>>>`.

2. **`kernel_similarity`** — 2D grid, one thread per (u, v) pair. Threads in the upper triangle (u < v) compute the full Pearson formula and write both `SIM(u,v)` and `SIM(v,u)`. Threads with u ≥ v return immediately (no warp divergence at scale). Launch: `dim3 grid(⌈N/16⌉, ⌈N/16⌉), dim3 block(16, 16)`.

3. **`kernel_predictions`** — 2D grid (users × items), one thread per (user, item) prediction. Each thread maintains a TOP-K=20 min-heap in registers (`top_sim[20]`, `top_rat[20]`, `top_mu[20]`) since `qsort` is unavailable in device code.

**Memory Hierarchy:**  
- Registers: per-thread scalars and TOP-K arrays (sub-cycle latency)
- Global Memory: all large arrays (`d_ratings`, `d_sim`, `d_pred`) — 300–600 cycle latency
- Host-to-Device transfers via `cudaMemcpy` over PCIe before kernels; Device-to-Host after

**Asynchronous Timing:**  
Kernel launches return immediately to the CPU. `cudaEventRecord()` inserts GPU-side timestamps; `cudaEventSynchronize()` blocks the CPU until the GPU finishes before reading elapsed time. Standard `clock_gettime()` would measure near-zero time since the CPU doesn't wait.

---

## Diagram 6 — Hybrid MPI + OpenMP (Two-Level Parallelism)
**File:** `diagram_06_hybrid_mpi_openmp.drawio`

### Description
The Hybrid implementation combines MPI for coarse-grain inter-node distribution and OpenMP for fine-grain intra-node threading. This two-level parallelism is the most realistic model for modern HPC clusters where each node has multiple CPU cores and multiple nodes are connected by a high-speed network.

**How parallel programming concepts were applied:**

**Two-Level Decomposition:**  
- **Level 1 (MPI — coarse grain):** P MPI ranks partition users across nodes. Each rank owns a contiguous user range (e.g., 2 ranks → Rank 0: users 0–499, Rank 1: users 500–999).
- **Level 2 (OpenMP — fine grain):** Within each rank, T OpenMP threads further sub-divide the local user range. With T=4, Rank 0's 4 threads each handle 125 rows (0–124, 125–249, 250–374, 375–499).

**Thread Safety — MPI_THREAD_FUNNELED:**  
`MPI_Init_thread(MPI_THREAD_FUNNELED)` is called instead of `MPI_Init`. This guarantees that MPI calls are made only from the main thread (never from inside an OpenMP parallel region). After the OpenMP parallel section completes its implicit barrier, the main thread calls `MPI_Allgatherv` to exchange partial results with other ranks.

**Phase 3 — Similarity (Inner OpenMP, Outer MPI):**
```c
// MPI: each rank owns rows start_u..end_u
// OpenMP: threads sub-divide that range
#pragma omp parallel for schedule(dynamic, 4)
for (u = start_u; u < end_u; u++) {
    for (v = u+1; v < N_USERS; v++) {
        SIM(u,v) = SIM(v,u) = pearson(u, v);
    }
}
// After OMP barrier, main thread communicates:
MPI_Allgatherv(...);
```

**Phase 5 — Two-Level Reduction:**  
OpenMP `reduction(+:local_err)` merges per-thread error accumulators within each rank. `MPI_Reduce(MPI_SUM)` then sums local errors from all ranks into Rank 0 for the global MAE.

**Four Benchmarked Configurations (all 8 total workers):**

| Config | MPI Ranks (P) | OMP Threads (T) | Characteristic |
|--------|--------------|-----------------|----------------|
| 2×4 | 2 | 4 | Fewer MPI messages, more shared-mem parallelism |
| 4×2 | 4 | 2 | Balanced — moderate network + moderate threading |
| 8×1 | 8 | 1 | Equivalent to pure MPI (no OpenMP benefit) |
| 1×8 | 1 | 8 | Equivalent to pure OpenMP (no MPI distribution) |

These four configurations explore the trade-off between network communication overhead (increases with P) and shared-memory thread scalability (increases with T).

---

## Diagram 7 — Hybrid MPI + Pthreads (Two-Level Parallelism, POSIX inner)
**File:** `diagram_07_mpi_pthreads.drawio`

### Description
The second hybrid variant replaces the OpenMP inner layer with raw POSIX threads. The outer MPI structure is unchanged (same rank-based row partitioning, same `MPI_Allgatherv` / `MPI_Reduce` collectives, same `MPI_THREAD_FUNNELED` contract), but the in-node parallelism is now done with explicit `pthread_create` / `pthread_join` instead of `#pragma omp parallel for`.

**How parallel programming concepts were applied:**

**Two-Level Decomposition:**
- **Level 1 (MPI — coarse grain):** $P$ MPI ranks partition users across nodes via `get_range(rank, size, ...)`, identical to the OpenMP hybrid (e.g., 2 ranks → Rank 0: users 0–499, Rank 1: users 500–999).
- **Level 2 (Pthreads — fine grain):** Within each rank, the `run_parallel(fn, threads, args, start, end)` harness spawns $T$ pthreads. Each thread receives a `ThreadArgs` struct carrying its `tid`, `nthreads`, and the *rank's* row window (`rank_start`, `rank_end`), then derives its own slice via `work_range(rank_end - rank_start, tid, nthreads, ...)`. This re-uses the same static ceiling-division partition as the pure-pthreads version, but rebased to the rank's local range so the total worker count is exactly $P \times T$.

**Thread Safety — MPI_THREAD_FUNNELED:**
`MPI_Init_thread(MPI_THREAD_FUNNELED)` is used, identical to the OpenMP hybrid. Pthreads spawned inside `run_parallel` never call MPI; all collective calls (`MPI_Allgatherv` after Phases 2 and 3, `MPI_Reduce` after Phase 5) happen on the main thread *after* `pthread_join` completes.

**Phase 3 — Similarity (Inner Pthreads, Outer MPI):**
```c
// Per-phase pthread fan-out / join (no persistent thread pool, unlike OMP)
run_parallel(thread_similarities, threads, args, start_u, end_u);
// After pthread_join barrier, main thread communicates:
MPI_Allgatherv(&sim_matrix[start_u * N_USERS], ...);
```
Each pthread fills its slice of `SIM(u, v)` for $u$ in its sub-range of the rank's local window and all columns $v$. The fill pattern matches the OpenMP hybrid exactly so the resulting per-rank block layout is bit-identical and the similarity checksum stays at $942.387323$.

**Phase 4 — Predictions:**
Each pthread allocates its own private `SimPair *nbrs` buffer inside the thread function and frees it before returning (same race-free pattern as pure pthreads), then runs the TOP-K weighted average over its slice. No MPI call is needed because every rank already holds the complete $N \times N$ similarity matrix after the Phase 3 `MPI_Allgatherv`.

**Phase 5 — Per-Rank Sequential + MPI_Reduce:**
The test-set filter loop runs sequentially per rank (only ≈30k entries, threading overhead would dominate), then `MPI_Reduce(MPI_SUM)` collects local error sums into rank 0 for the global MAE/RMSE.

**Difference vs. MPI+OpenMP:**
- *Thread lifecycle:* pthreads are created and joined three separate times (Phases 2, 3, 4) — there is no persistent thread pool to reuse, so OS thread-creation overhead is paid per phase.
- *Load balancing:* the inner partition is static ceiling-division (like pure pthreads), so the triangular work imbalance in Phase 3 cannot be smoothed out the way `schedule(dynamic, 4)` does for the OpenMP hybrid. This is the main mechanical reason the two hybrids differ in measured runtime at the same $P \times T$.
- *Build:* `mpicc -O2 -Wall -o hybrid_pt_rec hybrid_pthreads_recommender.c -lpthread -lm` (no `-fopenmp`).
- *Launch:* `mpirun -np 2 ./hybrid_pt_rec 1000 1000 4` (threads passed as the 3rd CLI arg, matching the pure-pthreads convention, rather than via `OMP_NUM_THREADS`).

**Four Benchmarked Configurations (all 8 total workers):**

| Config | MPI Ranks (P) | PT Threads (T) | Characteristic |
|--------|---------------|----------------|----------------|
| 2×4 | 2 | 4 | Fewer MPI messages, more shared-mem parallelism |
| 4×2 | 4 | 2 | Balanced — moderate network + moderate threading |
| 8×1 | 8 | 1 | Equivalent to pure MPI (no inner threading) |
| 1×8 | 1 | 8 | Equivalent to pure pthreads (no MPI distribution) |

These mirror the OpenMP-hybrid configurations exactly so the two variants can be compared head-to-head at the same total worker count.

---

## How to Open Diagrams

1. Go to **app.diagrams.net** (free, no login required)
2. **File → Import from → Device**
3. Select any `.drawio` file from this folder
4. Export for report: **File → Export as → PNG** or **SVG**

### File Summary

| File | Diagram | Parallel Concept |
|------|---------|-----------------|
| `diagram_01_algorithm_pipeline.drawio` | 5-Phase Pipeline | Algorithm overview — what is parallelised |
| `diagram_02_openmp.drawio` | OpenMP Fork-Join | Shared memory, directives, dynamic scheduling |
| `diagram_03_pthreads.drawio` | Pthreads Lifecycle | Manual thread creation, join-as-barrier |
| `diagram_04_mpi.drawio` | MPI Rank Swimlanes | Distributed memory, Allgatherv, Reduce |
| `diagram_05_cuda.drawio` | CUDA Grid/Block/Thread | GPU hierarchy, memory hierarchy, 3 kernels |
| `diagram_06_hybrid_mpi_openmp.drawio` | Hybrid Two-Level (OMP inner) | MPI + OpenMP combined, FUNNELED safety |
| `diagram_07_mpi_pthreads.drawio` | Hybrid Two-Level (PT inner) | MPI + Pthreads combined, FUNNELED safety |
