# Project Next Steps — Parallel Pearson Correlation Recommender
## EC7207: High Performance Computing | Group 11

---

## Current Status

| Version     | File                              | Status      |
|-------------|-----------------------------------|-------------|
| Serial      | `serial/serial_recommender.c`    | ✅ Done     |
| OpenMP      | `openmp/openmp_recommender.c`    | ✅ Done     |
| Pthreads    | `pthreads/pthreads_recommender.c`| ❌ Missing  |
| MPI         | `mpi/mpi_recommender.c`          | ❌ Missing  |
| CUDA        | `cuda/cuda_recommender.cu`       | ❌ Missing  |
| Hybrid      | `hybrid/hybrid_recommender.c`    | ❌ Missing  |
| Benchmark   | `results/`                       | ❌ Missing  |

---

## Step 1 — Pthreads Version (POSIX Threads)

**Why it matters:** Pthreads is a lower-level shared-memory model than OpenMP.
You manage threads, barriers, and synchronisation manually. This demonstrates
you understand what OpenMP abstracts away.

### Folder & file

```
pthreads/pthreads_recommender.c
```

### Compile & run

```bash
gcc -O2 -Wall -o pthreads_rec pthreads/pthreads_recommender.c -lpthread -lm
./pthreads_rec              # default 1000 x 1000
./pthreads_rec 2000 1500 8  # 2000 users, 1500 items, 8 threads
```

### What to implement (phase by phase)

Follow the same 5-phase structure as the serial and OpenMP versions.
Add a third CLI argument for the number of threads (default 4).

**Thread launch pattern (use this everywhere):**

```c
#define MAX_THREADS 64

typedef struct {
    int tid;
    int nthreads;
} ThreadArgs;

pthread_t threads[MAX_THREADS];
ThreadArgs args[MAX_THREADS];
pthread_barrier_t barrier;   // declare globally

// in main, before any parallel section:
pthread_barrier_init(&barrier, NULL, nthreads);
```

**Phase 2 – User means (parallel):**
Each thread processes rows `[tid * chunk .. (tid+1) * chunk)` of the ratings
matrix and writes its own slice of `user_mean[]`. No synchronisation needed
because each thread writes to a disjoint range.

```c
void *thread_user_means(void *arg) {
    ThreadArgs *a = (ThreadArgs *)arg;
    int chunk = (N_USERS + a->nthreads - 1) / a->nthreads;
    int start = a->tid * chunk;
    int end   = (start + chunk < N_USERS) ? start + chunk : N_USERS;
    for (int u = start; u < end; u++) { /* same logic as serial */ }
    return NULL;
}
```

**Phase 3 – Similarity matrix (parallel, upper triangle):**
The upper triangle has `N*(N-1)/2` pairs. Distribute rows of the outer loop
across threads using the same chunk approach.

```c
void *thread_similarities(void *arg) {
    ThreadArgs *a = (ThreadArgs *)arg;
    int chunk = (N_USERS + a->nthreads - 1) / a->nthreads;
    int start = a->tid * chunk;
    int end   = (start + chunk < N_USERS) ? start + chunk : N_USERS;
    for (int u = start; u < end; u++) {
        SIM(u, u) = 1.0f;
        for (int v = u + 1; v < N_USERS; v++) {
            float s = pearson_similarity(u, v);
            SIM(u, v) = s;
            SIM(v, u) = s;
        }
    }
    return NULL;
}
```

> Note: Each thread writes to a different row `u` of `sim_matrix`, and also
> writes `SIM(v,u)` for `v > u`. Because `v > u`, these writes are in rows
> that other threads may also write to simultaneously. To avoid a race, either:
> - Do a second pass `SIM(v,u) = SIM(u,v)` after a barrier, or
> - Only fill the upper triangle during threading, then do a serial copy.

**Phase 4 – Predictions (parallel):**
Give each thread a contiguous block of users. Each thread needs its own
private `SimPair` buffer (thread-local malloc, freed at the end of the thread).

**Barrier usage:**
Use `pthread_barrier_wait(&barrier)` between phases to ensure that, for example,
all user means are complete before any thread starts computing similarities.

**Timing:**
Use `clock_gettime(CLOCK_MONOTONIC, ...)` exactly as in the serial version
(not `omp_get_wtime`).

---

## Step 2 — MPI Version (Distributed Memory)

**Why it matters:** MPI runs across multiple nodes/processes with no shared
memory. Each process has its own address space. This is the distributed-memory
part of the proposal.

### Folder & file

```
mpi/mpi_recommender.c
```

### Compile & run

```bash
mpicc -O2 -Wall -o mpi_rec mpi/mpi_recommender.c -lm
mpirun -np 1  ./mpi_rec
mpirun -np 2  ./mpi_rec
mpirun -np 4  ./mpi_rec
mpirun -np 8  ./mpi_rec
# Custom sizes
mpirun -np 4  ./mpi_rec 2000 1500
```

### Data distribution strategy

The ratings matrix is `N_USERS × N_ITEMS`. Distribute rows (users) across
processes. Process `rank` owns users in `[start_u .. end_u)`:

```
int chunk = N_USERS / size;          // size = total MPI processes
int start_u = rank * chunk;
int end_u   = (rank == size-1) ? N_USERS : start_u + chunk;
int local_rows = end_u - start_u;
```

### What to implement (phase by phase)

**Initialisation:**

```c
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
```

**Phase 1 – Data generation:**
Only rank 0 calls `generate_data()` to fill the full ratings matrix.
Then rank 0 broadcasts/scatters data to all other processes.

- `MPI_Bcast` the full ratings matrix (simple but uses more memory per node).
- Or `MPI_Scatterv` only the rows each process needs (more efficient but
  requires the full similarity computation to also be distributed — harder).

**Recommended approach for correctness first:** Broadcast the full matrix.
Every process has all ratings, but only computes similarity for its own rows.

```c
// rank 0 fills ratings[], all others have zeros
MPI_Bcast(ratings, N_USERS * N_ITEMS, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Bcast(user_mean, N_USERS, MPI_FLOAT, 0, MPI_COMM_WORLD);
```

**Phase 2 – User means:**
Each process computes means only for its `[start_u .. end_u)` slice, then
uses `MPI_Allgather` so every process has the complete `user_mean[]` array
(needed for pearson_similarity across all pairs).

```c
MPI_Allgatherv(local_means, local_rows, MPI_FLOAT,
               user_mean, recvcounts, displs, MPI_FLOAT,
               MPI_COMM_WORLD);
```

**Phase 3 – Similarity matrix:**
Each process computes rows `[start_u .. end_u)` of the similarity matrix
(all `v` values for each `u` it owns). Store only the local rows in a
`local_rows × N_USERS` sub-matrix.

After computing, use `MPI_Allgatherv` to assemble the full `N_USERS × N_USERS`
similarity matrix on all processes (needed for prediction).

**Phase 4 – Predictions:**
Each process predicts ratings only for its `[start_u .. end_u)` users.
After predictions, `MPI_Gatherv` the predictions to rank 0 for evaluation.

**Phase 5 – Evaluation (MAE):**
Rank 0 collects all predictions and computes MAE. Only rank 0 prints results.
Guard all `printf` with `if (rank == 0)`.

**Timing:**
Use `MPI_Wtime()` for wall-clock time. Measure phases on rank 0 only.

```c
double t0 = MPI_Wtime();
compute_all_similarities();
MPI_Barrier(MPI_COMM_WORLD);   // synchronise before stopping timer
double t1 = MPI_Wtime();
```

**Cleanup:**

```c
MPI_Finalize();
```

---

## Step 3 — CUDA Version (GPU)

**Why it matters:** The similarity matrix is the bottleneck —
`O(N² × M)` work. This maps perfectly to the GPU because each
`(u, v)` pair can be computed independently by a separate GPU thread.

### Folder & file

```
cuda/cuda_recommender.cu
```

### Compile & run

```bash
nvcc -O2 -arch=sm_75 -o cuda_rec cuda/cuda_recommender.cu -lm
./cuda_rec
./cuda_rec 2000 1500
```
> Replace `sm_75` with the correct compute capability for your GPU
> (e.g., `sm_86` for RTX 30xx, `sm_89` for RTX 40xx).

### Kernel design

**Phase 3 – Similarity kernel (most important):**

Launch a 2D grid where each thread computes one `(u, v)` pair.

```cuda
__global__ void similarity_kernel(float *ratings, float *user_mean,
                                   float *sim_matrix,
                                   int N_USERS, int N_ITEMS)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= N_USERS || v >= N_USERS) return;
    if (u == v) { sim_matrix[u*N_USERS+v] = 1.0f; return; }
    if (u > v)  { return; }   // compute upper triangle only; mirror below

    // ... pearson similarity logic using ratings, user_mean ...

    sim_matrix[u*N_USERS+v] = s;
    sim_matrix[v*N_USERS+u] = s;
}
```

**Launch configuration:**

```cuda
dim3 block(16, 16);    // 256 threads per block
dim3 grid((N_USERS + 15) / 16, (N_USERS + 15) / 16);
similarity_kernel<<<grid, block>>>(d_ratings, d_user_mean, d_sim, N_USERS, N_ITEMS);
cudaDeviceSynchronize();
```

**Phase 4 – Prediction kernel:**

Launch one thread per `(user, item)` pair.

```cuda
__global__ void prediction_kernel(float *ratings, float *user_mean,
                                   float *sim_matrix, float *predictions,
                                   int N_USERS, int N_ITEMS, int TOP_K)
{
    int u    = blockIdx.x * blockDim.x + threadIdx.x;
    int item = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= N_USERS || item >= N_ITEMS) return;
    // ... neighbour selection and weighted average ...
}
```

> For TOP_K selection on GPU: a simple approach is to sort all neighbours
> by similarity (use `thrust::sort`) or do a partial reduction in shared
> memory to pick the top K.

**Memory management (boilerplate):**

```cuda
float *d_ratings, *d_user_mean, *d_sim, *d_pred;
cudaMalloc(&d_ratings,  N_USERS * N_ITEMS * sizeof(float));
cudaMalloc(&d_user_mean, N_USERS * sizeof(float));
cudaMalloc(&d_sim,      N_USERS * N_USERS * sizeof(float));
cudaMalloc(&d_pred,     N_USERS * N_ITEMS * sizeof(float));

cudaMemcpy(d_ratings, ratings, ..., cudaMemcpyHostToDevice);
// ... run kernels ...
cudaMemcpy(predictions, d_pred, ..., cudaMemcpyDeviceToHost);

cudaFree(d_ratings); cudaFree(d_user_mean);
cudaFree(d_sim);     cudaFree(d_pred);
```

**Timing:**
Use CUDA events to measure GPU kernel time (excludes host-device transfer),
and also report total time including transfers.

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start);
similarity_kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("[Timing] Similarity kernel  : %.4f s\n", ms / 1000.0f);
```

---

## Step 4 — Hybrid Version (MPI + OpenMP)

**Why it matters:** Real HPC clusters combine distributed nodes (MPI) with
multi-core processors (OpenMP). This is the most advanced version in the proposal.

### Folder & file

```
hybrid/hybrid_recommender.c
```

### Compile & run

```bash
mpicc -O2 -Wall -fopenmp -o hybrid_rec hybrid/hybrid_recommender.c -lm
mpirun -np 2 --bind-to socket env OMP_NUM_THREADS=4 ./hybrid_rec
mpirun -np 4 env OMP_NUM_THREADS=2 ./hybrid_rec 2000 1500
```

### Design

Take the MPI version and add `#pragma omp parallel for` inside each
MPI process's work loops. The key insight:

- **MPI** divides users across processes (nodes).
- **OpenMP** parallelises the inner loops within each process (cores).

**Similarity phase (hybrid):**

```c
// Each MPI process handles rows [start_u .. end_u)
// Within that range, OpenMP threads share the rows further
#pragma omp parallel for schedule(dynamic, 4)
for (int u = start_u; u < end_u; u++) {
    SIM(u, u) = 1.0f;
    for (int v = u + 1; v < N_USERS; v++) {
        float s = pearson_similarity(u, v);
        SIM(u, v) = s;
        SIM(v, u) = s;
    }
}
MPI_Barrier(MPI_COMM_WORLD);
MPI_Allgatherv(/* local rows */ ... );
```

**Print thread/process info in header:**

```c
if (rank == 0) {
    int nthreads;
    #pragma omp parallel
    { #pragma omp single
      nthreads = omp_get_num_threads(); }
    printf("    MPI Processes: %d | OpenMP Threads/Process: %d\n",
           size, nthreads);
}
```

---

## Step 5 — Benchmarking & Results

Create a `results/` folder. Run all versions at multiple scales and record
the key timings. Below is the exact table to fill for your report.

### Suggested input sizes

| Label  | N_USERS | N_ITEMS |
|--------|---------|---------|
| Small  |   500   |   500   |
| Medium |  1000   |  1000   |
| Large  |  2000   |  1500   |

### Metric 1 — Speedup (Strong Scaling)

Fix problem size at **1000 × 1000**. Vary thread/process count.

| Version   | Threads/Procs | Sim Time (s) | Pred Time (s) | Total (s) | Speedup |
|-----------|---------------|-------------|---------------|-----------|---------|
| Serial    | 1             |             |               |           | 1.00    |
| OpenMP    | 2             |             |               |           |         |
| OpenMP    | 4             |             |               |           |         |
| OpenMP    | 8             |             |               |           |         |
| Pthreads  | 2             |             |               |           |         |
| Pthreads  | 4             |             |               |           |         |
| Pthreads  | 8             |             |               |           |         |
| MPI       | 2             |             |               |           |         |
| MPI       | 4             |             |               |           |         |
| CUDA      | GPU           |             |               |           |         |
| Hybrid    | 2×4           |             |               |           |         |

**Speedup formula:**
```
Speedup(p) = T_serial / T_parallel(p)
```

### Metric 2 — Efficiency

```
Efficiency(p) = Speedup(p) / p
```

Efficiency close to 1.0 = good parallelism. Efficiency dropping fast = overhead.

### Metric 3 — Correctness (MAE agreement)

All versions must produce the **same MAE** (within floating-point rounding,
i.e., ±0.001) as the serial version for the same input size and seed.
If the MAE differs significantly, there is a bug (race condition, incorrect
data distribution, etc.).

| Version  | MAE (1000×1000) | Matches Serial? |
|----------|-----------------|-----------------|
| Serial   |                 | baseline        |
| OpenMP   |                 |                 |
| Pthreads |                 |                 |
| MPI      |                 |                 |
| CUDA     |                 |                 |
| Hybrid   |                 |                 |

### Similarity checksum

Every version already prints `[Check] Sim-matrix checksum`. This must also
match across all versions for the same input. Use it as a fast sanity check.

---

## Step 6 — Recommended Implementation Order

Work in this order — each step builds on the previous:

```
1. Pthreads   → shared memory, manual control   (1–2 days)
2. MPI        → distributed memory              (2–3 days)
3. CUDA       → GPU kernel                      (2–3 days)
4. Hybrid     → MPI + OpenMP                    (1 day, combines steps 2+serial OpenMP)
5. Benchmarks → run all, fill the tables above  (1 day)
6. Report     → analysis, plots, conclusion     (1–2 days)
```

---

## Step 7 — Project Folder Structure to Target

```
hpc-pearson-correlation/
├── serial/
│   └── serial_recommender.c          ✅ Done
├── openmp/
│   └── openmp_recommender.c          ✅ Done
├── pthreads/
│   └── pthreads_recommender.c        ← Next
├── mpi/
│   └── mpi_recommender.c             ← After Pthreads
├── cuda/
│   └── cuda_recommender.cu           ← After MPI
├── hybrid/
│   └── hybrid_recommender.c          ← After CUDA
├── results/
│   ├── timing_1000x1000.txt          ← Raw outputs
│   ├── speedup_table.csv             ← For report plots
│   └── mae_comparison.txt            ← Correctness check
├── README.md                          ✅ Done
└── RECOMMENDATION.md                  ✅ This file
```

---

## Key Things to Keep Identical Across All Versions

1. **Same `#define` constants** — `SPARSITY 0.70`, `TOP_K 20`, `SEED 42`, `TEST_RATIO 0.10`
2. **Same `generate_data()` logic** — serial RNG with `srand(42)`, so all versions
   produce the identical dataset.
3. **Same `pearson_similarity()` formula** — copy it verbatim; only the
   loop that *calls* it changes.
4. **Same output format** — `[Timing]`, `[Check]`, `[Eval]` lines so results
   are easy to compare side-by-side.
