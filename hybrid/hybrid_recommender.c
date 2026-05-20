/*
 * hybrid_recommender.c
 * Pearson Correlation Recommender – Hybrid MPI + OpenMP Version
 *
 * ╔═════════════════════════════════════════════════════════════════════════╗
 * ║  WHAT IS THE HYBRID MODEL?                                              ║
 * ║                                                                         ║
 * ║  Real HPC clusters are built from many compute nodes connected by a    ║
 * ║  high-speed network (e.g. InfiniBand).  Each node is itself a multi-   ║
 * ║  core processor (8, 16, 32+ cores) sharing one block of RAM.           ║
 * ║                                                                         ║
 * ║  A pure-MPI program launches one process per core, which wastes        ║
 * ║  bandwidth: cores on the SAME node must go through the MPI layer        ║
 * ║  even though they could read each other's memory directly.              ║
 * ║                                                                         ║
 * ║  A pure-OpenMP program cannot cross node boundaries at all.            ║
 * ║                                                                         ║
 * ║  The Hybrid model uses BOTH:                                            ║
 * ║                                                                         ║
 * ║   ┌───────────────────────────────────────────────────────────────┐    ║
 * ║   │  CLUSTER                                                       │    ║
 * ║   │  ┌─────────────────┐        ┌─────────────────┐              │    ║
 * ║   │  │  Node 0          │  MPI   │  Node 1          │             │    ║
 * ║   │  │  Rank 0          │◄──────►│  Rank 1          │             │    ║
 * ║   │  │  ┌─┐ ┌─┐ ┌─┐ ┌─┐│        │  ┌─┐ ┌─┐ ┌─┐ ┌─┐│             │    ║
 * ║   │  │  │T│ │T│ │T│ │T││        │  │T│ │T│ │T│ │T││             │    ║
 * ║   │  │  │0│ │1│ │2│ │3││        │  │0│ │1│ │2│ │3││             │    ║
 * ║   │  │  └─┘ └─┘ └─┘ └─┘│        │  └─┘ └─┘ └─┘ └─┘│             │    ║
 * ║   │  │  OpenMP threads  │        │  OpenMP threads  │             │    ║
 * ║   │  └─────────────────┘        └─────────────────┘              │    ║
 * ║   └───────────────────────────────────────────────────────────────┘    ║
 * ║                                                                         ║
 * ║  ● MPI  divides work at the COARSE grain: different ranks own          ║
 * ║    different rows (users) of the similarity matrix.  Each rank         ║
 * ║    has its own private address space — data is exchanged via           ║
 * ║    collective operations (MPI_Allgatherv, MPI_Reduce).                 ║
 * ║                                                                         ║
 * ║  ● OpenMP divides work at the FINE grain: inside each MPI rank,       ║
 * ║    multiple threads cooperate over the rank's own user rows using      ║
 * ║    shared memory.  No extra MPI messages are needed — threads can      ║
 * ║    read each other's sim_matrix values directly.                        ║
 * ║                                                                         ║
 * ║  RESULT: with P MPI ranks and T OpenMP threads each, the program       ║
 * ║  uses P × T workers total while sending only P messages per collective  ║
 * ║  instead of (P × T) messages in a flat MPI model.                      ║
 * ╚═════════════════════════════════════════════════════════════════════════╝
 *
 * ╔═════════════════════════════════════════════════════════════════════════╗
 * ║  TWO-LEVEL PARALLELISM IN EACH PHASE                                    ║
 * ║                                                                         ║
 * ║  Phase 1  Data gen    Serial on every rank (same seed → same data).    ║
 * ║  Phase 2  Means       MPI rows → OpenMP loop over rows in range.       ║
 * ║  Phase 3  Similarity  MPI rows → OpenMP dynamic loop over rows.        ║
 * ║                       MPI_Allgatherv assembles global matrix.           ║
 * ║  Phase 4  Predictions MPI rows → OpenMP loop, private nbrs buffer.     ║
 * ║  Phase 5  MAE         OpenMP reduction inside rank → MPI_Reduce.       ║
 * ╚═════════════════════════════════════════════════════════════════════════╝
 *
 * Compile:
 *   mpicc -O2 -Wall -fopenmp -o hybrid_rec hybrid/hybrid_recommender.c -lm
 *
 * Run:
 *   # 2 MPI ranks, 4 OpenMP threads each = 8 workers total
 *   mpirun -np 2 env OMP_NUM_THREADS=4 ./hybrid_rec
 *
 *   # 4 MPI ranks, 2 OpenMP threads each = 8 workers total
 *   mpirun -np 4 env OMP_NUM_THREADS=2 ./hybrid_rec
 *
 *   # Custom problem size
 *   mpirun -np 2 env OMP_NUM_THREADS=4 ./hybrid_rec 2000 1500
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>   /* MPI: inter-node message passing                         */
#include <omp.h>   /* OpenMP: intra-node thread parallelism                   */

/* ── Fixed algorithm parameters (identical across all project versions) ─── */
#define DEFAULT_USERS  1000
#define DEFAULT_ITEMS  1000
#define SPARSITY       0.70f
#define TOP_K            20
#define SEED             42
#define TEST_RATIO      0.10f

/* ── Runtime sizes and MPI identity (set in main) ───────────────────────── */
static int N_USERS;
static int N_ITEMS;
static int mpi_rank;  /* this rank's 0-based index within MPI_COMM_WORLD     */
static int mpi_size;  /* total number of MPI processes                       */

/*
 * Data arrays – every rank allocates the full matrices.
 *
 * Allocating full arrays on every rank avoids the need to route global
 * index lookups through gather/scatter during computation.  The memory cost
 * is N_USERS×N_ITEMS×4 bytes per rank (≈ 4 GB for 32k×32k — fine for
 * our 1k×1k–2k×1.5k benchmark sizes).
 */
static float *ratings;      /* [N_USERS × N_ITEMS]  0 = unrated              */
static float *user_mean;    /* [N_USERS]                                      */
static float *sim_matrix;   /* [N_USERS × N_USERS]                           */
static float *predictions;  /* [N_USERS × N_ITEMS]                           */

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;

/* ── 2-D index macros ────────────────────────────────────────────────────── */
#define R(u,i)    ratings[(size_t)(u)*N_ITEMS  + (i)]
#define SIM(u,v)  sim_matrix[(size_t)(u)*N_USERS + (v)]
#define PRED(u,i) predictions[(size_t)(u)*N_ITEMS + (i)]

/* Use MPI's monotonic wall-clock timer everywhere */
static inline double now_sec(void) { return MPI_Wtime(); }

/* ── Row-range helper ────────────────────────────────────────────────────── */
/*
 * get_range() – compute [start, end) row slice owned by a given MPI rank.
 *
 * Ceiling division (N_USERS + size - 1) / size guarantees that all N_USERS
 * rows are covered even when the division is uneven.  The last rank receives
 * at most (chunk - 1) fewer rows than the others.
 */
static void get_range(int rank, int size, int *start, int *end)
{
    int chunk = (N_USERS + size - 1) / size;
    *start = rank * chunk;
    *end   = *start + chunk;
    if (*end > N_USERS) *end = N_USERS;
}

/* ── Memory management ───────────────────────────────────────────────────── */
static void alloc_arrays(void)
{
    ratings     = (float *)calloc((size_t)N_USERS * N_ITEMS, sizeof(float));
    user_mean   = (float *)calloc(N_USERS,                    sizeof(float));
    sim_matrix  = (float *)calloc((size_t)N_USERS * N_USERS,  sizeof(float));
    predictions = (float *)calloc((size_t)N_USERS * N_ITEMS,  sizeof(float));

    if (!ratings || !user_mean || !sim_matrix || !predictions) {
        fprintf(stderr, "Rank %d: allocation failed.\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

static void free_arrays(void)
{
    free(ratings); free(user_mean);
    free(sim_matrix); free(predictions);
    free(test_set);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 1 – Data generation  (serial, every rank, no communication)
 *
 * Every rank runs the same srand(SEED) + rand() sequence and therefore builds
 * identical ratings[] and test_set[] without any MPI calls.  This is cheaper
 * than broadcasting a large array from rank 0 and produces exactly the same
 * numerical results as the serial baseline.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void generate_data(void)
{
    srand(SEED);

    int capacity = (int)((size_t)N_USERS * N_ITEMS * (1.0f - SPARSITY)) + 1000;
    test_set  = (TestEntry *)malloc(capacity * sizeof(TestEntry));
    test_size = 0;

    for (int u = 0; u < N_USERS; u++) {
        for (int i = 0; i < N_ITEMS; i++) {
            if ((float)rand() / RAND_MAX < SPARSITY) continue;
            float rating = (float)(rand() % 5) + 1.0f;
            if ((float)rand() / RAND_MAX < TEST_RATIO && test_size < capacity) {
                test_set[test_size].user   = u;
                test_set[test_size].item   = i;
                test_set[test_size].rating = rating;
                test_size++;
            } else {
                R(u, i) = rating;
            }
        }
    }

    if (mpi_rank == 0)
        printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | "
               "Test ratings: %d\n",
               N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 2 – User means  (OpenMP within rank + MPI_Allgatherv across ranks)
 *
 * ── LEVEL 1 (MPI): row partitioning ──────────────────────────────────────
 *   Each MPI rank is responsible for computing user_mean[u] for the users
 *   u in [start_u, end_u).  Different ranks own disjoint row ranges.
 *
 * ── LEVEL 2 (OpenMP): thread loop over the rank's rows ───────────────────
 *   Inside the rank, #pragma omp parallel for distributes the [start_u, end_u)
 *   loop across T threads.  Each thread writes to a different index of
 *   user_mean[], so there are no data races.
 *
 *   schedule(static) gives each thread a fixed contiguous block of rows.
 *   This is optimal here because every user row has equal work (N_ITEMS
 *   iterations), so no load-balancing is needed.
 *
 * ── MPI_Allgatherv: assemble global user_mean[] on every rank ────────────
 *   After the OpenMP loop, each rank holds its own slice of user_mean[].
 *   MPI_Allgatherv combines all slices so that every rank's user_mean[]
 *   is fully populated.  The full array is needed because Phase 3's
 *   pearson_similarity(u, v) reads user_mean[v] for any v, not just local v.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void compute_user_means(int start_u, int end_u,
                                int *recvcounts, int *displs)
{
    /*
     * OpenMP parallel region: threads share ratings[] and user_mean[] (read-only
     * and write-to-disjoint-indices respectively).  No synchronisation needed.
     */
    #pragma omp parallel for schedule(static)
    for (int u = start_u; u < end_u; u++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < N_ITEMS; i++) {
            if (R(u, i) != 0.0f) { sum += R(u, i); cnt++; }
        }
        user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }
    /* All OpenMP threads have joined before MPI_Allgatherv — omp parallel
     * has an implicit barrier at its closing brace. */

    MPI_Allgatherv(
        &user_mean[start_u], end_u - start_u, MPI_FLOAT,
        user_mean, recvcounts, displs, MPI_FLOAT,
        MPI_COMM_WORLD
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 3 – Similarity matrix  (OpenMP within rank + MPI_Allgatherv)
 *
 * pearson_similarity() is a pure function that reads only ratings[] and
 * user_mean[] — both fully populated and never modified during Phase 3.
 * This makes it safe to call from multiple OpenMP threads simultaneously.
 *
 * ── LEVEL 1 (MPI): each rank computes ALL columns for its row slice ───────
 *   Rank r fills rows [start_u, end_u) of sim_matrix — every column v for
 *   each of those rows.  Computing both SIM(u,v) and SIM(v,u) (rather than
 *   upper-triangle only) doubles the Pearson calls but avoids any write
 *   contention: every thread writes only to rows it owns.
 *
 * ── LEVEL 2 (OpenMP): dynamic scheduling for load balance ─────────────────
 *   schedule(dynamic, 4) lets the OpenMP runtime assign rows to threads
 *   on demand in chunks of 4.  This is important because rows with small u
 *   have fewer neighbours than rows with large u (upper-triangle skew), so
 *   static chunking would leave some threads idle.
 *
 *   ┌── Thread-safety of sim_matrix writes ─────────────────────────────┐
 *   │  Thread t handles rows u ∈ {start_u + t_chunk ...}.              │
 *   │  It writes SIM(u, v) for ALL v — that is, row u of sim_matrix.   │
 *   │  No two threads write to the same row u, so no race condition.   │
 *   └────────────────────────────────────────────────────────────────────┘
 *
 * ── MPI_Allgatherv: assemble full sim_matrix on every rank ───────────────
 *   After OpenMP finishes all threads join the barrier), each rank sends its
 *   completed row block.  Collective assembly gives every rank the full
 *   N_USERS×N_USERS matrix needed for prediction.
 * ═══════════════════════════════════════════════════════════════════════════ */
static float pearson_similarity(int u, int v)
{
    double num = 0.0, den_u = 0.0, den_v = 0.0;
    int    co  = 0;
    float  mu  = user_mean[u], mv = user_mean[v];

    for (int i = 0; i < N_ITEMS; i++) {
        if (R(u, i) != 0.0f && R(v, i) != 0.0f) {
            double du = R(u, i) - mu;
            double dv = R(v, i) - mv;
            num   += du * dv;
            den_u += du * du;
            den_v += dv * dv;
            co++;
        }
    }

    if (co < 2) return 0.0f;
    double denom = sqrt(den_u) * sqrt(den_v);
    if (denom < 1e-10) return 0.0f;

    float s = (float)(num / denom);
    if (s >  1.0f) s =  1.0f;
    if (s < -1.0f) s = -1.0f;
    return s;
}

static void compute_all_similarities(int start_u, int end_u,
                                      int *recvcounts, int *displs)
{
    /*
     * schedule(dynamic, 4): rows near start_u have fewer v-values than rows
     * near end_u (because we compute all v, but the Pearson inner loop varies
     * by co-rated item count).  Dynamic scheduling prevents thread starvation.
     *
     * Each iteration writes only to row u of sim_matrix — no cross-thread
     * writes, so no mutex or atomic is needed.
     */
    #pragma omp parallel for schedule(dynamic, 4)
    for (int u = start_u; u < end_u; u++) {
        SIM(u, u) = 1.0f;
        for (int v = 0; v < N_USERS; v++) {
            if (v == u) continue;
            SIM(u, v) = pearson_similarity(u, v);
        }
    }
    /* Implicit OpenMP barrier here — all threads have written their rows. */

    /*
     * MPI_Allgatherv: every rank sends its (local_rows × N_USERS) float block.
     * recvcounts[r] and displs[r] are pre-scaled by N_USERS (set up in main).
     * After this call sim_matrix is complete and identical on every rank.
     */
    MPI_Allgatherv(
        &sim_matrix[(size_t)start_u * N_USERS],
        (end_u - start_u) * N_USERS, MPI_FLOAT,
        sim_matrix, recvcounts, displs, MPI_FLOAT,
        MPI_COMM_WORLD
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 4 – Predictions  (OpenMP within rank, no MPI communication)
 *
 * After Phase 3, every rank holds the complete sim_matrix[].  Each rank
 * independently predicts ratings for its own users [start_u, end_u) without
 * any MPI calls.
 *
 * ── Thread-private neighbour buffer ──────────────────────────────────────
 *   The neighbour list (nbrs) must NOT be shared among threads — each thread
 *   sorts its own list independently.  The buffer is allocated INSIDE the
 *   omp parallel region, which means each thread allocates its own copy on
 *   its private stack/heap.  It is freed at the end of each thread's work.
 *
 * ── schedule(dynamic, 2) ─────────────────────────────────────────────────
 *   Items vary in how many neighbours they attract.  Dynamic scheduling with
 *   chunk size 2 gives a good balance between scheduling overhead and
 *   preventing thread starvation on sparse-item rows.
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct { int idx; float val; } SimPair;

static int cmp_sim_desc(const void *a, const void *b)
{
    float fa = ((const SimPair *)a)->val;
    float fb = ((const SimPair *)b)->val;
    return (fb > fa) - (fb < fa);
}

static void compute_all_predictions(int start_u, int end_u)
{
    /*
     * The omp parallel region lets each thread allocate its own nbrs buffer.
     * Without this, a shared nbrs would require a mutex around the qsort call,
     * which would serialize the entire prediction phase.
     */
    #pragma omp parallel
    {
        /* Each thread gets its own private heap allocation — zero race risk. */
        SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));
        if (!nbrs) {
            fprintf(stderr, "Rank %d thread %d: malloc failed.\n",
                    mpi_rank, omp_get_thread_num());
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        #pragma omp for schedule(dynamic, 2)
        for (int u = start_u; u < end_u; u++) {
            for (int item = 0; item < N_ITEMS; item++) {

                if (R(u, item) != 0.0f) { PRED(u, item) = R(u, item); continue; }

                int cnt = 0;
                for (int v = 0; v < N_USERS; v++) {
                    if (v == u || R(v, item) == 0.0f) continue;
                    float s = SIM(u, v);
                    if (s <= 0.0f) continue;
                    nbrs[cnt].idx = v;
                    nbrs[cnt].val = s;
                    cnt++;
                }

                if (cnt == 0) { PRED(u, item) = user_mean[u]; continue; }

                qsort(nbrs, cnt, sizeof(SimPair), cmp_sim_desc);
                int k = (cnt < TOP_K) ? cnt : TOP_K;

                double num = 0.0, den = 0.0;
                for (int j = 0; j < k; j++) {
                    float s = nbrs[j].val;
                    num += s * (R(nbrs[j].idx, item) - user_mean[nbrs[j].idx]);
                    den += s;
                }

                float pred = (den > 1e-10)
                             ? user_mean[u] + (float)(num / den)
                             : user_mean[u];
                if (pred < 1.0f) pred = 1.0f;
                if (pred > 5.0f) pred = 5.0f;
                PRED(u, item) = pred;
            }
        }
        /* Implicit OpenMP barrier — all threads done before free(). */

        free(nbrs);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 5 – MAE evaluation  (OpenMP reduction inside rank + MPI_Reduce)
 *
 * Two-level reduction mirrors the two-level parallelism:
 *
 *   Level 2 (OpenMP): threads within a rank each accumulate a partial error
 *     sum over the test entries belonging to that rank's user rows.
 *     The "reduction(+:local_err, local_cnt)" clause automatically generates
 *     private copies of the variables for each thread and sums them safely
 *     into the shared variables at the end of the parallel region.
 *
 *   Level 1 (MPI): MPI_Reduce sums the per-rank totals to rank 0, which
 *     computes and prints the final global MAE.
 * ═══════════════════════════════════════════════════════════════════════════ */
static float evaluate_mae(int start_u, int end_u)
{
    double local_err = 0.0;
    int    local_cnt = 0;

    /*
     * reduction(+:local_err, local_cnt): each OpenMP thread maintains private
     * copies of both variables.  At the end of the parallel region the runtime
     * sums all private copies into the originals — equivalent to a lock-free
     * parallel sum.
     */
    #pragma omp parallel for reduction(+:local_err, local_cnt) schedule(static)
    for (int t = 0; t < test_size; t++) {
        int u = test_set[t].user;
        if (u < start_u || u >= end_u) continue;
        local_err += fabs(PRED(u, test_set[t].item) - test_set[t].rating);
        local_cnt++;
    }
    /* OpenMP barrier: local_err and local_cnt are now the rank-wide totals. */

    double global_err = 0.0;
    int    global_cnt = 0;

    /* MPI_Reduce: sum all ranks' local_err / local_cnt values to rank 0. */
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0 && global_cnt > 0)
        return (float)(global_err / global_cnt);
    return 0.0f;
}

/* RMSE — same two-level reduction (OpenMP within rank, MPI_Reduce across
 * ranks) accumulating squared error.  Collective: called by every rank. */
static float evaluate_rmse(int start_u, int end_u)
{
    double local_sq  = 0.0;
    int    local_cnt = 0;

    #pragma omp parallel for reduction(+:local_sq, local_cnt) schedule(static)
    for (int t = 0; t < test_size; t++) {
        int u = test_set[t].user;
        if (u < start_u || u >= end_u) continue;
        double d = PRED(u, test_set[t].item) - test_set[t].rating;
        local_sq += d * d;
        local_cnt++;
    }

    double global_sq  = 0.0;
    int    global_cnt = 0;

    MPI_Reduce(&local_sq,  &global_sq,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0 && global_cnt > 0)
        return (float)sqrt(global_sq / global_cnt);
    return 0.0f;
}

/* Checksum over the full sim_matrix — run on rank 0 after Allgatherv. */
static double similarity_checksum(void)
{
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += SIM(u, v);
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main()
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    /*
     * MPI_Init_thread requests MPI support for multi-threaded use.
     *
     * ── Why MPI_Init_thread instead of MPI_Init? ─────────────────────────
     *
     * When OpenMP threads are active inside an MPI rank, MPI_Allgatherv is
     * called only from the main thread (never from inside a parallel region).
     * This satisfies the MPI_THREAD_FUNNELED level:
     *
     *   MPI_THREAD_SINGLE    → only one thread exists         (pure MPI)
     *   MPI_THREAD_FUNNELED  → MPI called from main thread only (our case)
     *   MPI_THREAD_SERIALIZED → MPI calls serialised by the app
     *   MPI_THREAD_MULTIPLE  → any thread may call MPI at any time
     *
     * Requesting MPI_THREAD_FUNNELED is sufficient for this program and
     * allows MPI implementations to apply optimisations not possible at
     * higher thread-safety levels.
     *
     * 'provided' reports what the MPI library actually supports; we only
     * warn if it is below what we requested.
     */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED && mpi_rank == 0)
        fprintf(stderr, "Warning: MPI thread support below FUNNELED (%d < %d).\n",
                provided, MPI_THREAD_FUNNELED);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    N_USERS = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;

    if (N_USERS <= 0 || N_ITEMS <= 0) {
        if (mpi_rank == 0)
            fprintf(stderr, "Usage: %s [num_users] [num_items]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* ── Compute this rank's row range ─────────────────────────────────── */
    int start_u, end_u;
    get_range(mpi_rank, mpi_size, &start_u, &end_u);

    /*
     * omp_get_max_threads() returns the number of threads that would be
     * created if a parallel region opened now with no num_threads clause.
     * This is determined by OMP_NUM_THREADS (or a runtime default).
     * We read it on rank 0 before any parallel region opens.
     */
    int omp_threads = omp_get_max_threads();

    if (mpi_rank == 0) {
        printf("=== Pearson Correlation Recommender – Hybrid MPI+OpenMP Version ===\n");
        printf("    Users: %d | Items: %d | Top-K: %d\n", N_USERS, N_ITEMS, TOP_K);
        printf("    MPI Processes: %d | OpenMP Threads/Process: %d"
               " | Total workers: %d\n\n",
               mpi_size, omp_threads, mpi_size * omp_threads);
    }

    /*
     * Build Allgatherv descriptor arrays for user_mean and sim_matrix.
     *
     * mean_recvcounts[r] = number of floats rank r contributes to user_mean.
     * sim_recvcounts[r]  = number of floats rank r contributes to sim_matrix
     *                      = local_rows × N_USERS.
     *
     * displs[r] is the offset (in elements) in the receive buffer where
     * rank r's data will be placed.
     */
    int *mean_recvcounts = (int *)malloc(mpi_size * sizeof(int));
    int *mean_displs     = (int *)malloc(mpi_size * sizeof(int));
    int *sim_recvcounts  = (int *)malloc(mpi_size * sizeof(int));
    int *sim_displs      = (int *)malloc(mpi_size * sizeof(int));

    for (int r = 0; r < mpi_size; r++) {
        int s, e;
        get_range(r, mpi_size, &s, &e);
        int local_rows      = e - s;
        mean_recvcounts[r]  = local_rows;
        mean_displs[r]      = s;
        sim_recvcounts[r]   = local_rows * N_USERS;
        sim_displs[r]       = s * N_USERS;
    }

    alloc_arrays();

    double t0, t1, t_sim, t_pred;

    /* ── Phase 1: Data generation ──────────────────────────────────────── */
    /*
     * MPI_Barrier before each timed section ensures that rank 0's timer
     * captures the true wall-clock cost, including any inter-rank load
     * imbalance that might otherwise be hidden inside a collective.
     */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    generate_data();
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = now_sec();
    if (mpi_rank == 0)
        printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    /* ── Phase 2: User means  (OpenMP + MPI_Allgatherv) ────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    compute_user_means(start_u, end_u, mean_recvcounts, mean_displs);
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = now_sec();
    if (mpi_rank == 0)
        printf("[Timing] User mean compute  : %.4f s"
               "  [MPI %d × OMP %d]\n", t1 - t0, mpi_size, omp_threads);

    /* ── Phase 3: Similarity  (OpenMP + MPI_Allgatherv) ────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    compute_all_similarities(start_u, end_u, sim_recvcounts, sim_displs);
    MPI_Barrier(MPI_COMM_WORLD);
    t_sim = now_sec() - t0;
    if (mpi_rank == 0) {
        printf("[Timing] Similarity matrix  : %.4f s"
               "  [MPI %d × OMP %d]\n", t_sim, mpi_size, omp_threads);
        printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());
    }

    /* ── Phase 4: Predictions  (OpenMP, no MPI) ─────────────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = now_sec();
    compute_all_predictions(start_u, end_u);
    MPI_Barrier(MPI_COMM_WORLD);
    t_pred = now_sec() - t0;
    if (mpi_rank == 0)
        printf("[Timing] Prediction phase   : %.4f s"
               "  [MPI %d × OMP %d]\n", t_pred, mpi_size, omp_threads);

    /* ── Phase 5: MAE  (OpenMP reduction + MPI_Reduce) ─────────────────── */
    float mae  = evaluate_mae(start_u, end_u);
    float rmse = evaluate_rmse(start_u, end_u);
    if (mpi_rank == 0) {
        printf("[Eval]   MAE on test set    : %.4f  (test size: %d)\n",
               mae, test_size);
        printf("[Eval]   RMSE on test set   : %.4f  (test size: %d)\n",
               rmse, test_size);
        printf("[Timing] Total (sim+pred)   : %.4f s\n", t_sim + t_pred);
    }

    /* Sample predictions – rank 0 always owns users 0..4 */
    if (mpi_rank == 0) {
        int show_u = (N_USERS < 5) ? N_USERS : 5;
        int show_i = (N_ITEMS < 5) ? N_ITEMS : 5;
        printf("\n--- Sample Predictions (first %d users, %d items) ---\n",
               show_u, show_i);
        printf("%-9s", "User\\Item");
        for (int i = 0; i < show_i; i++) printf("  Item%-3d", i);
        printf("\n");
        for (int u = 0; u < show_u; u++) {
            printf("User %-4d", u);
            for (int i = 0; i < show_i; i++) printf("  %5.2f  ", PRED(u, i));
            printf("\n");
        }
    }

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    free(mean_recvcounts); free(mean_displs);
    free(sim_recvcounts);  free(sim_displs);
    free_arrays();

    /*
     * MPI_Finalize must be the last MPI call.  All ranks must reach it
     * before the process exits, otherwise some ranks may hang waiting
     * for a collective that will never complete.
     */
    MPI_Finalize();
    return 0;
}
