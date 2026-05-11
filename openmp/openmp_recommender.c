/*
 * openmp_recommender.c
 * Pearson Correlation Recommender – OpenMP (Shared Memory) Version
 *
 * ╔═════════════════════════════════════════════════════════════════════════╗
 * ║  WHAT IS OpenMP?                                                        ║
 * ║                                                                         ║
 * ║  OpenMP (Open Multi-Processing) is an API for shared-memory parallel   ║
 * ║  programming in C, C++, and Fortran.  It works through COMPILER        ║
 * ║  DIRECTIVES — special comments (#pragma omp ...) that instruct the     ║
 * ║  compiler to generate multi-threaded code.                             ║
 * ║                                                                         ║
 * ║  The key idea is the FORK-JOIN model:                                   ║
 * ║                                                                         ║
 * ║   single thread ──────────┬──────────────────────┬──────► single thread║
 * ║   (master)                │  FORK: spawn threads │  JOIN: threads done ║
 * ║                           ▼                      ▼                     ║
 * ║                    T0  T1  T2  T3  ...  Tn   (parallel region)        ║
 * ║                    │   │   │   │        │                              ║
 * ║                    └───┴───┴───┴────────┘                              ║
 * ║                         (all threads join here)                        ║
 * ║                                                                         ║
 * ║  The master thread runs the program until it hits a                    ║
 * ║  #pragma omp parallel block.  It then forks T worker threads           ║
 * ║  (including itself), all sharing the SAME memory space.  At the       ║
 * ║  closing brace of the parallel block, all threads join back into one. ║
 * ╚═════════════════════════════════════════════════════════════════════════╝
 *
 * ╔═════════════════════════════════════════════════════════════════════════╗
 * ║  SHARED MEMORY MODEL                                                    ║
 * ║                                                                         ║
 * ║  All threads see the SAME arrays (ratings, sim_matrix, etc.) because   ║
 * ║  they all share one address space — one pool of RAM.                   ║
 * ║                                                                         ║
 * ║   Thread 0 ──┐                                                         ║
 * ║   Thread 1 ──┼──► shared: ratings[], user_mean[], sim_matrix[], ...   ║
 * ║   Thread 2 ──┤                                                         ║
 * ║   Thread 3 ──┘                                                         ║
 * ║                                                                         ║
 * ║  This is powerful (no data copying) but dangerous: if two threads       ║
 * ║  write to the SAME memory location at the SAME time, the result is     ║
 * ║  undefined (a DATA RACE).  We design our loops so each thread writes   ║
 * ║  to a DISJOINT set of indices, eliminating races without locks.        ║
 * ╚═════════════════════════════════════════════════════════════════════════╝
 *
 * ╔═════════════════════════════════════════════════════════════════════════╗
 * ║  WHAT CHANGED FROM THE SERIAL VERSION?                                  ║
 * ║                                                                         ║
 * ║  The algorithm and data structures are IDENTICAL.  The only changes    ║
 * ║  are #pragma omp directives added before loops:                        ║
 * ║                                                                         ║
 * ║   Phase 2 – User means:      #pragma omp parallel for schedule(static) ║
 * ║   Phase 3 – Similarity:      #pragma omp parallel for schedule(dynamic)║
 * ║   Phase 4 – Predictions:     #pragma omp parallel + private nbrs buf   ║
 * ║   Phase 5 – MAE & checksum:  #pragma omp parallel for reduction(+:var) ║
 * ╚═════════════════════════════════════════════════════════════════════════╝
 *
 * Compile:
 *   gcc -O2 -Wall -fopenmp -o openmp_rec openmp/openmp_recommender.c -lm
 *
 *   -fopenmp tells gcc to activate OpenMP support and link the runtime.
 *   Without -fopenmp, all #pragma omp lines are silently ignored and the
 *   program runs serially (useful for verifying correctness).
 *
 * Run:
 *   ./openmp_rec                          # use default thread count
 *   OMP_NUM_THREADS=4  ./openmp_rec       # 4 threads, default size
 *   OMP_NUM_THREADS=8  ./openmp_rec 2000 1500
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>    /* OpenMP header — provides omp_get_wtime(), omp_get_num_threads(), etc. */

/* ── Fixed algorithm parameters (identical across all versions) ─────────── */
#define DEFAULT_USERS  1000
#define DEFAULT_ITEMS  1000
#define SPARSITY       0.70f
#define TOP_K            20
#define SEED             42
#define TEST_RATIO      0.10f

/* ── Runtime sizes (set from CLI in main) ───────────────────────────────── */
static int N_USERS;
static int N_ITEMS;

/*
 * Data arrays — shared across ALL threads.
 *
 * Because OpenMP uses shared memory, every thread can read and write these
 * arrays directly.  The arrays are allocated once on the heap (calloc) by
 * the master thread before any parallel region opens.
 *
 * RACE SAFETY PLAN:
 *   ratings[]     → read-only after Phase 1, written serially in Phase 1.
 *   user_mean[]   → each thread writes to a unique index (its own users).
 *   sim_matrix[]  → each iteration writes to SIM(u,v) and SIM(v,u) for a
 *                    unique pair (u,v) — no two iterations share a cell.
 *   predictions[] → each iteration writes to PRED(u, item) for a unique u.
 */
static float *ratings;
static float *user_mean;
static float *sim_matrix;
static float *predictions;

typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;

/* ── 2-D index macros ────────────────────────────────────────────────────── */
#define R(u,i)    ratings[(u)*N_ITEMS  + (i)]
#define SIM(u,v)  sim_matrix[(u)*N_USERS + (v)]
#define PRED(u,i) predictions[(u)*N_ITEMS + (i)]

/* ═══════════════════════════════════════════════════════════════════════════
 * TIMING
 *
 * omp_get_wtime() is OpenMP's built-in wall-clock timer.
 * It is preferred over clock_gettime() in OpenMP programs because:
 *   ● It is defined by the OpenMP standard — portable across compilers.
 *   ● It measures WALL TIME, not CPU time (important for parallel code where
 *     multiple CPU cores run simultaneously).
 *   ● clock() measures total CPU time across all threads, which would give
 *     "4 seconds" when 4 threads each run for 1 second — very misleading.
 * ═══════════════════════════════════════════════════════════════════════════ */
static inline double now_sec(void) { return omp_get_wtime(); }

/* ── Memory allocation / deallocation ───────────────────────────────────── */
static void alloc_arrays(void)
{
    ratings     = (float *)calloc(N_USERS * N_ITEMS, sizeof(float));
    user_mean   = (float *)calloc(N_USERS,            sizeof(float));
    sim_matrix  = (float *)calloc(N_USERS * N_USERS,  sizeof(float));
    predictions = (float *)calloc(N_USERS * N_ITEMS,  sizeof(float));

    if (!ratings || !user_mean || !sim_matrix || !predictions) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }
}

static void free_arrays(void)
{
    free(ratings); free(user_mean);
    free(sim_matrix); free(predictions);
    free(test_set);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 1 – Data generation  (SERIAL — must not be parallelised)
 *
 * The C standard rand() function is NOT thread-safe: calling it from multiple
 * threads simultaneously causes a data race on its internal state and produces
 * non-deterministic results.
 *
 * We keep data generation serial so the output is identical to the baseline.
 * This means every parallel version can verify its MAE against this one.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void generate_data(void)
{
    srand(SEED);

    int capacity = (int)(N_USERS * N_ITEMS * (1.0f - SPARSITY)) + 1000;
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

    printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | "
           "Test ratings: %d\n",
           N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 2 – User means  (PARALLEL)
 *
 * ── The pragma ───────────────────────────────────────────────────────────
 *   #pragma omp parallel for schedule(static)
 *
 *   "parallel"       → fork a team of threads; each thread executes this block.
 *   "for"            → distribute the for-loop iterations across the threads.
 *                      Each iteration is assigned to exactly ONE thread.
 *   "schedule(static)"→ divide the iterations into equal-size contiguous chunks
 *                      and assign them to threads at compile time (round-robin).
 *
 *   With N_USERS=1000 and 4 threads:
 *     Thread 0 → u = 0..249
 *     Thread 1 → u = 250..499
 *     Thread 2 → u = 500..749
 *     Thread 3 → u = 750..999
 *
 * ── Why schedule(static) here? ───────────────────────────────────────────
 *   Every user row has EXACTLY the same amount of work (scan N_ITEMS items).
 *   Static scheduling distributes equal work → no load imbalance.
 *   Static also has zero runtime overhead (chunks are fixed at compile time).
 *
 * ── Why is this race-free? ───────────────────────────────────────────────
 *   Thread t writes ONLY to user_mean[u] for the u values in its chunk.
 *   No two threads write to the same index.  sum and cnt are LOCAL VARIABLES
 *   declared inside the loop body — each thread has its own private copy on
 *   its stack.  (Variables declared inside a parallel region are automatically
 *   thread-private in OpenMP.)
 * ═══════════════════════════════════════════════════════════════════════════ */
static void compute_user_means(void)
{
    #pragma omp parallel for schedule(static)
    for (int u = 0; u < N_USERS; u++) {
        double sum = 0.0;   /* private to each thread (loop-body variable) */
        int    cnt = 0;     /* private to each thread */
        for (int i = 0; i < N_ITEMS; i++) {
            if (R(u, i) != 0.0f) { sum += R(u, i); cnt++; }
        }
        user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }
    /* Implicit barrier here: all threads finish before the function returns. */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 3 – Pearson similarity
 *
 * pearson_similarity() is a PURE FUNCTION:
 *   ● It only READS ratings[] and user_mean[] — both fully populated before
 *     this phase and never modified during it.
 *   ● It only WRITES to local variables (num, den_u, den_v, co, s).
 *   ● No global state is touched.
 *
 * Pure functions are inherently thread-safe and can be called simultaneously
 * by any number of threads without coordination.
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

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 3 – Similarity matrix  (PARALLEL)
 *
 * ── The pragma ───────────────────────────────────────────────────────────
 *   #pragma omp parallel for schedule(dynamic, 4)
 *
 *   "schedule(dynamic, 4)":
 *     Instead of pre-assigning fixed chunks, threads request NEW work
 *     from a central queue whenever they finish their current chunk.
 *     The chunk size (4) is how many iterations are handed out at a time.
 *
 * ── Why dynamic here and not static? ─────────────────────────────────────
 *   The outer loop runs for u = 0, 1, 2, …, N_USERS-1.
 *   The inner loop runs for v = u+1, …, N_USERS-1.
 *
 *   Iteration 0 (u=0) does N_USERS-1 Pearson calls.
 *   Iteration 1 (u=1) does N_USERS-2 calls.
 *   …
 *   Iteration N-1 (u=N-1) does 0 calls.
 *
 *   Work per iteration DECREASES as u grows.  With static scheduling,
 *   the thread assigned to small-u iterations does much more work than the
 *   thread assigned to large-u iterations → severe load imbalance.
 *   Dynamic scheduling lets fast-finishing threads pick up more iterations,
 *   keeping all cores busy until the end.
 *
 *   The chunk size of 4 is a tuning knob: smaller = better balance but more
 *   queue overhead; larger = less overhead but less flexibility.
 *
 * ── Race-free write strategy ─────────────────────────────────────────────
 *   Each outer iteration (value of u) is assigned to exactly ONE thread.
 *   That thread writes SIM(u, v) for all v > u, AND writes SIM(v, u)
 *   for those same v.
 *
 *   Could SIM(v, u) be written by two threads at once?
 *     Thread A owns outer iteration u=5 → writes SIM(5,7) and SIM(7,5).
 *     Thread B owns outer iteration u=7 → writes SIM(7, v) for v > 7.
 *     Thread B NEVER writes SIM(7, 5) because v > u=7 means v≥8.
 *   So the writes are disjoint — no race.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void compute_all_similarities(void)
{
    /* Each thread sets its own slice of the diagonal. */
    #pragma omp parallel for schedule(static)
    for (int u = 0; u < N_USERS; u++)
        SIM(u, u) = 1.0f;

    /*
     * Dynamic scheduling balances the decreasing inner-loop work across
     * threads.  Each thread processes a contiguous chunk of 4 outer
     * iterations before requesting the next chunk.
     */
    #pragma omp parallel for schedule(dynamic, 4)
    for (int u = 0; u < N_USERS; u++) {
        for (int v = u + 1; v < N_USERS; v++) {
            float s = pearson_similarity(u, v);
            SIM(u, v) = s;
            SIM(v, u) = s;   /* mirror: safe because no other thread owns row u */
        }
    }
    /* Implicit OpenMP barrier: all threads done before we read sim_matrix. */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 4 – Predictions  (PARALLEL with thread-private buffer)
 *
 * ── Problem: shared state ────────────────────────────────────────────────
 *   The serial version allocates ONE nbrs[] buffer and reuses it for every
 *   (user, item) cell.  In a parallel version, if all threads shared a
 *   single nbrs[] buffer and one thread sorted it while another thread was
 *   reading it, we would have a DATA RACE.
 *
 * ── Solution: thread-private allocation ──────────────────────────────────
 *   We open a #pragma omp parallel block (without "for").  Inside this
 *   block, each thread executes ALL the code — including the malloc.
 *   malloc() is thread-safe: each call returns a DIFFERENT heap allocation
 *   for each thread.  So thread 0 gets its own nbrs, thread 1 gets its own,
 *   and so on.  No sharing, no race.
 *
 *   The #pragma omp for inside the parallel block then distributes the
 *   outer loop (over users u) across the threads, each using its private
 *   nbrs buffer.
 *
 * ── Structure ────────────────────────────────────────────────────────────
 *
 *   #pragma omp parallel        ← fork T threads
 *   {
 *       SimPair *nbrs = malloc(...)   ← each thread allocates its OWN buffer
 *
 *       #pragma omp for schedule(dynamic, 2)   ← distribute u across threads
 *       for (int u = ...) {
 *           for (int item = ...) {
 *               ... use nbrs (private) ...
 *           }
 *       }
 *       // implicit barrier: all threads done with the for loop
 *
 *       free(nbrs);   ← each thread frees its own buffer
 *   }                 ← threads join (barrier)
 *
 * ── Why schedule(dynamic, 2) here? ──────────────────────────────────────
 *   Users with few rated items produce sparser neighbour lists → faster.
 *   Users with many rated items produce denser lists → slower.
 *   Dynamic scheduling lets fast threads pick up more users.
 *   Chunk size 2 gives fine-grained balancing (at the cost of some overhead).
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct { int idx; float val; } SimPair;

static int cmp_sim_desc(const void *a, const void *b)
{
    float fa = ((const SimPair *)a)->val;
    float fb = ((const SimPair *)b)->val;
    return (fb > fa) - (fb < fa);
}

static void compute_all_predictions(void)
{
    #pragma omp parallel
    {
        /*
         * Each thread allocates its own private neighbour buffer.
         * malloc() is internally synchronised — safe to call concurrently.
         * The buffer is created when the parallel region opens and freed
         * when it closes, so it exists for exactly one thread's lifetime.
         */
        SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));

        /*
         * #pragma omp for (without "parallel") distributes the loop that
         * follows across the ALREADY-OPEN team of threads.  Each value of u
         * is assigned to exactly one thread.
         *
         * schedule(dynamic, 2): work chunks of 2 users assigned on demand.
         */
        #pragma omp for schedule(dynamic, 2)
        for (int u = 0; u < N_USERS; u++) {
            for (int item = 0; item < N_ITEMS; item++) {

                if (R(u, item) != 0.0f) {
                    PRED(u, item) = R(u, item);
                    continue;
                }

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
        /* Implicit barrier: all threads finish the for loop before free(). */

        free(nbrs);   /* each thread frees its own allocation */
    }
    /* Implicit barrier: threads join here before we read predictions[]. */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 5 – Evaluation  (PARALLEL with reduction)
 *
 * ── The reduction clause ─────────────────────────────────────────────────
 *   #pragma omp parallel for reduction(+:err)
 *
 *   "reduction(+:err)" tells OpenMP:
 *     1. Give each thread a PRIVATE COPY of err, initialised to 0.0.
 *     2. Let each thread accumulate into its own private copy freely
 *        (no locks, no atomics — fast!).
 *     3. At the end of the parallel region, SUM all private copies into
 *        the ORIGINAL err variable.
 *
 *   Without the reduction clause, all threads would increment the shared err
 *   simultaneously → data race → wrong answer.
 *   With the clause, the compiler generates safe, efficient code for us.
 *
 * ── Why not just use an atomic? ──────────────────────────────────────────
 *   #pragma omp atomic update
 *   err += ...;
 *   This is correct but SERIALISES every addition — each thread must wait
 *   for the others to finish their atomic update before proceeding.
 *   reduction() is faster: each thread works in parallel on its private
 *   copy and the final merge (one addition per thread) is negligible.
 * ═══════════════════════════════════════════════════════════════════════════ */
static float evaluate_mae(void)
{
    if (test_size == 0) return 0.0f;
    double err = 0.0;

    /*
     * Each thread accumulates error for a subset of test entries into its
     * private copy of err.  At the closing brace, OpenMP sums all private
     * copies into the shared err.
     */
    #pragma omp parallel for reduction(+:err) schedule(static)
    for (int t = 0; t < test_size; t++)
        err += fabs(PRED(test_set[t].user, test_set[t].item) - test_set[t].rating);

    return (float)(err / test_size);
}

/*
 * similarity_checksum — parallelised with reduction for consistency.
 * The same reduction(+:s) pattern accumulates the double sum safely.
 */
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
    N_USERS = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;

    if (N_USERS <= 0 || N_ITEMS <= 0) {
        fprintf(stderr, "Usage: %s [num_users] [num_items]\n", argv[0]);
        return EXIT_FAILURE;
    }

    /*
     * Query the thread count BEFORE launching any parallel work.
     *
     * #pragma omp parallel         → fork a temporary team of threads.
     * #pragma omp single           → ONE thread (any) executes this block.
     *   omp_get_num_threads()      → how many threads are in the current team.
     *
     * We use omp_get_max_threads() outside a parallel region alternatively,
     * but the omp single pattern confirms the actual count the runtime will use.
     *
     * OMP_NUM_THREADS environment variable controls this count.
     * If unset, the default is usually the number of logical CPU cores.
     */
    int nthreads;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    printf("=== Pearson Correlation Recommender – OpenMP Version ===\n");
    printf("    Users: %d | Items: %d | Top-K: %d | Threads: %d\n\n",
           N_USERS, N_ITEMS, TOP_K, nthreads);

    double t0, t1, t_sim, t_pred;

    alloc_arrays();

    /* ── Phase 1: Data generation (serial — RNG is not thread-safe) ──────── */
    t0 = now_sec();
    generate_data();
    t1 = now_sec();
    printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    /* ── Phase 2: User means (parallel, static schedule) ─────────────────── */
    t0 = now_sec();
    compute_user_means();
    t1 = now_sec();
    printf("[Timing] User mean compute  : %.4f s  [parallel, %d threads]\n",
           t1 - t0, nthreads);

    /* ── Phase 3: Similarity matrix (parallel, dynamic schedule) ─────────── */
    t0 = now_sec();
    compute_all_similarities();
    t1 = now_sec();
    t_sim = t1 - t0;
    printf("[Timing] Similarity matrix  : %.4f s  [parallel, %d threads]\n",
           t_sim, nthreads);
    printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());

    /* ── Phase 4: Predictions (parallel, private nbrs buffer) ────────────── */
    t0 = now_sec();
    compute_all_predictions();
    t1 = now_sec();
    t_pred = t1 - t0;
    printf("[Timing] Prediction phase   : %.4f s  [parallel, %d threads]\n",
           t_pred, nthreads);

    /* ── Phase 5: Evaluation (parallel reduction) ─────────────────────────── */
    printf("[Eval]   MAE on test set    : %.4f  (test size: %d)\n",
           evaluate_mae(), test_size);
    printf("[Timing] Total (sim+pred)   : %.4f s\n", t_sim + t_pred);

    /* Sample predictions for visual inspection */
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

    free_arrays();
    return 0;
}
