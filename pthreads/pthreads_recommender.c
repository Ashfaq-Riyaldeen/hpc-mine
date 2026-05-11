/*
 * pthreads_recommender.c
 * Pearson Correlation Recommender – POSIX Threads (Pthreads) Version
 *
 * This version parallelises the three expensive phases (user-mean computation,
 * similarity matrix, and prediction) using raw POSIX threads.  Unlike OpenMP,
 * we manage every thread explicitly: creation, work partitioning, and joining.
 * The join after each phase acts as an implicit barrier – no thread moves on
 * to the next phase until every other thread has finished the current one.
 *
 * Compile:
 *   gcc -O2 -Wall -o pthreads_rec pthreads/pthreads_recommender.c -lpthread -lm
 *
 * Run:
 *   ./pthreads_rec                     # 1000 users, 1000 items, 4 threads
 *   ./pthreads_rec 2000 1500 8         # 2000 users, 1500 items, 8 threads
 *   ./pthreads_rec 500  500  2         # small scale, 2 threads
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

/* ── Fixed algorithm parameters (identical to serial / OpenMP baselines) ─── */
#define DEFAULT_USERS    1000
#define DEFAULT_ITEMS    1000
#define DEFAULT_THREADS     4   /* used when no third CLI argument is given    */
#define MAX_THREADS        64   /* hard cap so we can use stack-allocated arrays*/
#define SPARSITY         0.70f  /* fraction of entries that are unrated        */
#define TOP_K              20   /* max neighbours used when predicting a rating */
#define SEED               42   /* RNG seed – must match serial for same data  */
#define TEST_RATIO        0.10f /* fraction of known ratings held out for MAE  */

/* ── Runtime sizes – filled from CLI arguments in main() ────────────────── */
static int N_USERS;
static int N_ITEMS;
static int N_THREADS;

/*
 * All four matrices are flat 1-D arrays allocated on the heap.
 * Indexed via the macros below so the code reads like 2-D access.
 *
 * Threads read from ratings[] and user_mean[] but write to different
 * row-ranges of sim_matrix[] and predictions[], so no mutex is required
 * for those writes (see the race-condition analysis in thread_similarities).
 */
static float *ratings;       /* [N_USERS × N_ITEMS]  – 0.0 means "unrated"   */
static float *user_mean;     /* [N_USERS]             – mean of each user's ratings */
static float *sim_matrix;    /* [N_USERS × N_USERS]  – Pearson correlation    */
static float *predictions;   /* [N_USERS × N_ITEMS]  – final predicted ratings*/

/* Test set: ratings hidden from training, used to compute MAE */
typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;

/* ── Convenience macros for 2-D indexing ────────────────────────────────── */
#define R(u,i)    ratings[(u)*N_ITEMS  + (i)]
#define SIM(u,v)  sim_matrix[(u)*N_USERS + (v)]
#define PRED(u,i) predictions[(u)*N_ITEMS + (i)]

/* ─────────────────────────────────────────────────────────────────────────
 * ThreadArgs – passed to every thread function.
 *
 * Every thread receives its own index (tid, 0-based) and the total count
 * (nthreads).  From these two values each thread derives which slice of
 * work it owns, keeping the thread functions self-contained.
 * ───────────────────────────────────────────────────────────────────────── */
typedef struct {
    int tid;       /* this thread's index, 0 .. nthreads-1 */
    int nthreads;  /* total number of threads launched      */
} ThreadArgs;

/* ── Wall-clock timer (same as serial baseline) ─────────────────────────── */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ─────────────────────────────────────────────────────────────────────────
 * work_range() – compute the [start, end) slice for thread tid.
 *
 * We use ceiling division so all N items are covered even when N is not
 * divisible by nthreads.  The last thread may receive fewer items than
 * the others, but never more, and no item is processed twice.
 * ───────────────────────────────────────────────────────────────────────── */
static void work_range(int total, int tid, int nthreads, int *start, int *end)
{
    int chunk = (total + nthreads - 1) / nthreads;
    *start = tid * chunk;
    *end   = *start + chunk;
    if (*end > total) *end = total;
}

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
    free(ratings);
    free(user_mean);
    free(sim_matrix);
    free(predictions);
    free(test_set);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 1 – Data generation  (serial, identical to serial baseline)
 *
 * Must be serial so that the sequence of rand() calls – and therefore
 * the rating matrix produced – is byte-for-byte identical to the serial
 * version.  This guarantees a fair comparison: same data, different runtime.
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
                /* Hold this rating out for evaluation; leave R(u,i) = 0 */
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
 * PHASE 2 – User means  (parallel thread function)
 *
 * Thread tid computes user_mean[u] for every u in [start, end).
 *
 * Safety: each thread writes to a disjoint range of user_mean[], so no
 * synchronisation is needed.  The pthread_join in run_parallel() ensures
 * that all means are complete before the similarity phase begins.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void *thread_user_means(void *arg)
{
    ThreadArgs *a = (ThreadArgs *)arg;
    int start, end;
    work_range(N_USERS, a->tid, a->nthreads, &start, &end);

    for (int u = start; u < end; u++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < N_ITEMS; i++) {
            if (R(u, i) != 0.0f) { sum += R(u, i); cnt++; }
        }
        /* Default to 3.0 (mid-scale) if the user has no training ratings */
        user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 3 – Pearson similarity  (pure helper + parallel thread function)
 *
 * pearson_similarity() is a stateless helper: it reads from ratings[] and
 * user_mean[] (read-only after Phase 2) and returns a scalar.  It is safe
 * to call concurrently from multiple threads.
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

    if (co < 2) return 0.0f;          /* not enough co-rated items             */
    double denom = sqrt(den_u) * sqrt(den_v);
    if (denom < 1e-10) return 0.0f;   /* one user has zero variance on overlap */

    float s = (float)(num / denom);
    if (s >  1.0f) s =  1.0f;         /* clamp to [-1, 1] for numerical safety */
    if (s < -1.0f) s = -1.0f;
    return s;
}

/*
 * thread_similarities – parallel worker for the similarity matrix.
 *
 * Work partition: thread tid owns outer-loop rows u in [start, end).
 * For each owned u it computes similarities with all v > u (upper triangle)
 * and immediately mirrors the result: SIM(v, u) = SIM(u, v).
 *
 * Race-condition analysis (why no mutex is needed):
 *   - SIM(u, v) where v > u  →  written only by the thread that owns row u.
 *   - SIM(v, u) where v > u  →  also written only by the thread that owns row u
 *     (it is the mirror of SIM(u,v) computed in that thread's own loop).
 *   - No other thread touches column u of row v for v > u, because a thread
 *     only writes SIM(v, u') when u' is its own outer-loop value.  Since outer
 *     loop values are partitioned without overlap, writes to each cell come from
 *     exactly one thread.  ∴ all writes are disjoint → no data race.
 */
static void *thread_similarities(void *arg)
{
    ThreadArgs *a = (ThreadArgs *)arg;
    int start, end;
    work_range(N_USERS, a->tid, a->nthreads, &start, &end);

    for (int u = start; u < end; u++) {
        SIM(u, u) = 1.0f;                          /* a user is perfectly similar to themselves */
        for (int v = u + 1; v < N_USERS; v++) {
            float s = pearson_similarity(u, v);
            SIM(u, v) = s;                         /* upper triangle */
            SIM(v, u) = s;                         /* lower triangle mirror – race-free (see above) */
        }
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 4 – Rating prediction  (parallel thread function)
 *
 * For each (user, item) pair where no training rating exists, predict the
 * rating using the weighted average of the TOP_K most similar neighbours
 * who did rate that item.
 *
 * Thread safety:
 *   - nbrs[] is allocated privately inside each thread (heap, freed before
 *     the thread returns).  No two threads share this buffer.
 *   - Each thread writes to a disjoint row-range of predictions[], so no
 *     mutex is needed for that write either.
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct { int idx; float val; } SimPair;

static int cmp_sim_desc(const void *a, const void *b)
{
    float fa = ((const SimPair *)a)->val;
    float fb = ((const SimPair *)b)->val;
    return (fb > fa) - (fb < fa);   /* descending similarity order */
}

static void *thread_predictions(void *arg)
{
    ThreadArgs *a = (ThreadArgs *)arg;
    int start, end;
    work_range(N_USERS, a->tid, a->nthreads, &start, &end);

    /* Private scratch buffer for neighbour candidates – one per thread */
    SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));
    if (!nbrs) {
        fprintf(stderr, "Thread %d: malloc failed.\n", a->tid);
        return NULL;
    }

    for (int u = start; u < end; u++) {
        for (int item = 0; item < N_ITEMS; item++) {

            /* Keep the actual rating if the user already rated this item */
            if (R(u, item) != 0.0f) {
                PRED(u, item) = R(u, item);
                continue;
            }

            /* Collect all neighbours who (a) rated this item and
             * (b) have a positive similarity with user u              */
            int cnt = 0;
            for (int v = 0; v < N_USERS; v++) {
                if (v == u || R(v, item) == 0.0f) continue;
                float s = SIM(u, v);
                if (s <= 0.0f) continue;
                nbrs[cnt].idx = v;
                nbrs[cnt].val = s;
                cnt++;
            }

            /* No usable neighbours: fall back to the user's mean rating */
            if (cnt == 0) {
                PRED(u, item) = user_mean[u];
                continue;
            }

            /* Sort by similarity descending and keep the best TOP_K */
            qsort(nbrs, cnt, sizeof(SimPair), cmp_sim_desc);
            int k = (cnt < TOP_K) ? cnt : TOP_K;

            /* Weighted sum of neighbour deviations from their own means */
            double num = 0.0, den = 0.0;
            for (int j = 0; j < k; j++) {
                float s = nbrs[j].val;
                num += s * (R(nbrs[j].idx, item) - user_mean[nbrs[j].idx]);
                den += s;
            }

            float pred = (den > 1e-10)
                         ? user_mean[u] + (float)(num / den)
                         : user_mean[u];

            /* Clamp to valid rating scale [1, 5] */
            if (pred < 1.0f) pred = 1.0f;
            if (pred > 5.0f) pred = 5.0f;
            PRED(u, item) = pred;
        }
    }

    free(nbrs);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 5 – Evaluation (serial – small test set, not a bottleneck)
 * ═══════════════════════════════════════════════════════════════════════════ */
static float evaluate_mae(void)
{
    if (test_size == 0) return 0.0f;
    double err = 0.0;
    for (int t = 0; t < test_size; t++)
        err += fabs(PRED(test_set[t].user, test_set[t].item) - test_set[t].rating);
    return (float)(err / test_size);
}

static double similarity_checksum(void)
{
    double s = 0.0;
    for (int u = 0; u < N_USERS; u++)
        for (int v = 0; v < N_USERS; v++)
            s += SIM(u, v);
    return s;
}

/* ─────────────────────────────────────────────────────────────────────────
 * run_parallel() – launch all threads for one phase, wait for them, time it.
 *
 * Each thread receives a pointer to its own ThreadArgs entry (tid and
 * nthreads).  After pthread_join returns for the last thread, ALL threads
 * for this phase have completed – this acts as a phase barrier.
 *
 * Keeping thread launch/join per phase (rather than a persistent thread
 * pool with barriers) makes the control flow easy to follow and debug.
 * The overhead of pthread_create/join is negligible compared to the
 * O(N² × M) similarity computation.
 * ───────────────────────────────────────────────────────────────────────── */
static double run_parallel(void *(*fn)(void *),
                            pthread_t  *threads,
                            ThreadArgs *args)
{
    double t0 = now_sec();

    for (int t = 0; t < N_THREADS; t++) {
        args[t].tid      = t;
        args[t].nthreads = N_THREADS;
        if (pthread_create(&threads[t], NULL, fn, &args[t]) != 0) {
            fprintf(stderr, "Error: pthread_create failed for thread %d\n", t);
            exit(EXIT_FAILURE);
        }
    }

    for (int t = 0; t < N_THREADS; t++)
        pthread_join(threads[t], NULL);

    return now_sec() - t0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main()
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    N_USERS   = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS   = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;
    N_THREADS = (argc >= 4) ? atoi(argv[3]) : DEFAULT_THREADS;

    if (N_USERS <= 0 || N_ITEMS <= 0 || N_THREADS <= 0) {
        fprintf(stderr, "Usage: %s [num_users] [num_items] [num_threads]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    if (N_THREADS > MAX_THREADS) {
        fprintf(stderr, "Warning: capping threads at %d\n", MAX_THREADS);
        N_THREADS = MAX_THREADS;
    }

    printf("=== Pearson Correlation Recommender – Pthreads Version ===\n");
    printf("    Users: %d | Items: %d | Top-K: %d | Threads: %d\n\n",
           N_USERS, N_ITEMS, TOP_K, N_THREADS);

    /* Stack-allocated thread handles and argument structs */
    pthread_t  threads[MAX_THREADS];
    ThreadArgs args[MAX_THREADS];

    double t0, t1, t_sim, t_pred;

    alloc_arrays();

    /* ── Phase 1: Data generation – always serial ─────────────────────── */
    t0 = now_sec();
    generate_data();
    t1 = now_sec();
    printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    /* ── Phase 2: User means – parallel ──────────────────────────────── */
    t0 = now_sec();
    run_parallel(thread_user_means, threads, args);
    t1 = now_sec();
    printf("[Timing] User mean compute  : %.4f s  [pthreads, %d threads]\n",
           t1 - t0, N_THREADS);

    /* ── Phase 3: Similarity matrix – parallel ───────────────────────── */
    t_sim = run_parallel(thread_similarities, threads, args);
    printf("[Timing] Similarity matrix  : %.4f s  [pthreads, %d threads]\n",
           t_sim, N_THREADS);
    printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());

    /* ── Phase 4: Predictions – parallel ─────────────────────────────── */
    t_pred = run_parallel(thread_predictions, threads, args);
    printf("[Timing] Prediction phase   : %.4f s  [pthreads, %d threads]\n",
           t_pred, N_THREADS);

    printf("[Eval]   MAE on test set    : %.4f  (test size: %d)\n",
           evaluate_mae(), test_size);
    printf("[Timing] Total (sim+pred)   : %.4f s\n", t_sim + t_pred);

    /* ── Sample output (matches serial / OpenMP format) ──────────────── */
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
