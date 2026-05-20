/*
 * serial_recommender.c
 * Pearson Correlation Recommender – Serial (Baseline) Version
 *
 * ╔═════════════════════════════════════════════════════════════════════════╗
 * ║  WHAT IS THIS PROGRAM?                                                  ║
 * ║                                                                         ║
 * ║  This is a User-Based Collaborative Filtering recommender system.      ║
 * ║  Given a matrix of user ratings (rows = users, columns = items),       ║
 * ║  it predicts the rating a user would give to items they haven't        ║
 * ║  rated yet, using the ratings of "similar" users as a guide.           ║
 * ║                                                                         ║
 * ║  The similarity between two users is measured with PEARSON             ║
 * ║  CORRELATION — a standard statistical measure of how linearly          ║
 * ║  related two sets of numbers are, scaled to the range [-1, +1].        ║
 * ║                                                                         ║
 * ║  ● +1.0 → the two users always agree (rate items identically)          ║
 * ║  ●  0.0 → no relationship between their ratings                        ║
 * ║  ● -1.0 → they always disagree (one rates high where the other         ║
 * ║            rates low)                                                   ║
 * ╚═════════════════════════════════════════════════════════════════════════╝
 *
 * ╔═════════════════════════════════════════════════════════════════════════╗
 * ║  WHY A SERIAL VERSION?                                                  ║
 * ║                                                                         ║
 * ║  This file is the BASELINE — it runs on a single CPU core with no      ║
 * ║  parallelism.  Every other version (OpenMP, Pthreads, MPI, CUDA,       ║
 * ║  Hybrid) must produce the SAME numerical results as this one for the   ║
 * ║  same input, and must be faster.                                        ║
 * ║                                                                         ║
 * ║  The serial version lets us:                                            ║
 * ║   1. Verify correctness — if a parallel version's MAE differs by more  ║
 * ║      than ±0.001, there is a race condition or distribution bug.        ║
 * ║   2. Measure the SPEEDUP = T_serial / T_parallel.                      ║
 * ║   3. Understand the algorithm before optimising it.                    ║
 * ╚═════════════════════════════════════════════════════════════════════════╝
 *
 * ╔═════════════════════════════════════════════════════════════════════════╗
 * ║  ALGORITHM OVERVIEW (5 phases)                                          ║
 * ║                                                                         ║
 * ║  Phase 1 – DATA GENERATION                                             ║
 * ║    Synthesise a sparse ratings matrix using a fixed random seed.       ║
 * ║    Hold 10 % of ratings out as a test set for evaluation.              ║
 * ║                                                                         ║
 * ║  Phase 2 – USER MEANS                                                  ║
 * ║    For each user, compute the average of their known ratings.          ║
 * ║    Needed by the Pearson formula to measure deviation from the mean.   ║
 * ║                                                                         ║
 * ║  Phase 3 – SIMILARITY MATRIX                                           ║
 * ║    Compute Pearson correlation for every (user u, user v) pair.        ║
 * ║    Result: an N_USERS × N_USERS symmetric matrix.                      ║
 * ║    This is the BOTTLENECK: O(N² × M) work.                             ║
 * ║                                                                         ║
 * ║  Phase 4 – PREDICTIONS                                                 ║
 * ║    For each (user, item) where the rating is unknown, predict using    ║
 * ║    the weighted average of the TOP_K most similar neighbours.          ║
 * ║                                                                         ║
 * ║  Phase 5 – EVALUATION                                                  ║
 * ║    Compute Mean Absolute Error (MAE) on the held-out test set.         ║
 * ╚═════════════════════════════════════════════════════════════════════════╝
 *
 * Compile:
 *   gcc -O2 -Wall -o serial_rec serial/serial_recommender.c -lm
 *
 * Run:
 *   ./serial_rec              # default 1000 users, 1000 items
 *   ./serial_rec 500  300     # custom sizes
 *   ./serial_rec 2000 1500
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>    /* sqrt(), fabs() — compile with -lm */
#include <time.h>    /* clock_gettime() for wall-clock timing */

/* ═══════════════════════════════════════════════════════════════════════════
 * ALGORITHM CONSTANTS
 *
 * These values are identical across ALL versions of this project (serial,
 * OpenMP, Pthreads, MPI, CUDA, Hybrid).  Keeping them the same guarantees
 * that every version works on the same dataset and can be compared fairly.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define DEFAULT_USERS  1000   /* rows in the ratings matrix                   */
#define DEFAULT_ITEMS  1000   /* columns in the ratings matrix                */
#define SPARSITY       0.70f  /* 70 % of entries are unrated (0.0)            */
#define TOP_K            20   /* max neighbours used in the prediction formula */
#define SEED             42   /* fixed RNG seed — same data on every run      */
#define TEST_RATIO      0.10f /* 10 % of known ratings held out for MAE       */

/* ═══════════════════════════════════════════════════════════════════════════
 * RUNTIME SIZE VARIABLES
 *
 * Set from command-line arguments in main().  Using variables (not macros)
 * allows the problem size to be chosen at run time without recompiling.
 * Declared static so they are file-scoped (visible to all functions here).
 * ═══════════════════════════════════════════════════════════════════════════ */
static int N_USERS;
static int N_ITEMS;

/* ═══════════════════════════════════════════════════════════════════════════
 * DATA ARRAYS
 *
 * All matrices are stored as FLAT 1-D arrays (row-major order) for cache
 * efficiency.  A 2-D logical matrix M[row][col] is stored as:
 *   flat_array[row * num_cols + col]
 *
 * This is identical to how C stores 2-D arrays in memory and lets us use
 * a single malloc/calloc call per matrix.
 *
 *   ratings     [N_USERS × N_ITEMS]  — the input rating matrix.
 *                                       0.0 means "user has not rated this item".
 *                                       Ratings are integers 1–5 stored as floats.
 *
 *   user_mean   [N_USERS]            — mean of each user's known ratings.
 *                                       Used to normalise in the Pearson formula.
 *
 *   sim_matrix  [N_USERS × N_USERS]  — Pearson similarity between every
 *                                       pair of users.  Symmetric: SIM(u,v)==SIM(v,u).
 *
 *   predictions [N_USERS × N_ITEMS]  — predicted ratings for all (user, item)
 *                                       pairs.  For rated items, stores the actual.
 * ═══════════════════════════════════════════════════════════════════════════ */
static float *ratings;
static float *user_mean;
static float *sim_matrix;
static float *predictions;

/*
 * TestEntry — one held-out rating used for evaluation.
 * test_set[] stores the entries removed from ratings[] during data generation.
 * After predictions are made, we compare PRED(user, item) against rating.
 */
typedef struct { int user; int item; float rating; } TestEntry;
static TestEntry *test_set;
static int        test_size;   /* number of held-out entries */

/* ═══════════════════════════════════════════════════════════════════════════
 * INDEX MACROS
 *
 * These macros convert (row, col) 2-D indices into flat 1-D indices.
 * Using macros keeps the formulas readable without function-call overhead.
 *
 *   R(u, i)   → ratings[u * N_ITEMS  + i]  (user u, item i)
 *   SIM(u, v) → sim_matrix[u * N_USERS + v] (similarity between users u and v)
 *   PRED(u,i) → predictions[u * N_ITEMS + i]
 * ═══════════════════════════════════════════════════════════════════════════ */
#define R(u,i)    ratings[(size_t)(u)*N_ITEMS  + (i)]
#define SIM(u,v)  sim_matrix[(size_t)(u)*N_USERS + (v)]
#define PRED(u,i) predictions[(size_t)(u)*N_ITEMS + (i)]

/* ═══════════════════════════════════════════════════════════════════════════
 * TIMING
 *
 * clock_gettime(CLOCK_MONOTONIC) returns a monotonically increasing time
 * (it never goes backwards, unlike gettimeofday which can be adjusted by NTP).
 * Subtracting two now_sec() calls gives wall-clock elapsed time in seconds.
 * ═══════════════════════════════════════════════════════════════════════════ */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MEMORY ALLOCATION / DEALLOCATION
 *
 * calloc() is used instead of malloc() because it zero-initialises memory.
 * Zero is our sentinel value for "unrated" in the ratings matrix, so all
 * entries start as unrated before generate_data() fills them in.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void alloc_arrays(void)
{
    ratings     = (float *)calloc((size_t)N_USERS * N_ITEMS, sizeof(float));
    user_mean   = (float *)calloc(N_USERS,                    sizeof(float));
    sim_matrix  = (float *)calloc((size_t)N_USERS * N_USERS,  sizeof(float));
    predictions = (float *)calloc((size_t)N_USERS * N_ITEMS,  sizeof(float));

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
 * PHASE 1 – Data generation
 *
 * PURPOSE: Build a synthetic sparse ratings matrix and a held-out test set.
 *
 * HOW IT WORKS:
 *   We iterate over every (user, item) cell.  For each cell we roll three dice:
 *
 *   Die 1 – SPARSITY check (70 % chance of skipping):
 *     if rand() / RAND_MAX < 0.70, the cell stays unrated (0.0).
 *
 *   Die 2 – RATING value (for the 30 % of cells that are rated):
 *     rating = (rand() % 5) + 1  → integer in {1, 2, 3, 4, 5}.
 *
 *   Die 3 – TRAIN vs TEST split (10 % of rated cells go to test set):
 *     if rand() / RAND_MAX < 0.10:
 *       store in test_set[] (ratings matrix cell stays 0 — "unrated")
 *     else:
 *       store in ratings[] (training data)
 *
 * WHY srand(SEED)?
 *   Using a fixed seed makes the program DETERMINISTIC.  Every run produces
 *   exactly the same ratings matrix.  This is essential for comparing results
 *   across serial/parallel versions: they all see the same input and must
 *   produce the same MAE.
 *
 * COMPLEXITY: O(N_USERS × N_ITEMS) — linear in the matrix size.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void generate_data(void)
{
    srand(SEED);   /* seed the C standard library RNG — deterministic output */

    /*
     * Pre-allocate the test set array.
     * The expected number of test entries is N_USERS * N_ITEMS * (1-SPARSITY) * TEST_RATIO.
     * We add 1000 as a buffer to handle rand() variance.
     */
    int capacity = (int)((size_t)N_USERS * N_ITEMS * (1.0f - SPARSITY)) + 1000;
    test_set  = (TestEntry *)malloc(capacity * sizeof(TestEntry));
    test_size = 0;

    for (int u = 0; u < N_USERS; u++) {
        for (int i = 0; i < N_ITEMS; i++) {

            /* Die 1: skip this cell with probability SPARSITY */
            if ((float)rand() / RAND_MAX < SPARSITY) continue;

            /* Die 2: generate a rating in [1, 5] */
            float rating = (float)(rand() % 5) + 1.0f;

            /* Die 3: send to test set with probability TEST_RATIO */
            if ((float)rand() / RAND_MAX < TEST_RATIO && test_size < capacity) {
                test_set[test_size].user   = u;
                test_set[test_size].item   = i;
                test_set[test_size].rating = rating;
                test_size++;
                /* R(u, i) remains 0.0 — treated as "unrated" during training */
            } else {
                R(u, i) = rating;   /* goes into the training matrix */
            }
        }
    }

    printf("[Data]   Users: %d | Items: %d | Sparsity: %.0f%% | "
           "Test ratings: %d\n",
           N_USERS, N_ITEMS, SPARSITY * 100.0f, test_size);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 2 – User means
 *
 * PURPOSE: compute mean_u = (Σ ratings of user u) / (count of rated items).
 *
 * WHY DO WE NEED THIS?
 *   Different users have different "rating scales".  An optimistic user might
 *   rate everything 4–5, a pessimistic user 1–2.  The Pearson formula
 *   corrects for this by working with DEVIATIONS from each user's mean
 *   instead of raw ratings.
 *
 *   If a user has rated nothing (cnt == 0) we default to 3.0 (the midpoint
 *   of [1, 5]) so the prediction formula has something to work with.
 *
 * COMPLEXITY: O(N_USERS × N_ITEMS) — each cell is visited exactly once.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void compute_user_means(void)
{
    for (int u = 0; u < N_USERS; u++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < N_ITEMS; i++) {
            if (R(u, i) != 0.0f) {   /* only include rated items */
                sum += R(u, i);
                cnt++;
            }
        }
        user_mean[u] = (cnt > 0) ? (float)(sum / cnt) : 3.0f;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 3a – Pearson similarity between one pair of users (u, v)
 *
 * FORMULA:
 *
 *              Σᵢ (Rᵤᵢ - μᵤ)(Rᵥᵢ - μᵥ)
 *   sim(u,v) = ─────────────────────────────────────────
 *              √[ Σᵢ (Rᵤᵢ - μᵤ)² ]  ×  √[ Σᵢ (Rᵥᵢ - μᵥ)² ]
 *
 *   where the sum is over items i rated by BOTH u AND v (co-rated items),
 *   μᵤ = user_mean[u], μᵥ = user_mean[v].
 *
 * INTUITION:
 *   The numerator is the dot product of the two users' DEVIATION vectors.
 *   A positive dot product means they deviate in the same direction (both
 *   like or dislike the same items relative to their own means) → positive
 *   similarity.  The denominator normalises to [-1, +1].
 *
 * EDGE CASES handled:
 *   ● co < 2    → fewer than 2 co-rated items means we cannot compute a
 *                 meaningful correlation → return 0.
 *   ● denom ≈ 0 → one user gave the same rating to everything (zero variance)
 *                 → undefined correlation → return 0.
 *   ● clamping  → floating-point rounding can push s just outside [-1,1];
 *                 we clamp to keep the result valid.
 *
 * COMPLEXITY: O(N_ITEMS) per call.
 * Total for Phase 3: O(N_USERS² × N_ITEMS) — the dominant cost.
 * ═══════════════════════════════════════════════════════════════════════════ */
static float pearson_similarity(int u, int v)
{
    double num = 0.0;    /* numerator:   Σ(du × dv)  */
    double den_u = 0.0;  /* denominator: Σ(du²)       */
    double den_v = 0.0;  /* denominator: Σ(dv²)       */
    int    co  = 0;      /* count of co-rated items   */
    float  mu  = user_mean[u];
    float  mv  = user_mean[v];

    for (int i = 0; i < N_ITEMS; i++) {
        /* Only items rated by BOTH users contribute to the correlation. */
        if (R(u, i) != 0.0f && R(v, i) != 0.0f) {
            double du = R(u, i) - mu;   /* u's deviation from their mean */
            double dv = R(v, i) - mv;   /* v's deviation from their mean */
            num   += du * dv;
            den_u += du * du;
            den_v += dv * dv;
            co++;
        }
    }

    /* Need at least 2 co-rated items for a meaningful correlation. */
    if (co < 2) return 0.0f;

    double denom = sqrt(den_u) * sqrt(den_v);
    if (denom < 1e-10) return 0.0f;   /* zero variance in one user's ratings */

    float s = (float)(num / denom);
    if (s >  1.0f) s =  1.0f;   /* clamp rounding overshoot */
    if (s < -1.0f) s = -1.0f;
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 3b – Build the full similarity matrix
 *
 * STRATEGY – upper triangle only:
 *   Since sim(u,v) == sim(v,u), we only need to compute each pair once.
 *   The outer loop runs over all u; the inner loop runs over v > u only.
 *   We then copy the result to SIM(v,u) as well (mirror step).
 *
 *   Number of unique pairs = N*(N-1)/2.
 *   For N=1000: 499,500 Pearson calls, each scanning up to 1000 items.
 *   Total work ≈ 500 million operations — the bottleneck of the program.
 *   This is exactly the work that parallelism targets in other versions.
 *
 *   The diagonal (SIM(u,u)) is set to 1.0 explicitly because a user is
 *   perfectly correlated with themselves by definition.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void compute_all_similarities(void)
{
    /* Every user has perfect similarity with themselves. */
    for (int u = 0; u < N_USERS; u++)
        SIM(u, u) = 1.0f;

    /* Compute the upper triangle and mirror it to the lower triangle. */
    for (int u = 0; u < N_USERS; u++) {
        for (int v = u + 1; v < N_USERS; v++) {
            float s = pearson_similarity(u, v);
            SIM(u, v) = s;
            SIM(v, u) = s;   /* Pearson is symmetric: sim(u,v) = sim(v,u) */
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 4 – Predictions
 *
 * PURPOSE: for each (user u, item i) where R(u,i) == 0, predict the rating.
 *
 * FORMULA (weighted average of neighbour deviations):
 *
 *                         Σₖ [ sim(u,k) × (R(k,i) - mean(k)) ]
 *   pred(u,i) = mean(u) + ────────────────────────────────────
 *                                    Σₖ sim(u,k)
 *
 *   where k ranges over the TOP_K most similar neighbours who have rated i.
 *
 * INTUITION:
 *   Start from user u's average rating (mean(u)).  Then adjust up or down
 *   based on how neighbours rated item i RELATIVE TO THEIR OWN AVERAGES.
 *   If neighbour k (who is similar to u) rated item i much higher than
 *   their own average, it's a signal that item i is genuinely good →
 *   push pred(u,i) higher.  Weights are the similarity values.
 *
 * NEIGHBOUR SELECTION:
 *   We only use neighbours with POSITIVE similarity (negative means the
 *   users disagree — using them would push the prediction in the wrong
 *   direction).  We sort by similarity descending and take the top TOP_K.
 *
 * FALLBACK CASES:
 *   ● R(u,i) != 0  → user already rated this item; use the actual rating.
 *   ● cnt == 0     → no valid neighbours found; fall back to user's mean.
 *   ● den ≈ 0      → weights sum to near zero; fall back to user's mean.
 *   ● Clamp result to [1, 5] — the valid rating scale.
 *
 * COMPLEXITY: O(N_USERS × N_ITEMS × N_USERS) — each cell scans all users.
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * SimPair — bundles a neighbour index with its similarity value so we can
 * sort neighbours by similarity and then look up their ratings.
 */
typedef struct { int idx; float val; } SimPair;

/*
 * Comparator for qsort — sorts SimPairs by similarity in DESCENDING order.
 * qsort requires the comparator to return:
 *   negative  if a should come before b
 *   zero      if equal
 *   positive  if a should come after b
 * We want descending (largest first), so we invert the usual sign.
 */
static int cmp_sim_desc(const void *a, const void *b)
{
    float fa = ((const SimPair *)a)->val;
    float fb = ((const SimPair *)b)->val;
    return (fb > fa) - (fb < fa);   /* equivalent to (fb > fa) ? 1 : (fb < fa) ? -1 : 0 */
}

static void compute_all_predictions(void)
{
    /*
     * nbrs[] is a temporary buffer to collect eligible neighbours for one
     * (user, item) cell before sorting.  Its maximum size is N_USERS
     * (worst case: all users have rated this item).
     */
    SimPair *nbrs = (SimPair *)malloc(N_USERS * sizeof(SimPair));

    for (int u = 0; u < N_USERS; u++) {
        for (int item = 0; item < N_ITEMS; item++) {

            /* If the user already rated this item, keep the actual rating. */
            if (R(u, item) != 0.0f) { PRED(u, item) = R(u, item); continue; }

            /* Collect all neighbours who rated this item with positive sim. */
            int cnt = 0;
            for (int v = 0; v < N_USERS; v++) {
                if (v == u) continue;           /* skip self                    */
                if (R(v, item) == 0.0f) continue; /* v hasn't rated this item   */
                float s = SIM(u, v);
                if (s <= 0.0f) continue;        /* only use positively similar  */
                nbrs[cnt].idx = v;
                nbrs[cnt].val = s;
                cnt++;
            }

            /* No eligible neighbours — fall back to user's mean rating. */
            if (cnt == 0) { PRED(u, item) = user_mean[u]; continue; }

            /* Sort by similarity descending, then take the best TOP_K. */
            qsort(nbrs, cnt, sizeof(SimPair), cmp_sim_desc);
            int k = (cnt < TOP_K) ? cnt : TOP_K;

            /* Compute the weighted-average deviation and add to user's mean. */
            double num = 0.0, den = 0.0;
            for (int j = 0; j < k; j++) {
                float s = nbrs[j].val;
                /* Neighbour's deviation: how much they rated this item above
                 * or below their own average — removes individual bias. */
                num += s * (R(nbrs[j].idx, item) - user_mean[nbrs[j].idx]);
                den += s;
            }

            float pred = (den > 1e-10)
                         ? user_mean[u] + (float)(num / den)
                         : user_mean[u];

            /* Clamp to the valid rating range. */
            if (pred < 1.0f) pred = 1.0f;
            if (pred > 5.0f) pred = 5.0f;
            PRED(u, item) = pred;
        }
    }

    free(nbrs);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PHASE 5 – Evaluation
 *
 * Mean Absolute Error (MAE):
 *
 *   MAE = (1 / |T|) × Σ |pred(u,i) - actual(u,i)|
 *                      (u,i) ∈ T
 *
 * where T is the held-out test set built during Phase 1.
 *
 * MAE is in the same units as the ratings (1–5 scale).
 * A MAE of 0.5 means predictions are off by half a star on average.
 * Lower is better.
 *
 * This value MUST match across all parallel versions for the same input
 * (within ±0.001 for floating-point rounding differences).
 * ═══════════════════════════════════════════════════════════════════════════ */
static float evaluate_mae(void)
{
    if (test_size == 0) return 0.0f;
    double err = 0.0;
    for (int t = 0; t < test_size; t++)
        err += fabs(PRED(test_set[t].user, test_set[t].item) - test_set[t].rating);
    return (float)(err / test_size);
}

/* Root Mean Square Error: penalises large errors more than MAE. */
static float evaluate_rmse(void)
{
    if (test_size == 0) return 0.0f;
    double sq = 0.0;
    for (int t = 0; t < test_size; t++) {
        double d = PRED(test_set[t].user, test_set[t].item) - test_set[t].rating;
        sq += d * d;
    }
    return (float)sqrt(sq / test_size);
}

/*
 * similarity_checksum() — a fast sanity check across versions.
 *
 * Summing all N_USERS² elements of sim_matrix gives a single scalar that
 * is sensitive to any wrong value in the matrix.  If two versions produce
 * the same checksum, their similarity matrices are very likely identical.
 * If they differ, there is a bug in one version's similarity computation.
 */
static double similarity_checksum(void)
{
    double s = 0.0;
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
     * Parse optional CLI arguments.
     * argc is the argument count (including the program name).
     * argv[1] = num_users, argv[2] = num_items (both optional).
     * atoi() converts a string to an integer ("1000" → 1000).
     */
    N_USERS = (argc >= 2) ? atoi(argv[1]) : DEFAULT_USERS;
    N_ITEMS = (argc >= 3) ? atoi(argv[2]) : DEFAULT_ITEMS;

    if (N_USERS <= 0 || N_ITEMS <= 0) {
        fprintf(stderr, "Usage: %s [num_users] [num_items]\n", argv[0]);
        return EXIT_FAILURE;
    }

    printf("=== Pearson Correlation Recommender – Serial Version ===\n");
    printf("    Users: %d | Items: %d | Top-K: %d\n\n",
           N_USERS, N_ITEMS, TOP_K);

    /*
     * t0 and t1 are used to bracket each phase with wall-clock timestamps.
     * t_sim and t_pred are saved so we can report Total = sim + pred at the end.
     */
    double t0, t1, t_sim, t_pred;

    alloc_arrays();

    /* ── Phase 1: Data generation ──────────────────────────────────────── */
    t0 = now_sec();
    generate_data();
    t1 = now_sec();
    printf("[Timing] Data generation    : %.4f s\n", t1 - t0);

    /* ── Phase 2: User means ────────────────────────────────────────────── */
    t0 = now_sec();
    compute_user_means();
    t1 = now_sec();
    printf("[Timing] User mean compute  : %.4f s\n", t1 - t0);

    /* ── Phase 3: Similarity matrix (the bottleneck) ────────────────────── */
    t0 = now_sec();
    compute_all_similarities();
    t1 = now_sec();
    t_sim = t1 - t0;
    printf("[Timing] Similarity matrix  : %.4f s\n", t_sim);
    printf("[Check]  Sim-matrix checksum: %.6f\n", similarity_checksum());

    /* ── Phase 4: Predictions ───────────────────────────────────────────── */
    t0 = now_sec();
    compute_all_predictions();
    t1 = now_sec();
    t_pred = t1 - t0;
    printf("[Timing] Prediction phase   : %.4f s\n", t_pred);

    /* ── Phase 5: Evaluation ────────────────────────────────────────────── */
    printf("[Eval]   MAE on test set    : %.4f  (test size: %d)\n",
           evaluate_mae(), test_size);
    printf("[Eval]   RMSE on test set   : %.4f  (test size: %d)\n",
           evaluate_rmse(), test_size);
    printf("[Timing] Total (sim+pred)   : %.4f s\n", t_sim + t_pred);

    /* Print a small sample of predictions for a visual sanity check. */
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
