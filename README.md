## 1. Serial Version

### Compile (run once from the folder)

```bash
gcc -O2 -Wall -o serial_rec serial/serial_recommender.c -lm
```

### Run

```bash
# Default: 1000 users, 1000 items
./serial_rec

# Custom sizes: ./serial_rec [num_users] [num_items]
./serial_rec 500 300
./serial_rec 2000 1500
```

---

## 2. OpenMP Shared Memory Version

### Compile (run once from the folder)

```bash
gcc -O2 -Wall -fopenmp -o openmp_rec openmp/openmp_recommender.c -lm
```

### Run – thread count via `OMP_NUM_THREADS`

```bash
# Default sizes (1000 users, 1000 items) – vary threads only
./openmp_rec
OMP_NUM_THREADS=2  ./openmp_rec
OMP_NUM_THREADS=4  ./openmp_rec
OMP_NUM_THREADS=8  ./openmp_rec

# Custom sizes: ./openmp_rec [num_users] [num_items]
OMP_NUM_THREADS=4  ./openmp_rec 500 300
OMP_NUM_THREADS=8  ./openmp_rec 2000 1500
```

---

## 3. Pthreads Shared Memory Version

### Compile (run once from the folder)

```bash
gcc -O2 -Wall -o pthreads_rec pthreads/pthreads_recommender.c -lpthread -lm
```

### Run – thread count as the third argument

```bash
# Default: 1000 users, 1000 items, 4 threads
./pthreads_rec

# Vary thread count only (fixed 1000 x 1000 problem)
./pthreads_rec 1000 1000 1
./pthreads_rec 1000 1000 2
./pthreads_rec 1000 1000 4
./pthreads_rec 1000 1000 8

# Custom sizes with explicit thread count
./pthreads_rec 500  500  2
./pthreads_rec 2000 1500 8
```

---

## 4. Speedup Benchmark (for the report)

```bash
# Compile all three shared-memory versions
gcc -O2 -Wall -o serial_rec   serial/serial_recommender.c              -lm
gcc -O2 -Wall -fopenmp -o openmp_rec   openmp/openmp_recommender.c     -lm
gcc -O2 -Wall -o pthreads_rec pthreads/pthreads_recommender.c -lpthread -lm

# 1. Serial baseline
./serial_rec 1000 1000

# 2. OpenMP – vary thread count
OMP_NUM_THREADS=1  ./openmp_rec 1000 1000
OMP_NUM_THREADS=2  ./openmp_rec 1000 1000
OMP_NUM_THREADS=4  ./openmp_rec 1000 1000
OMP_NUM_THREADS=8  ./openmp_rec 1000 1000

# 3. Pthreads – vary thread count
./pthreads_rec 1000 1000 1
./pthreads_rec 1000 1000 2
./pthreads_rec 1000 1000 4
./pthreads_rec 1000 1000 8
```

Record the `[Timing] Total (sim+pred)` line from each run.
Speedup = Serial_total / Parallel_total.