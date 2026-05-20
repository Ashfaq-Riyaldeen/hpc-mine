#!/usr/bin/env bash
# =============================================================================
# Windows benchmark runner — HPC Group 11   (INTERLEAVED, min-of-N)
# Platform: Windows 11, MinGW-w64 gcc 13.2.0, MS-MPI 10.1, 4 cores / 8 threads
# Problem: 1000 x 1000, SEED=42, TOP_K=20
#
# Methodology: every configuration is measured once per ROUND, and ROUNDS
# rounds are run. The analysis takes the MINIMUM time per phase across rounds.
# Interleaving configs within a round means each version experiences a similar
# distribution of background OS load, making cross-version comparison fair on
# a noisy consumer machine. CUDA is code-only (no GPU/toolkit).
# =============================================================================
set -u
N=1000; M=1000; ROUNDS=5
OUT="results/timing_raw.txt"

echo "HPC Group 11 - Windows Benchmark Run  $(date)"            >  "$OUT"
echo "Problem size: $N $M  |  SEED=42  |  TOP_K=20  |  ROUNDS=$ROUNDS (min per config)" >> "$OUT"
echo "Method: interleaved configs per round; minimum time per phase reported." >> "$OUT"
echo "Platform: Windows 11 | MinGW gcc 13.2.0 | MS-MPI 10.1 | 4 cores / 8 logical" >> "$OUT"
echo "==========================================" >> "$OUT"

emit () { echo "" >> "$OUT"; echo "=== $1 [round $2] ===" >> "$OUT"; }

for r in $(seq 1 "$ROUNDS"); do
  emit "Serial (1 workers)" "$r";        ./serial_rec.exe "$N" "$M"                          >> "$OUT" 2>&1
  for t in 1 2 4 8; do
    emit "OpenMP ($t workers)" "$r";     OMP_NUM_THREADS="$t" ./openmp_rec.exe "$N" "$M"     >> "$OUT" 2>&1
  done
  for t in 1 2 4 8; do
    emit "Pthreads ($t workers)" "$r";   ./pthreads_rec.exe "$N" "$M" "$t"                   >> "$OUT" 2>&1
  done
  for p in 1 2 4 8; do
    emit "MPI ($p workers)" "$r";        mpiexec -n "$p" ./mpi_rec.exe "$N" "$M"             >> "$OUT" 2>&1
  done
  emit "Hybrid_2x4 (8 workers)" "$r"; mpiexec -n 2 -env OMP_NUM_THREADS 4 ./hybrid_rec.exe "$N" "$M" >> "$OUT" 2>&1
  emit "Hybrid_4x2 (8 workers)" "$r"; mpiexec -n 4 -env OMP_NUM_THREADS 2 ./hybrid_rec.exe "$N" "$M" >> "$OUT" 2>&1
  emit "Hybrid_8x1 (8 workers)" "$r"; mpiexec -n 8 -env OMP_NUM_THREADS 1 ./hybrid_rec.exe "$N" "$M" >> "$OUT" 2>&1
  emit "Hybrid_1x8 (8 workers)" "$r"; mpiexec -n 1 -env OMP_NUM_THREADS 8 ./hybrid_rec.exe "$N" "$M" >> "$OUT" 2>&1
  echo "ROUND $r DONE"
done
echo "" >> "$OUT"; echo "DONE" >> "$OUT"; echo "ALL DONE"
