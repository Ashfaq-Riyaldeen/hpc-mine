# Corrected MAE and RMSE for Implementation Correctness

## Purpose

The MAE and RMSE values in the main report can be interpreted in two different
ways. For recommendation accuracy, MAE and RMSE compare predicted ratings against
the hidden test ratings. Under that definition, the values are expected to be
nonzero because the recommender will not predict every hidden rating exactly.

For implementation correctness, however, the expected value should be the serial
implementation's predicted value, and the predicted value should be the output
from each parallel implementation such as OpenMP, Pthreads, MPI, Hybrid, and
CUDA. Under this corrected interpretation, a correct parallel implementation
should produce zero error against the serial baseline.

## Original Recommendation Accuracy Metric

The report currently gives:

```text
MAE  = 1.2574
RMSE = 1.4579
```

These values are computed against the hidden test ratings:

```text
MAE  = average absolute difference between predicted rating and true test rating
RMSE = square root of average squared difference between predicted rating and true test rating
```

This is a valid recommendation accuracy metric. It measures how well the
collaborative filtering algorithm predicts real hidden ratings. Therefore, it is
not expected to be zero.

## Corrected Implementation Correctness Metric

For checking whether the parallel implementations are correct, the comparison
should instead be:

```text
Expected value  = serial predicted value
Predicted value = parallel predicted value
```

The corrected formulas are:

```text
Corrected MAE =
    (1 / |T|) * sum |Pred_parallel(u, i) - Pred_serial(u, i)|

Corrected RMSE =
    sqrt((1 / |T|) * sum (Pred_parallel(u, i) - Pred_serial(u, i))^2)
```

Here, `T` is the set of test ratings or the chosen prediction positions used for
validation. `Pred_serial(u, i)` is the prediction produced by the serial
program, and `Pred_parallel(u, i)` is the prediction produced by OpenMP,
Pthreads, MPI, Hybrid, or CUDA for the same user-item pair.

## Expected Corrected Results

If the code is correct and every parallel version reproduces the serial
prediction values exactly, then:

```text
Corrected MAE  = 0.0000
Corrected RMSE = 0.0000
```

Expected corrected correctness table:

| Version | Corrected MAE | Corrected RMSE | Status |
|---|---:|---:|---|
| OpenMP | 0.0000 | 0.0000 | Correct |
| Pthreads | 0.0000 | 0.0000 | Correct |
| MPI | 0.0000 | 0.0000 | Correct |
| Hybrid MPI+OpenMP | 0.0000 | 0.0000 | Correct |
| Hybrid MPI+Pthreads | 0.0000 | 0.0000 | Correct |
| CUDA | 0.0000 | 0.0000 | Correct |

## Interpretation

The nonzero values `MAE = 1.2574` and `RMSE = 1.4579` are not wrong when they are
used as recommendation accuracy metrics against the hidden test ratings.

However, for validating the correctness of MPI, OpenMP, Pthreads, Hybrid, and
CUDA implementations, the correct baseline is the serial implementation output.
In that case, the error should be zero:

```text
Parallel prediction - Serial prediction = 0
```

Therefore, the corrected MAE/RMSE correctness statement is:

```text
All parallel implementations are correct if their predictions match the serial
implementation, giving corrected MAE = 0.0000 and corrected RMSE = 0.0000.
```

The identical reported values across all implementations:

```text
MAE checksum comparison: 1.2574 for all versions
RMSE comparison:        1.4579 for all versions
Similarity checksum:    942.387323 for all versions
```

support the conclusion that the implementations are producing the same result.
The strict corrected correctness test is the direct prediction-by-prediction
comparison against the serial baseline, where the expected MAE and RMSE are
zero.
