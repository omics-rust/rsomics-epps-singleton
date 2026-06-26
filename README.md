# rsomics-epps-singleton

Epps-Singleton (ES) two-sample test on the empirical characteristic function —
a value-exact Rust port of `scipy.stats.epps_singleton_2samp`.

The ES test asks whether two samples come from the same distribution. Unlike
Kolmogorov-Smirnov it does not assume a continuous distribution, and Epps &
Singleton report higher power than KS across many cases — it is the recommended
choice for discrete or mixed samples.

## Usage

```
rsomics-epps-singleton X.tsv Y.tsv [--t 0.4,0.8]
```

Each input is a single column of numbers (one value per line; `-` reads stdin).
Output is one line, `statistic<TAB>p`. `--t` sets the (positive, distinct)
frequencies at which the characteristic function is sampled; the default
`0.4,0.8` is the pair recommended in the original paper. The asymptotic
chi-squared p-value has `df = 2·len(t)` (4 by default). Note `--t` is long-only;
`-t`/`--threads` is the thread count, not the frequency list.

```
$ rsomics-epps-singleton x.tsv y.tsv
15.177899584008259	0.004346113320362424
```

Supplying more frequencies (`--t 0.4,0.8,1.2,1.6`) is fully supported and stays
value-exact against SciPy. As in SciPy, a poor choice of `t` — many tightly
clustered frequencies — drives `est_cov` toward singularity; the statistic is
then dominated by `1/σ_min` of a barely-resolvable direction, where no two SVD
implementations agree (SciPy's own `dgesdd`, `dgesvd`, and `eigh` paths differ
by ~1e-4). This matches the paper's caution that such `t` make the test
inconsistent. Spread the frequencies (e.g. `0.6,1.2,1.8,2.4,3.0`) to stay in the
well-conditioned, reproducible regime.

If every observation is identical the pooled IQR is zero, so SciPy's `pinv`
raises `LinAlgError: SVD did not converge`; this port exits non-zero with the
same `SVD did not converge` message rather than emitting a meaningless number.

## Method

Following the SciPy implementation: the pooled samples set a semi-interquartile
range `σ = IQR(x ∪ y) / 2` that rescales the frequencies, `ts = t / σ`. At each
scaled frequency the cos and sin moments form a `2k`-row `g`-vector per sample;
the biased covariance of each is combined into `est_cov = (n/nx)·cov_x +
(n/ny)·cov_y`, and the statistic is the quadratic form

```
W = n · (ḡx − ḡy)' · est_cov⁺ · (ḡx − ḡy)
```

asymptotically chi-squared on `df = rank(est_cov⁺)`. The pseudo-inverse `est_cov⁺`
matches `numpy.linalg.pinv` (Moore-Penrose, `rcond = 1e-15`): because `est_cov`
is symmetric positive-semidefinite, its SVD equals its eigendecomposition, so a
backward-stable cyclic Jacobi eigensolver yields the eigenpairs, eigenvalues
`≤ rcond·max` are dropped, and the kept count is the chi-squared `df`. When
`max(nx, ny) < 25` the small-sample correction
`W ·= 1 / (1 + n^-0.45 + 10.1·(nx^-1.7 + ny^-1.7))` from the paper is applied.
The p-value is `chi2.sf(W, df)`.

The p-value path is a direct port of the Cephes `igamc` (regularized upper
incomplete gamma) that SciPy itself calls, so it is bit-identical rather than
merely close. Parsing uses a single zero-allocation pass (`fast-float2`,
Lemire's algorithm) and the term reductions reproduce NumPy's `pairwise_sum`
order, keeping the statistic bit-stable at large N.

## Origin

This crate is an independent Rust reimplementation of
`scipy.stats.epps_singleton_2samp` based on:

- The published method: T. W. Epps and K. J. Singleton, "An omnibus test for the
  two-sample problem using the empirical characteristic function", *Journal of
  Statistical Computation and Simulation* 26, p. 177–203, 1986.
- The SciPy implementation `scipy/stats/_hypotests.py::epps_singleton_2samp`
  (SciPy 1.17.1, BSD-3-Clause) for the exact matrix algebra (biased covariance,
  the `(n-1)` correction, and the small-sample rescale).
- The Cephes `igam`/`igamc` incomplete-gamma routines (public domain, as
  redistributed in SciPy) for the chi-squared survival function.

SciPy is BSD-3-Clause licensed; reading and citing its source is permitted.
Goldens were produced by running SciPy 1.17.1 once and committed, so the compat
test runs without SciPy.

License: MIT OR Apache-2.0.
Upstream credit: SciPy (https://scipy.org, BSD-3-Clause).
