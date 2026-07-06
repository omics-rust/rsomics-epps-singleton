//! Epps-Singleton two-sample test on the empirical characteristic function.
//!
//! Port of `scipy.stats.epps_singleton_2samp` (scipy 1.17.1). The pooled
//! semi-interquartile range rescales the frequencies `t`; at each scaled
//! frequency we evaluate the empirical characteristic function via cos/sin
//! moments, stack them into a `2k`-row `g`-vector per sample, estimate the
//! (biased) covariance, and form the quadratic form
//! `W = n · g_diff' · est_cov⁺ · g_diff`, asymptotically chi-squared on
//! `df = rank(est_cov⁺)` (= `2k` for non-degenerate inputs). A small-sample
//! correction scales `W` when `max(nx, ny) < 25`.

use serde::Serialize;

use crate::chi2::chi2_sf;
use crate::jacobi::pinv_symmetric;
use crate::pairwise::pairwise_sum;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EsResult {
    pub statistic: f64,
    pub pvalue: f64,
}

#[derive(Debug)]
pub enum EsError {
    TooFew { nx: usize, ny: usize },
    NonPositiveT,
    SvdDidNotConverge,
}

impl std::fmt::Display for EsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EsError::TooFew { nx, ny } => write!(
                f,
                "x and y should have at least 5 elements, but len(x) = {nx} and len(y) = {ny}."
            ),
            EsError::NonPositiveT => write!(f, "t must contain positive elements only."),
            EsError::SvdDidNotConverge => write!(f, "SVD did not converge"),
        }
    }
}

/// Epps-Singleton statistic and asymptotic chi-squared p-value.
///
/// `t` holds the (positive, distinct) frequencies at which the empirical
/// characteristic function is sampled; the default `(0.4, 0.8)` is from Epps &
/// Singleton (1986).
pub fn epps_singleton(x: &[f64], y: &[f64], t: &[f64]) -> Result<EsResult, EsError> {
    // scipy's default nan_policy='propagate' catches only NaN (not inf), and it
    // fires before frequency validation, so a NaN sample short-circuits to
    // (nan, nan) regardless of `t`.
    if x.iter().chain(y).any(|v| v.is_nan()) {
        return Ok(EsResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }

    let nx = x.len();
    let ny = y.len();
    if nx < 5 || ny < 5 {
        return Err(EsError::TooFew { nx, ny });
    }
    if t.iter().any(|&ti| ti <= 0.0) {
        return Err(EsError::NonPositiveT);
    }

    // An inf is not caught by nan_policy, so scipy reaches the `t > 0` check
    // above and only then lets inf flow through the transform to (nan, nan).
    // Short-circuiting here — after validation — reproduces both: an invalid
    // `t` still fails loud, while a valid `t` yields (nan, nan) on inf input.
    if x.iter().chain(y).any(|v| v.is_infinite()) {
        return Ok(EsResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }
    let n = nx + ny;
    let k = t.len();
    let dim = 2 * k;

    let sigma = pooled_semi_iqr(x, y);
    let ts: Vec<f64> = t.iter().map(|&ti| ti / sigma).collect();

    // g[row][col]: rows 0..k are cos(ts*sample), rows k..2k are sin(ts*sample).
    let gx = g_matrix(&ts, x);
    let gy = g_matrix(&ts, y);

    let cov_x = biased_cov(&gx, dim, nx);
    let cov_y = biased_cov(&gy, dim, ny);

    let scale_x = n as f64 / nx as f64;
    let scale_y = n as f64 / ny as f64;
    let mut est_cov = vec![0.0_f64; dim * dim];
    for idx in 0..dim * dim {
        est_cov[idx] = scale_x * cov_x[idx] + scale_y * cov_y[idx];
    }

    // A degenerate pooled IQR makes sigma = 0, so ts = inf and the cos/sin
    // moments become NaN; scipy's pinv then fails the same way LAPACK does.
    if est_cov.iter().any(|v| !v.is_finite()) {
        return Err(EsError::SvdDidNotConverge);
    }

    let g_diff: Vec<f64> = (0..dim)
        .map(|r| row_mean(&gx, r, nx) - row_mean(&gy, r, ny))
        .collect();

    let (inv, rank) = pinv_symmetric(&est_cov, dim);

    // W = n · g_diff' · inv · g_diff
    let mut w = 0.0_f64;
    for i in 0..dim {
        let mut acc = 0.0_f64;
        for j in 0..dim {
            acc += inv[i * dim + j] * g_diff[j];
        }
        w += g_diff[i] * acc;
    }
    w *= n as f64;

    if nx.max(ny) < 25 {
        let corr = 1.0
            / (1.0
                + (n as f64).powf(-0.45)
                + 10.1 * ((nx as f64).powf(-1.7) + (ny as f64).powf(-1.7)));
        w *= corr;
    }

    let p = chi2_sf(rank as f64, w);
    Ok(EsResult {
        statistic: w,
        pvalue: p,
    })
}

/// Semi-interquartile range of the pooled sample: `iqr(x ∪ y) / 2` with numpy's
/// default linear-interpolation percentile.
fn pooled_semi_iqr(x: &[f64], y: &[f64]) -> f64 {
    let mut pool: Vec<f64> = Vec::with_capacity(x.len() + y.len());
    pool.extend_from_slice(x);
    pool.extend_from_slice(y);
    pool.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (percentile_linear(&pool, 75.0) - percentile_linear(&pool, 25.0)) / 2.0
}

/// numpy `percentile(..., method='linear')` on an already-sorted slice.
fn percentile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let pos = (n - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// `g[r][c]` flattened row-major: rows `0..k` are `cos(ts[r]·sample[c])`,
/// rows `k..2k` are `sin(ts[r-k]·sample[c])`.
fn g_matrix(ts: &[f64], sample: &[f64]) -> Vec<f64> {
    let k = ts.len();
    let ncol = sample.len();
    let mut g = vec![0.0_f64; 2 * k * ncol];
    for (r, &tr) in ts.iter().enumerate() {
        let cos_row = r * ncol;
        let sin_row = (k + r) * ncol;
        for (c, &v) in sample.iter().enumerate() {
            let arg = tr * v;
            g[cos_row + c] = arg.cos();
            g[sin_row + c] = arg.sin();
        }
    }
    g
}

/// Mean of row `r` of a `dim×ncol` matrix, using NumPy's pairwise reduction.
fn row_mean(g: &[f64], r: usize, ncol: usize) -> f64 {
    pairwise_sum(&g[r * ncol..r * ncol + ncol]) / ncol as f64
}

/// Biased (`ddof = 0`) covariance of the `dim×ncol` matrix `g` — scipy's
/// `np.cov(g) * (ncol-1)/ncol`, i.e. `Σ (gi-mi)(gj-mj) / ncol`.
fn biased_cov(g: &[f64], dim: usize, ncol: usize) -> Vec<f64> {
    let means: Vec<f64> = (0..dim).map(|r| row_mean(g, r, ncol)).collect();
    let mut cov = vec![0.0_f64; dim * dim];
    let mut prod = vec![0.0_f64; ncol];
    for i in 0..dim {
        for j in i..dim {
            let ri = i * ncol;
            let rj = j * ncol;
            for c in 0..ncol {
                prod[c] = (g[ri + c] - means[i]) * (g[rj + c] - means[j]);
            }
            let s = pairwise_sum(&prod) / ncol as f64;
            cov[i * dim + j] = s;
            cov[j * dim + i] = s;
        }
    }
    cov
}

#[cfg(test)]
mod tests {
    use super::{EsError, epps_singleton};

    const T: [f64; 2] = [0.4, 0.8];
    fn s(n: usize, off: f64) -> Vec<f64> {
        (0..n).map(|i| i as f64 + off).collect()
    }

    #[test]
    fn nan_short_circuits_before_t_validation() {
        // scipy nan_policy='propagate' catches NaN before the `t > 0` check.
        let mut x = s(6, 0.0);
        x[3] = f64::NAN;
        let bad_t = [0.4, -0.8];
        let r = epps_singleton(&x, &s(6, 0.0), &bad_t).unwrap();
        assert!(r.statistic.is_nan() && r.pvalue.is_nan());
    }

    #[test]
    fn inf_with_invalid_t_fails_loud() {
        // inf is not caught by nan_policy, so an invalid `t` still raises.
        let mut x = s(6, 0.0);
        x[3] = f64::INFINITY;
        assert!(matches!(
            epps_singleton(&x, &s(6, 0.0), &[0.4, -0.8]),
            Err(EsError::NonPositiveT)
        ));
    }

    #[test]
    fn inf_with_valid_t_propagates_nan() {
        let mut x = s(6, 0.0);
        x[3] = f64::INFINITY;
        let r = epps_singleton(&x, &s(6, 0.0), &T).unwrap();
        assert!(r.statistic.is_nan() && r.pvalue.is_nan());
    }

    #[test]
    fn finite_with_invalid_t_fails_loud() {
        assert!(matches!(
            epps_singleton(&s(6, 0.0), &s(6, 0.5), &[0.4, 0.0]),
            Err(EsError::NonPositiveT)
        ));
    }
}
