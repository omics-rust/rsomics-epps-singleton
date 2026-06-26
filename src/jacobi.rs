//! Symmetric eigendecomposition via cyclic Jacobi rotations.
//!
//! `est_cov` is symmetric positive-semidefinite (a sum of scaled covariance
//! matrices), so its SVD coincides with its eigendecomposition: singular value
//! σ_i = λ_i, singular vectors = eigenvectors. The Jacobi sweep is
//! backward-stable, so the pseudo-inverse it builds tracks `np.linalg.pinv`
//! across the moderately-conditioned matrices reached for three or four
//! frequencies, where a plain Gauss-Jordan inverse drifts. Once `est_cov`
//! becomes ill-conditioned (cond ≳ 1e6, common at five frequencies with
//! clustered `t`) the statistic is dominated by `1/σ_min` with σ_min barely
//! above the rcond cutoff; there no two SVD implementations agree — numpy's own
//! `dgesdd`, `dgesvd`, and `eigh` paths diverge on the statistic — and exact
//! reproduction is not achievable for anyone.

/// Eigenvalues and column-stacked eigenvectors of a symmetric `dim×dim` matrix.
///
/// `eigvecs[i*dim + j]` is component `i` of eigenvector `j`; `eigvals[j]` is its
/// eigenvalue. Eigenvalues are not sorted.
pub fn eigh_symmetric(a: &[f64], dim: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        v[i * dim + i] = 1.0;
    }

    let off = |m: &[f64]| -> f64 {
        let mut s = 0.0;
        for p in 0..dim {
            for q in (p + 1)..dim {
                s += 2.0 * m[p * dim + q] * m[p * dim + q];
            }
        }
        s.sqrt()
    };

    let frob = |m: &[f64]| -> f64 {
        let mut s = 0.0;
        for &x in m {
            s += x * x;
        }
        s.sqrt()
    };

    let tol = frob(&m) * f64::EPSILON;
    for _ in 0..100 {
        if off(&m) <= tol {
            break;
        }
        for p in 0..dim {
            for q in (p + 1)..dim {
                let apq = m[p * dim + q];
                if apq == 0.0 {
                    continue;
                }
                let app = m[p * dim + p];
                let aqq = m[q * dim + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                for k in 0..dim {
                    let mkp = m[k * dim + p];
                    let mkq = m[k * dim + q];
                    m[k * dim + p] = c * mkp - s * mkq;
                    m[k * dim + q] = s * mkp + c * mkq;
                }
                for k in 0..dim {
                    let mpk = m[p * dim + k];
                    let mqk = m[q * dim + k];
                    m[p * dim + k] = c * mpk - s * mqk;
                    m[q * dim + k] = s * mpk + c * mqk;
                }
                for k in 0..dim {
                    let vkp = v[k * dim + p];
                    let vkq = v[k * dim + q];
                    v[k * dim + p] = c * vkp - s * vkq;
                    v[k * dim + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let eigvals: Vec<f64> = (0..dim).map(|i| m[i * dim + i]).collect();
    (eigvals, v)
}

/// Moore-Penrose pseudo-inverse of a symmetric PSD matrix, matching
/// `np.linalg.pinv` (`rcond = 1e-15`: drop eigenvalues `≤ rcond·max`), with the
/// rank `np.linalg.matrix_rank` reports for that pseudo-inverse.
///
/// `matrix_rank`'s threshold on the pseudo-inverse's singular values
/// (`1/λ_i` for the kept `λ_i`) is `max(1/λ_i)·dim·eps`; for a clean
/// eigenvalue gap this counts exactly the eigenvalues kept above the `pinv`
/// cutoff, so the kept count is the chi-squared degrees of freedom.
pub fn pinv_symmetric(a: &[f64], dim: usize) -> (Vec<f64>, usize) {
    let (eigvals, eigvecs) = eigh_symmetric(a, dim);

    let max_lambda = eigvals.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = 1e-15 * max_lambda;

    let mut inv = vec![0.0_f64; dim * dim];
    let mut max_inv_sv = 0.0_f64;
    let mut kept = Vec::new();
    for (j, &lambda) in eigvals.iter().enumerate() {
        if lambda > cutoff {
            let inv_lambda = 1.0 / lambda;
            max_inv_sv = max_inv_sv.max(inv_lambda);
            kept.push((j, inv_lambda));
        }
    }
    for &(j, inv_lambda) in &kept {
        for r in 0..dim {
            let vr = eigvecs[r * dim + j];
            for c in 0..dim {
                inv[r * dim + c] += inv_lambda * vr * eigvecs[c * dim + j];
            }
        }
    }

    let rank_tol = max_inv_sv * dim as f64 * f64::EPSILON;
    let rank = kept
        .iter()
        .filter(|&&(_, inv_lambda)| inv_lambda > rank_tol)
        .count();
    (inv, rank)
}

#[cfg(test)]
mod tests {
    use super::{eigh_symmetric, pinv_symmetric};

    fn matmul(a: &[f64], b: &[f64], dim: usize) -> Vec<f64> {
        let mut c = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut s = 0.0;
                for k in 0..dim {
                    s += a[i * dim + k] * b[k * dim + j];
                }
                c[i * dim + j] = s;
            }
        }
        c
    }

    #[test]
    fn eigh_reconstructs_matrix() {
        let dim = 3;
        let a = [2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let (vals, vecs) = eigh_symmetric(&a, dim);
        let mut recon = vec![0.0; dim * dim];
        for j in 0..dim {
            for r in 0..dim {
                for c in 0..dim {
                    recon[r * dim + c] += vals[j] * vecs[r * dim + j] * vecs[c * dim + j];
                }
            }
        }
        for (g, w) in recon.iter().zip(a.iter()) {
            assert!((g - w).abs() < 1e-12);
        }
    }

    #[test]
    fn pinv_of_full_rank_is_true_inverse() {
        let dim = 3;
        let a = [4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let (inv, rank) = pinv_symmetric(&a, dim);
        assert_eq!(rank, 3);
        let prod = matmul(&a, &inv, dim);
        for i in 0..dim {
            for j in 0..dim {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((prod[i * dim + j] - want).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn pinv_drops_tiny_eigenvalue() {
        // Rank-2 PSD matrix: outer products of two orthogonal vectors.
        let dim = 3;
        let u = [1.0, 0.0, 0.0];
        let w = [0.0, 1.0, 0.0];
        let mut a = vec![0.0; dim * dim];
        for r in 0..dim {
            for c in 0..dim {
                a[r * dim + c] = 3.0 * u[r] * u[c] + 5.0 * w[r] * w[c];
            }
        }
        let (_, rank) = pinv_symmetric(&a, dim);
        assert_eq!(rank, 2);
    }
}
