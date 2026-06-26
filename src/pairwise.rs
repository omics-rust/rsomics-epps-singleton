//! NumPy's `pairwise_sum` reduction order.
//!
//! NumPy reduces with an 8-accumulator unrolled leaf (`n <= 128`) and a
//! half-split recursion rounding the split down to a multiple of 8. Matching
//! this exactly keeps a sum-over-N statistic bit-identical to NumPy at large N,
//! where a naive left fold drifts by ~1e-11.

/// Sum `xs` in NumPy's pairwise order.
#[must_use]
pub fn pairwise_sum(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n < 8 {
        let mut acc = xs[0];
        for &v in &xs[1..] {
            acc += v;
        }
        return acc;
    }
    if n <= 128 {
        let mut acc = [xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7]];
        let mut i = 8;
        while i + 8 <= n {
            for (a, &v) in acc.iter_mut().zip(&xs[i..i + 8]) {
                *a += v;
            }
            i += 8;
        }
        let mut res =
            ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        while i < n {
            res += xs[i];
            i += 1;
        }
        return res;
    }
    let mut half = n / 2;
    half -= half % 8;
    pairwise_sum(&xs[..half]) + pairwise_sum(&xs[half..])
}

#[cfg(test)]
mod tests {
    use super::pairwise_sum;

    #[test]
    fn small_matches_naive() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(pairwise_sum(&xs), 15.0);
    }

    #[test]
    fn large_n_is_stable() {
        let xs: Vec<f64> = (0..1_000_000).map(|i| (i as f64) * 1e-7 + 0.1).collect();
        let pw = pairwise_sum(&xs);
        let mut naive = 0.0;
        for &v in &xs {
            naive += v;
        }
        // pairwise is the accurate one; assert it lands near the true value
        let exact: f64 = 1_000_000.0 * 0.1 + 1e-7 * (999_999.0 * 1_000_000.0 / 2.0);
        assert!((pw - exact).abs() <= (naive - exact).abs() + 1e-6);
    }
}
