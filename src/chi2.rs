//! Upper-tail chi-squared survival function via a Cephes `igamc` port.
//!
//! scipy computes the Epps-Singleton p-value as `chi2.sf(W, df)`, which inside
//! SciPy is `cephes.chdtrc(df, W) = igamc(df/2, W/2)` — the regularized upper
//! incomplete gamma. We port Cephes `igam`/`igamc` directly so the tail matches
//! SciPy's special-function path rather than a different incomplete-gamma series.

const MACHEP: f64 = 1.110_223_024_625_156_5e-16;
const BIG: f64 = 4.503_599_627_370_496e15;
const BIGINV: f64 = 2.220_446_049_250_313e-16;

/// Chi-squared survival function P(X > x) with `df` degrees of freedom.
#[must_use]
pub fn chi2_sf(df: f64, x: f64) -> f64 {
    if x.is_nan() || df.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    igamc(0.5 * df, 0.5 * x)
}

/// Regularized lower incomplete gamma P(a, x), Cephes power-series branch.
fn igam(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x > 1.0 && x > a {
        return 1.0 - igamc(a, x);
    }
    let ax = a * x.ln() - x - lgam(a);
    if ax < -MAXLOG {
        return 0.0;
    }
    let ax = ax.exp();

    let mut r = a;
    let mut c = 1.0;
    let mut ans = 1.0;
    loop {
        r += 1.0;
        c *= x / r;
        ans += c;
        if c / ans <= MACHEP {
            break;
        }
    }
    ans * ax / a
}

/// Regularized upper incomplete gamma Q(a, x), Cephes continued-fraction branch.
///
/// Short-circuits on a NaN argument so a NaN statistic returns NaN rather than
/// spinning the convergence loop forever — `t <= MACHEP` is never reached for
/// NaN `t`.
#[must_use]
pub fn igamc(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 || a <= 0.0 {
        return 1.0;
    }
    if x < 1.0 || x < a {
        return 1.0 - igam(a, x);
    }

    let ax = a * x.ln() - x - lgam(a);
    if ax < -MAXLOG {
        return 0.0;
    }
    let ax = ax.exp();

    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0.0;
    let mut pkm2 = 1.0;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut ans = pkm1 / qkm1;

    loop {
        c += 1.0;
        y += 1.0;
        z += 2.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;
        let t = if qk != 0.0 {
            let r = pk / qk;
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        } else {
            1.0
        };
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if pk.abs() > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if t <= MACHEP {
            break;
        }
    }
    ans * ax
}

const MAXLOG: f64 = 7.097_827_128_933_84e2;

fn lgam(x: f64) -> f64 {
    libm::lgamma(x)
}

#[cfg(test)]
mod tests {
    use super::{chi2_sf, igamc};

    fn close(got: f64, want: f64, rel: f64) {
        let d = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
        assert!(d <= rel, "got {got:e} want {want:e} rel {d:e} > {rel:e}");
    }

    #[test]
    fn igamc_matches_scipy_gammaincc() {
        let cases = [
            (0.5, 0.1, 0.654_720_846_018_576_8),
            (1.0, 2.0, 0.135_335_283_236_612_7),
            (2.0, 3.5, 0.135_888_225_400_433_27),
            (2.0, 7.0, 0.007_295_055_724_436_127),
            (0.5, 0.25, 0.479_500_122_186_953_37),
            (1.5, 3.0, 0.111_610_225_094_712_51),
            (3.0, 1.0, 0.919_698_602_928_605_8),
            (10.0, 5.0, 0.968_171_942_693_795_1),
            (10.0, 25.0, 0.000_221_476_638_248_783_3),
            (0.5, 10.0, 7.744_216_431_044_084e-6),
            (5.0, 0.5, 0.999_827_884_370_044_1),
            (50.0, 60.0, 0.084_406_681_093_691_88),
            (0.5, 0.0001, 0.988_716_584_444_150_3),
            (100.0, 80.0, 0.982_891_686_964_866_8),
            (2.5, 12.5, 0.000_139_333_791_185_626_3),
            (20.0, 40.0, 0.000_176_302_897_738_567_7),
            (0.5, 700.0, 2.101_014_516_264_400_3e-306),
            (1.0, 700.0, 9.859_676_543_759_39e-305),
        ];
        for (a, x, want) in cases {
            close(igamc(a, x), want, 1e-12);
        }
    }

    #[test]
    fn chi2_sf_df4_matches_scipy() {
        // chi2.sf(W, 4) reference values from scipy.stats.chi2.sf
        close(
            chi2_sf(4.0, 5.028_457_653_794_002),
            0.284_390_006_484_636_25,
            1e-12,
        );
        close(
            chi2_sf(4.0, 28.413_304_802_889_236),
            1.028_401_909_960_339_8e-5,
            1e-11,
        );
        close(
            chi2_sf(6.0, 28.736_704_683_465_15),
            6.822_811_673_650_14e-5,
            1e-11,
        );
    }

    #[test]
    fn nan_short_circuits() {
        assert!(chi2_sf(4.0, f64::NAN).is_nan());
        assert!(igamc(f64::NAN, 1.0).is_nan());
        assert!(igamc(2.0, f64::NAN).is_nan());
    }
}
