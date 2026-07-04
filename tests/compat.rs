//! Value-exactness vs `scipy.stats.epps_singleton_2samp` 1.17.1.
//!
//! Expected statistic/p-value were produced by scipy once during development and
//! committed to `tests/golden/expected.tsv`; this test runs WITHOUT scipy.

use std::path::Path;

use rsomics_epps_singleton::epps_singleton;

fn read(path: &Path) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect()
}

fn relerr(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn matches_scipy_golden() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let expected = std::fs::read_to_string(dir.join("expected.tsv")).unwrap();

    let mut checked = 0;
    for line in expected.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let label = cols[0];
        let want_stat: f64 = cols[1].parse().unwrap();
        let want_p: f64 = cols[2].parse().unwrap();
        let t: Vec<f64> = cols[3].split(',').map(|v| v.parse().unwrap()).collect();

        let x = read(&dir.join(format!("{label}_x.tsv")));
        let y = read(&dir.join(format!("{label}_y.tsv")));
        let res = epps_singleton(&x, &y, &t).unwrap();

        if want_stat.is_nan() || want_p.is_nan() {
            assert!(
                res.statistic.is_nan() && res.pvalue.is_nan(),
                "case {label}: non-finite input must give nan/nan, got {}/{}",
                res.statistic,
                res.pvalue
            );
            checked += 1;
            continue;
        }

        let rs = relerr(res.statistic, want_stat);
        let rp = relerr(res.pvalue, want_p);
        assert!(
            rs <= 1e-10,
            "case {label}: statistic {} vs scipy {want_stat} (relerr {rs:e})",
            res.statistic
        );
        assert!(
            rp <= 1e-10,
            "case {label}: pvalue {} vs scipy {want_p} (relerr {rp:e})",
            res.pvalue
        );
        checked += 1;
    }
    assert!(
        checked >= 6,
        "expected at least 6 golden cases, got {checked}"
    );
}

#[test]
fn cli_emits_statistic_and_p() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let bin = env!("CARGO_BIN_EXE_rsomics-epps-singleton");
    let out = std::process::Command::new(bin)
        .arg(dir.join("b_x.tsv"))
        .arg(dir.join("b_y.tsv"))
        .arg("-t1")
        .output()
        .unwrap();
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    let fields: Vec<&str> = text.trim().split('\t').collect();
    assert_eq!(
        fields.len(),
        2,
        "expected two TAB-separated fields: {text:?}"
    );
    let stat: f64 = fields[0].parse().unwrap();
    assert!(relerr(stat, 15.177899584007768) <= 1e-10);
}

/// Non-finite input must exit cleanly with `nan<TAB>nan`, never panic (the
/// old code panicked on NaN and hit "SVD did not converge" on inf).
#[test]
fn cli_non_finite_gives_nan_without_panic() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let bin = env!("CARGO_BIN_EXE_rsomics-epps-singleton");
    for label in ["g", "h"] {
        let t = if label == "g" {
            "0.4,0.8"
        } else {
            "0.3,0.6,0.9"
        };
        let out = std::process::Command::new(bin)
            .arg(dir.join(format!("{label}_x.tsv")))
            .arg(dir.join(format!("{label}_y.tsv")))
            .arg(format!("--t={t}"))
            .arg("-t1")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "case {label}: exit {:?}, stderr {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8(out.stdout).unwrap();
        let fields: Vec<&str> = text.trim().split('\t').collect();
        assert_eq!(fields.len(), 2, "case {label}: got {text:?}");
        assert!(
            fields[0].parse::<f64>().unwrap().is_nan()
                && fields[1].parse::<f64>().unwrap().is_nan(),
            "case {label}: expected nan/nan, got {text:?}"
        );
    }
}

#[test]
fn rejects_too_few_observations() {
    let r = epps_singleton(
        &[1.0, 2.0, 3.0, 4.0],
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[0.4, 0.8],
    );
    assert!(r.is_err());
}

#[test]
fn rejects_non_positive_t() {
    let x: Vec<f64> = (0..10).map(f64::from).collect();
    assert!(epps_singleton(&x, &x, &[0.4, -0.8]).is_err());
}

/// All-identical inputs give a zero pooled IQR, so scipy's `pinv` raises
/// `LinAlgError: SVD did not converge`; we surface the same message.
#[test]
fn degenerate_iqr_matches_scipy_error() {
    let x = vec![3.0; 10];
    let err = epps_singleton(&x, &x, &[0.4, 0.8]).unwrap_err();
    assert_eq!(err.to_string(), "SVD did not converge");
}
