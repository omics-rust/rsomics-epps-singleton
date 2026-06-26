//! Epps-Singleton two-sample test — `scipy.stats.epps_singleton_2samp` equivalent.
//!
//! Two single-column inputs give the samples; the test reports the `W`
//! statistic and the asymptotic chi-squared p-value. A characteristic-function
//! test, more powerful than Kolmogorov-Smirnov for discrete or mixed data.

mod chi2;
mod es;
mod jacobi;
mod pairwise;
mod parse;

pub use chi2::chi2_sf;
pub use es::{EsError, EsResult, epps_singleton};
pub use parse::{parse_buffer, read_values};

/// Default frequencies `(0.4, 0.8)` from Epps & Singleton (1986).
pub const DEFAULT_T: [f64; 2] = [0.4, 0.8];
