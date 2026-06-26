use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use rsomics_common::{CommonFlags, RsomicsError, ToolMeta, run};

use rsomics_epps_singleton::{DEFAULT_T, epps_singleton, read_values};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

/// Epps-Singleton two-sample test (`scipy.stats.epps_singleton_2samp`).
///
/// Each input is a single-column file (one value per line); `-` reads stdin (at
/// most one input may be stdin). Output is a single line `statistic<TAB>p`.
#[derive(Parser, Debug)]
#[command(name = "rsomics-epps-singleton", version, about, long_about = None)]
pub struct Cli {
    /// First sample (`x`): one value per line.
    #[arg(value_name = "X")]
    pub x: PathBuf,

    /// Second sample (`y`): one value per line.
    #[arg(value_name = "Y")]
    pub y: PathBuf,

    /// Comma-separated positive frequencies for the characteristic function
    /// (long-only `--t`; `-t`/`--threads` sets worker threads, not frequencies).
    #[arg(long, default_value = "0.4,0.8", value_name = "T1,T2,...")]
    pub t: String,

    #[command(flatten)]
    pub common: CommonFlags,
}

fn parse_t(s: &str) -> rsomics_common::Result<Vec<f64>> {
    if s.trim() == "0.4,0.8" {
        return Ok(DEFAULT_T.to_vec());
    }
    let ts: Vec<f64> = s
        .split(',')
        .map(|tok| {
            tok.trim().parse::<f64>().map_err(|_| {
                RsomicsError::InvalidInput(format!("--t value '{tok}' is not a number"))
            })
        })
        .collect::<rsomics_common::Result<_>>()?;
    if ts.is_empty() {
        return Err(RsomicsError::InvalidInput("--t is empty".into()));
    }
    Ok(ts)
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            let t = parse_t(&self.t)?;
            let xs = read_values(&self.x)?;
            let ys = read_values(&self.y)?;
            let result = epps_singleton(&xs, &ys, &t)
                .map_err(|e| RsomicsError::InvalidInput(e.to_string()))?;
            if !common.json {
                println!("{}\t{}", result.statistic, result.pvalue);
            }
            Ok(result)
        })
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
