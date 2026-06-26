use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_epps_singleton::{epps_singleton, parse_buffer};
use std::hint::black_box;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn sample(n: usize, seed: u64, shift: f64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s) + shift).collect()
}

fn bench(c: &mut Criterion) {
    let x = sample(200_000, 1, 0.0);
    let y = sample(250_000, 2, 0.3);

    c.bench_function("epps_compute_450k", |b| {
        b.iter(|| epps_singleton(black_box(&x), black_box(&y), black_box(&[0.4, 0.8])).unwrap());
    });

    let mut buf = Vec::new();
    for v in &x {
        buf.extend_from_slice(format!("{v}\n").as_bytes());
    }
    c.bench_function("parse_200k", |b| {
        b.iter(|| parse_buffer(black_box(&buf)).unwrap());
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
