use criterion::{Criterion, criterion_group, criterion_main};
use quentangled::{Circuit, Simulator};
use std::hint::black_box;
use std::time::Duration;

fn bench_speed(c: &mut Criterion) {
    let simulator = Simulator::default();
    let mut group = c.benchmark_group("quentangled_speed");

    // Lower sample size and measurement time to prevent 20-qubit
    // simulations from stalling your CI/CD pipeline.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    // ---------------------------------------------------------
    // 1. The Baseline (10 Qubits, GHZ State)
    // Fits entirely in L1 CPU Cache. Tests Rust loop overhead.
    // ---------------------------------------------------------
    let mut circ_10 = Circuit::new(10);
    circ_10.h(0).unwrap();
    for i in 0..9 {
        circ_10.cnot(i, i + 1).unwrap();
    }
    group.bench_function("ghz_10_qubits_baseline", |b| {
        b.iter(|| simulator.execute(black_box(&circ_10)))
    });

    // ---------------------------------------------------------
    // 2. The Bandwidth Test (20 Qubits, GHZ State)
    // Exceeds CPU cache (~33 MB). Tests RAM memory throughput.
    // ---------------------------------------------------------
    let mut circ_20 = Circuit::new(20);
    circ_20.h(0).unwrap();
    for i in 0..19 {
        circ_20.cnot(i, i + 1).unwrap();
    }
    group.bench_function("ghz_20_qubits_bandwidth", |b| {
        b.iter(|| simulator.execute(black_box(&circ_20)))
    });

    // ---------------------------------------------------------
    // 3. Adjacent CNOT (20 Qubits)
    // Best-case cache locality for a 2-qubit bitwise operation.
    // ---------------------------------------------------------
    let mut circ_adj = Circuit::new(20);
    circ_adj.h(0).unwrap();
    circ_adj.cnot(0, 1).unwrap();
    group.bench_function("cnot_20_qubits_adjacent", |b| {
        b.iter(|| simulator.execute(black_box(&circ_adj)))
    });

    // ---------------------------------------------------------
    // 4. Distant CNOT (20 Qubits)
    // Worst-case cache locality. Forces massive memory strides.
    // ---------------------------------------------------------
    let mut circ_dist = Circuit::new(20);
    circ_dist.h(0).unwrap();
    circ_dist.cnot(0, 19).unwrap();
    group.bench_function("cnot_20_qubits_distant", |b| {
        b.iter(|| simulator.execute(black_box(&circ_dist)))
    });

    // Signal to Criterion that the group configuration is complete
    group.finish();
}

criterion_group!(benches, bench_speed);
criterion_main!(benches);
