use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::time::Duration;

use quentangled::{Circuit, EngineConfig, Simulator, SimulatorConfig};
use std::hint::black_box;

fn threads_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("threads_bench");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for qubits in 10..=20 {
        let mut circuit = Circuit::new(qubits);
        circuit.h(0).unwrap();
        for i in 0..qubits - 1 {
            circuit.cnot(i, i + 1).unwrap();
        }

        let sequential_simulator = Simulator::new(SimulatorConfig {
            engine: EngineConfig {
                parallel_threshold: usize::MAX,
                ..EngineConfig::default()
            },
            ..SimulatorConfig::default()
        })
        .unwrap();
        let parallel_simulator = Simulator::new(SimulatorConfig {
            engine: EngineConfig {
                parallel_threshold: 0,
                ..EngineConfig::default()
            },
            ..SimulatorConfig::default()
        })
        .unwrap();

        // 1. Force Sequential Mode
        group.bench_with_input(BenchmarkId::new("sequential", qubits), &qubits, |b, _| {
            b.iter(|| sequential_simulator.execute(black_box(&circuit)))
        });

        // 2. Force Parallel Mode
        group.bench_with_input(BenchmarkId::new("parallel", qubits), &qubits, |b, _| {
            b.iter(|| parallel_simulator.execute(black_box(&circuit)))
        });
    }

    group.finish();
}

criterion_group!(benches, threads_bench);
criterion_main!(benches);
