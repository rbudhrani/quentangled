use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::time::Duration;

use quentangled::{Circuit, EngineConfig, Simulator, SimulatorConfig};
use std::hint::black_box;

fn bench_chunk(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_bench");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    // Lock the circuit at 20 qubits (~33 MB of RAM)
    let qubits = 20;
    let mut circuit = Circuit::new(qubits);

    circuit.h(0).unwrap();
    for i in 0..qubits - 1 {
        circuit.cnot(i, i + 1).unwrap();
    }

    // Sweep through different cache sizes
    // 512, 1024, 2048, 4096, 8192, 16384, 32768
    let chunk_sizes = [512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768];

    for &size in chunk_sizes.iter() {
        let simulator = Simulator::new(SimulatorConfig {
            engine: EngineConfig {
                parallel_threshold: 0,
                chunk_size: size,
            },
            ..SimulatorConfig::default()
        })
        .unwrap();

        group.bench_with_input(BenchmarkId::new("chunk", size), &size, |b, _| {
            b.iter(|| simulator.execute(black_box(&circuit)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_chunk);
criterion_main!(benches);
