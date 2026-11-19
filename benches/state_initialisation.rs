use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex64;
use std::hint::black_box;
use std::time::Duration;

mod core {
    pub mod config {
        pub use quentangled::EngineConfig;
    }
}

// Include the backend source so this benchmark can access its crate-private initialiser.
#[allow(dead_code)]
#[path = "../src/backend/state_vector.rs"]
mod state_vector;

use state_vector::StateVector;

fn bench_state_initialisation(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_vector_initialization");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    let plus = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    for num_qubits in [10, 15, 20] {
        let initial_state = vec![(plus, plus); num_qubits];
        group.throughput(Throughput::Elements(1 << num_qubits));

        group.bench_function(BenchmarkId::new("zero_state", num_qubits), |b| {
            b.iter(|| black_box(StateVector::new(num_qubits)))
        });
        group.bench_function(BenchmarkId::new("product_state", num_qubits), |b| {
            b.iter(|| {
                black_box(StateVector::with_initial_state(
                    num_qubits,
                    black_box(&initial_state),
                ))
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_state_initialisation);
criterion_main!(benches);
