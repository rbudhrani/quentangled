use dhat::{Alloc, Profiler};
use quentangled::{Circuit, Simulator};

#[global_allocator]
static ALLOC: Alloc = Alloc;

fn main() {
    let _profiler = Profiler::builder().build();

    println!("Building 20-qubit GHZ circuit...");
    let mut circuit = Circuit::new(20);
    let simulator = Simulator::default();
    circuit.h(0).unwrap();
    for i in 0..19 {
        circuit.cnot(i, i + 1).unwrap();
    }

    println!("Executing state vector (Expect ~33 MB allocation)...");
    let _state = simulator.execute(&circuit);

    println!("Done!");
}
