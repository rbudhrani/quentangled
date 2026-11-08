use dhat::{Alloc, Profiler};
use quentangled::Circuit;

#[global_allocator]
static ALLOC: Alloc = Alloc;

fn main() {
    let _profiler = Profiler::builder().build();

    println!("Building 25-qubit circuit struct...");
    let _circuit = Circuit::new(25);

    println!("Done! Check DHAT output for zero allocations.");
}
