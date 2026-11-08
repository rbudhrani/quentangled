# Quentangled

[![CI](https://github.com/rbudhrani/quentangled/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rbudhrani/quentangled/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/quentangled.svg)](https://crates.io/crates/quentangled)
[![Documentation](https://docs.rs/quentangled/badge.svg)](https://docs.rs/quentangled)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**A Quantum Circuit Simulator.**

Quentangled is a quantum circuit simulator capable of constructing quantum circuits, and executing them using a high-performance mathematical backend.
Instead of using vector matrix multiplications, `quentangled` uses a system of 2 vectors. The vectors contain the probability amplitudes of the quantum state.
A 6 qubit quantum state is represented with a $2^6 = 64$ length vector, for example.
The gates execute their logic and replace the probability amplitude in the affected locations using a second buffer vector.
This results in a simulator with a smaller memory footprint than a vector/matrix simulator.

## 📊 Benchmarks

### Memory profile
Using `dhat` and the example script `examples/twenty_qubits.rs`, the memory footprint of creating a quantum circuit with 20 qubits is 32 MiB.
Each probability amplitude is represented using 16 bytes (8 bytes for the real part and 8 for the imaginery).
20 qubits result in 1,048,576 states.
Therefore 20 qubits take up $1,048,576 * 16$ bytes = 16,777,216 bytes.
Quentangled uses a buffer vector as well, which takes up another 16,777,216 bytes.
In total, 20 qubits take up 33,554,432 bytes = 32 MiB
This can be generalised for `n` qubits:

$m = 16 * 2 * 2^n = 2^{(n + 5)}$

Of course the simulator itself has overhead, but the memory footprint is dicated by the number of qubits.

### Speed
The benchmarks in `benches` can be run using
```zsh
cargo bench
```
Each bench serves a different purpose:
- `threads_bench`: Creates a circuit to make the GHZ state. The simulator is then swept from 10 to 20 qubits. For each of those runs, it does a sequential run and parallel run. The sweep compares the parallel and sequential run to figure out when it is more beneficial to switch to parallelism.
- `chunk_bench`: Runs the simulator using parallelism enabled for 20 qubits. It sweeps for the chunk size that is passed to each thread.
- `speed_bench`: Runs the simulator with various circuits to benchmark the spped profile of the simulator.

## 🚀 Installation

Quentangled is a Rust library and does not provide a standalone executable. New GitHub releases are published to crates.io by the release workflow. For versions not yet available there, add the Git dependency to your `Cargo.toml`:

```toml
[dependencies]
quentangled = { git = "https://github.com/rbudhrani/quentangled"}
```

GitHub provides source archives for each [release](../../releases). To build the library locally, clone the repository and run:

```sh
git clone https://github.com/rbudhrani/quentangled.git
cd quentangled
cargo build --release -p quentangled
```

## 💻 Usage
Quentangled allows the user to setup certain configuration parameters. These fall under 2 categories:
1. Engine related parameters
2. Gate related parameters

These parameters can be provided in a TOML file called `quentangled.toml`. An example file is shown below:
```toml
[engine]
parallel_threshold = 16384

[gates.x]
time_ns = 25.0
rot_angle_err_rad = 0.2

[gates.cnot]
time_ns = 150.0
```

For each configured gate, choose one error model: the existing `rot_angle_err_rad`
setting, or an explicit unitary `error_matrix`. Do not set both. For example, an
explicit single-qubit error matrix can be configured as:

```toml
[gates.x]
error_matrix = [
  [1.0, 0.0], [0.0, 0.0],
  [0.0, 0.0], [-1.0, 0.0],
]
```

Each pair is `[real, imaginary]`; elements are flattened in row-major order. A
single-qubit matrix has 4 complex elements. Two-qubit gates (`cnot`, `cz`, and
`swap`) accept a full 4x4 error matrix containing 16 complex elements. For a
two-qubit gate, matrix basis order is `|q_first q_second>`, following the order
of the gate's qubit operands. For example, a SWAP error after CNOT can be
specified with:

```toml
[gates.cnot]
error_matrix = [
  [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
  [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0],
  [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0],
  [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0],
]
```

The configured error matrix is applied after the ideal gate, so the effective
operation is `error_matrix * gate_matrix`. Configuration loading rejects
matrices of the wrong size, non-finite or non-unitary matrices, and settings
that specify both error models. These matrices model coherent unitary errors;
stochastic noise channels are not represented. A configured matrix under
`[gates.unitary]` is applied after each custom single-qubit unitary operation.
The optional error matrix passed to `Circuit::unitary` remains available; if
both are supplied, the simulator-configured matrix is applied last. An angle
error is undefined for an arbitrary custom unitary and is rejected.

Load the configuration into a simulator instance. Settings belong to that instance, so separate simulators can use different configurations without affecting one another:

```rust
use quentangled::{Circuit, Simulator, SimulatorConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let config = SimulatorConfig::from_file("quentangled.toml")?;
  let simulator = Simulator::new(config).expect("valid simulator config");

  let mut circuit = Circuit::new(2);
  circuit.h(0)?.cnot(0, 1)?;
  let state = simulator.execute(&circuit);

  println!("{:?}", state.probabilities());
  Ok(())
}
```

Execution returns a `SimulationResult` with amplitude and probability accessors; the state-vector backend remains an implementation detail.

Missing settings use the defaults defined by `quentangled`.
Gate names in TOML are case-insensitive; unknown names are rejected during configuration loading.

### Engine parameters
The 2 parameters that can be set here are:
1. `parallel_threshold`: The minimum number of state-vector elements before single-qubit gates use parallel execution.
2. `chunk_size`: The number of state-vector elements grouped into each parallel task.

You may run the benchmarks in `benches` to tune them for your machine.

### Gate related parameters
The parameters that can be set here are:
1. `rot_angle_err_rad`: The rotation angle error of a quantum gate in radians.
2. `error_matrix`: An optional row-major unitary error matrix, with each complex element given as `[real, imaginary]`. It replaces `rot_angle_err_rad`; the two error settings are mutually exclusive.
3. `time_ns`: The time taken for a gate to execute in nanoseconds.

Gate error settings can be set for each supported gate. Single-qubit gates use
2x2 error matrices; two-qubit gates use full 4x4 matrices.

## 🛠️ Contributing
To contribute to this project, please open an issue and then create a pull request from it. To trigger the automated release job of the pipeline, the commits in the `main` branch must follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). `release-please` parses commits using this specification and then decides whether the MAJOR, MINOR or PATCH version should be updated and updates the `CHANGELOG` using the commits as descriptions, so the commit message is very important. Before merging the PR ensure that the squashed commit message follows the conventional commit convention. `release-please` opens up a release pull request that is updated with the latest commits in `main`. Once this pull request is merged, a new tag is created along with a corresponding GitHub release.

The release workflow publishes each newly created release to crates.io. Configure the repository secret `CARGO_REGISTRY_TOKEN` with a crates.io API token that has permission to publish this crate.

The following commits will result in a version bump:
1. **fix:** a commit of the _type_ `fix` patches a bug in your codebase (this correlates with [`PATCH`](http://semver.org/#summary) in Semantic Versioning).
2. **feat:** a commit of the _type_ `feat` introduces a new feature to the codebase (this correlates with [`MINOR`](http://semver.org/#summary) in Semantic Versioning).
3. **BREAKING CHANGE:** a commit that has a footer `BREAKING CHANGE:`, or appends a `!` after the type/scope, introduces a breaking API change (correlating with [`MAJOR`](http://semver.org/#summary) in Semantic Versioning).
A BREAKING CHANGE can be part of commits of any _type_.
4. _types_ other than `fix:` and `feat:` are allowed, for example [@commitlint/config-conventional](https://github.com/conventional-changelog/commitlint/tree/master/%40commitlint/config-conventional) (based on the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines)) recommends `build:`, `chore:`,
  `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others..
5. _footers_ other than `BREAKING CHANGE: <description>` may be provided and follow a convention similar to
  [git trailer format](https://git-scm.com/docs/git-interpret-trailers).

Additional types are not mandated by the Conventional Commits specification, and have no implicit effect in Semantic Versioning (unless they include a BREAKING CHANGE).