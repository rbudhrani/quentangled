//! Executes quantum circuits by mapping abstract gates to physical matrices.
//!
//! The executor acts as the bridge between the logical representation of a quantum
//! circuit (defined in the core crate) and the mathematical simulation engine.
//! It translates instructions into linear algebra operations.

use super::state_vector::StateVector;
use crate::backend::gate_matrices::GateMatrixRegistry;
use crate::core::circuit::Circuit;
use crate::core::config::{ConfigError, EngineConfig, SimulatorConfig};
use crate::core::gate::Gate;
use crate::core::operation::Operation;
use num_complex::Complex64;

/// Public output from a circuit simulation, without exposing backend storage.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    num_qubits: usize,
    pub(crate) amplitudes: Vec<Complex64>,
}

impl SimulationResult {
    /// Converts the internal state vector into the public simulation result.
    ///
    /// # Arguments
    /// * `state` - The completed backend state produced by executing a circuit.
    ///
    /// # Returns
    /// A result containing the qubit count and amplitudes, without the backend buffer.
    fn from_state_vector(state: StateVector) -> Self {
        let (num_qubits, amplitudes) = state.into_parts();
        Self {
            num_qubits,
            amplitudes,
        }
    }

    /// Returns the number of qubits represented by this result.
    ///
    /// # Returns
    /// The number of qubits in the simulated circuit.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the complex amplitude for each computational-basis state.
    ///
    /// Amplitude at index `i` corresponds to basis state `|i>` under the simulator's
    /// little-endian qubit-index convention.
    ///
    /// # Returns
    /// A slice containing one complex amplitude for each computational-basis state.
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }

    /// Returns the probability of measuring each computational-basis state.
    ///
    /// Each value is the squared magnitude of the corresponding amplitude.
    ///
    /// # Returns
    /// A vector whose element at each index is the probability of measuring that basis state.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes
            .iter()
            .map(|amplitude| amplitude.norm_sqr())
            .collect()
    }

    /// Returns the probability of measuring one computational-basis state.
    ///
    /// The probability is the squared magnitude of that state's amplitude. Basis-state
    /// indices use the simulator's little-endian qubit-index convention.
    ///
    /// # Arguments
    /// * `basis_state` - The integer index of the computational-basis state.
    ///
    /// # Returns
    /// `Some(probability)` if `basis_state` is in the range `0..2^num_qubits`, or
    /// `None` if the index is outside the result's state space.
    pub fn probability(&self, basis_state: usize) -> Option<f64> {
        self.amplitudes
            .get(basis_state)
            .map(|amplitude| amplitude.norm_sqr())
    }
}

/// A quantum circuit simulator with configuration isolated to this instance.
#[derive(Debug, Clone)]
pub struct Simulator {
    config: SimulatorConfig,
    gates: GateMatrixRegistry,
}

impl Simulator {
    /// Creates a simulator using the supplied configuration.
    ///
    /// # Arguments
    /// * `config` - Per-instance engine and gate settings.
    ///
    /// # Returns
    /// A simulator initialized with the validated configuration.
    ///
    /// # Errors
    /// Returns [`ConfigError`] if any engine or gate setting is invalid.
    pub fn new(config: SimulatorConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let gates = GateMatrixRegistry::from_config(&config);
        Ok(Self { config, gates })
    }

    /// Returns this simulator's immutable configuration.
    ///
    /// # Returns
    /// A shared reference to this simulator's configuration.
    pub fn config(&self) -> &SimulatorConfig {
        &self.config
    }

    /// Executes a circuit from the all-zero initial state.
    ///
    /// # Arguments
    /// * `circuit` - The validated sequence of gates to execute.
    ///
    /// # Returns
    /// The final state amplitudes and derived probability accessors.
    pub fn execute(&self, circuit: &Circuit) -> SimulationResult {
        execute(circuit, &self.gates, self.config.engine)
    }

    /// Estimates total circuit duration by summing configured gate durations.
    ///
    /// This is a gate-time estimate; it does not include scheduling or simulator runtime.
    ///
    /// # Arguments
    /// * `circuit` - The circuit whose configured gate durations are summed.
    ///
    /// # Returns
    /// The estimated total duration in nanoseconds.
    pub fn estimated_duration_ns(&self, circuit: &Circuit) -> f64 {
        circuit.estimated_duration_ns(&self.config)
    }
}

impl Default for Simulator {
    fn default() -> Self {
        Self::new(SimulatorConfig::default()).expect("default simulator config is valid")
    }
}

/// Applies a validated circuit using explicit gate matrices and engine settings.
///
/// This function iterates chronologically through every operation in the circuit,
/// matches each abstract [`Gate`] to its backend operation, and applies it to the target qubit(s).
///
/// # Arguments
///
/// * `circuit` - The circuit containing the sequence of validated operations to run.
/// * `gates` - The gate matrices and phase settings used during execution.
/// * `engine` - The parallel threshold and chunk size used for gate execution.
///
/// # Returns
///
/// A [`SimulationResult`] with the final amplitudes and probabilities.
///
/// # Panics
/// Panics if `engine.chunk_size` is zero. Public simulator construction rejects this setting.
pub(crate) fn execute(
    circuit: &Circuit,
    gates: &GateMatrixRegistry,
    engine: EngineConfig,
) -> SimulationResult {
    assert!(engine.chunk_size > 0, "chunk_size must be greater than 0");
    let mut state = StateVector::new(circuit.num_qubits());

    for op in circuit.operations() {
        match op {
            Operation::ApplyGate { gate, qubits, .. } => {
                match gate {
                    // --- Two-Qubit Gates ---
                    Gate::CNOT => {
                        state.apply_controlled_gate(qubits[0], qubits[1], gates.cnot, engine);
                        if let Some(error) = gates.cnot_error_matrix {
                            state.apply_two_qubit_gate(qubits[0], qubits[1], error, engine);
                        }
                    }
                    Gate::CZ => {
                        state.apply_controlled_gate(qubits[0], qubits[1], gates.cz, engine);
                        if let Some(error) = gates.cz_error_matrix {
                            state.apply_two_qubit_gate(qubits[0], qubits[1], error, engine);
                        }
                    }
                    Gate::SWAP => {
                        state.apply_swap(qubits[0], qubits[1], gates.swap, engine);
                        if let Some(error) = gates.swap_error_matrix {
                            state.apply_two_qubit_gate(qubits[0], qubits[1], error, engine);
                        }
                    }

                    // --- Single-Qubit Gates ---
                    Gate::PauliX => {
                        state.apply_single_qubit_gate(qubits[0], gates.x, engine);
                        if let Some(error) = gates.x_error_matrix {
                            state.apply_single_qubit_gate(qubits[0], error, engine);
                        }
                    }
                    Gate::PauliY => {
                        state.apply_single_qubit_gate(qubits[0], gates.y, engine);
                        if let Some(error) = gates.y_error_matrix {
                            state.apply_single_qubit_gate(qubits[0], error, engine);
                        }
                    }
                    Gate::PauliZ => {
                        state.apply_single_qubit_gate(qubits[0], gates.z, engine);
                        if let Some(error) = gates.z_error_matrix {
                            state.apply_single_qubit_gate(qubits[0], error, engine);
                        }
                    }
                    Gate::Hadamard => {
                        state.apply_single_qubit_gate(qubits[0], gates.h, engine);
                        if let Some(error) = gates.h_error_matrix {
                            state.apply_single_qubit_gate(qubits[0], error, engine);
                        }
                    }
                    Gate::Phase(theta) => {
                        let matrix = gates.phase(*theta);
                        state.apply_single_qubit_gate(qubits[0], matrix, engine);
                        if let Some(error) = gates.phase_error_matrix {
                            state.apply_single_qubit_gate(qubits[0], error, engine);
                        }
                    }
                    Gate::Unitary2x2(matrix) => {
                        state.apply_single_qubit_gate(qubits[0], *matrix, engine);
                        if let Some(error) = gates.unitary_error_matrix {
                            state.apply_single_qubit_gate(qubits[0], error, engine);
                        }
                    }
                }
            }
        }
    }

    SimulationResult::from_state_vector(state)
}

#[cfg(test)]
mod tests {
    //! Unit tests for the executor module.

    use crate::core::config::{GatePropsConfig, SimulatorConfig};

    use super::*;
    use num_complex::Complex64;
    use std::f64::consts::FRAC_1_SQRT_2;
    use std::f64::consts::PI;
    use std::sync::LazyLock;

    static ERR_GATE_MATRIX_REGISTRY: LazyLock<GateMatrixRegistry> = LazyLock::new(|| {
        let mut config = SimulatorConfig::default();
        for name in ["x", "y", "z", "h", "phase"] {
            config.gates.insert(
                name.to_string(),
                GatePropsConfig {
                    time_ns: Some(0.0),
                    rot_angle_err_rad: Some(PI),
                    error_matrix: None,
                },
            );
        }
        GateMatrixRegistry::from_config(&config)
    });

    /// Helper to compare complex amplitudes within a small margin of error.
    fn approx_eq(a: Complex64, b: Complex64) -> bool {
        (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10
    }

    struct SingleQubitErrorCase {
        name: &'static str,
        expected: [Complex64; 2],
        apply_gate: fn(&mut Circuit),
    }

    struct TwoQubitErrorCase {
        name: &'static str,
        expected_index: usize,
        apply_gate: fn(&mut Circuit),
    }

    #[test]
    fn test_execute_empty_circuit_state_unchanged() {
        let circuit = Circuit::new(2);
        let gate_matrix_registry: GateMatrixRegistry = Default::default();
        let state = execute(&circuit, &gate_matrix_registry, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(1.0, 0.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[2], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[3], Complex64::new(0.0, 0.0)));
    }

    #[test]
    #[should_panic(expected = "chunk_size must be greater than 0")]
    fn test_execute_rejects_zero_chunk_size() {
        let circuit = Circuit::new(1);
        let gates = GateMatrixRegistry::default();
        let engine = EngineConfig {
            parallel_threshold: 0,
            chunk_size: 0,
        };

        execute(&circuit, &gates, engine);
    }

    #[test]
    fn test_simulator_rejects_invalid_config() {
        let config = SimulatorConfig {
            engine: EngineConfig {
                chunk_size: 0,
                ..EngineConfig::default()
            },
            ..SimulatorConfig::default()
        };

        assert_eq!(
            Simulator::new(config).unwrap_err(),
            ConfigError::InvalidChunkSize
        );
    }

    #[test]
    fn test_execute_pauli_x_flips_qubit() {
        let mut circuit = Circuit::new(1);
        let gate_matrix_registry: GateMatrixRegistry = Default::default();
        circuit.x(0).unwrap();

        let state = execute(&circuit, &gate_matrix_registry, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(1.0, 0.0)));
    }

    #[test]
    fn test_execute_hadamard_circuit_results_in_superposition() {
        let mut circuit = Circuit::new(1);
        let gate_matrix_registry: GateMatrixRegistry = Default::default();
        circuit.h(0).unwrap();

        let state = execute(&circuit, &gate_matrix_registry, EngineConfig::default());

        assert!(approx_eq(
            state.amplitudes[0],
            Complex64::new(FRAC_1_SQRT_2, 0.0)
        ));
        assert!(approx_eq(
            state.amplitudes[1],
            Complex64::new(FRAC_1_SQRT_2, 0.0)
        ));
    }

    #[test]
    fn test_execute_pauli_y_flips_and_adds_phase_to_qubit() {
        let mut circuit = Circuit::new(1);
        let gate_matrix_registry: GateMatrixRegistry = Default::default();
        circuit.y(0).unwrap();

        let state = execute(&circuit, &gate_matrix_registry, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(0.0, 1.0)));
    }

    #[test]
    fn test_execute_pauli_z_adds_phase_to_qubit() {
        let mut circuit = Circuit::new(1);
        let gate_matrix_registry: GateMatrixRegistry = Default::default();
        // start with state |1> so the Z gate can apply the phase
        circuit.x(0).unwrap().z(0).unwrap();

        let state = execute(&circuit, &gate_matrix_registry, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(-1.0, 0.0)));
    }

    #[test]
    fn test_execute_phase_adds_phase_to_qubit() {
        let mut circuit = Circuit::new(1);
        let gate_matrix_registry: GateMatrixRegistry = Default::default();
        // Put qubit into superposition: |0> -> (|0> + |1>) / sqrt(2)
        circuit.h(0).unwrap();

        // Apply a 90-degree (PI/2) Phase shift
        // This multiplies the |1> amplitude by e^(i * PI/2) = i
        circuit.phase(0, PI / 2.0).unwrap();

        let state = execute(&circuit, &gate_matrix_registry, EngineConfig::default());

        // Due to floating-point math, we use a small tolerance for assertions
        let tolerance = 1e-10;
        let expected_mag = std::f64::consts::FRAC_1_SQRT_2; // 1 / sqrt(2)

        // Amplitude of |0> should remain real: (1/sqrt(2)) + 0i
        assert!((state.amplitudes[0].re - expected_mag).abs() < tolerance);
        assert!((state.amplitudes[0].im - 0.0).abs() < tolerance);

        // Amplitude of |1> should now be purely imaginary: 0 + (1/sqrt(2))i
        assert!((state.amplitudes[1].re - 0.0).abs() < tolerance);
        assert!((state.amplitudes[1].im - expected_mag).abs() < tolerance);
    }

    #[test]
    fn test_unitary2x2_gate_as_x_gate_flips_qubit() {
        let mut circuit = Circuit::new(1);
        let gate_matrix_registry: GateMatrixRegistry = Default::default();

        // Manually define the Pauli-X (NOT) matrix:
        // [0, 1]
        // [1, 0]
        let custom_x_matrix = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        circuit.unitary(0, custom_x_matrix, None, 0.0).unwrap();

        let state = execute(&circuit, &gate_matrix_registry, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(1.0, 0.0)));
    }

    #[test]
    fn test_unitary_applies_optional_error_after_requested_gate() {
        let mut circuit = Circuit::new(1);
        let gates = GateMatrixRegistry::default();
        let x = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let z = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ];

        circuit.unitary(0, x, Some(z), 0.0).unwrap();

        let state = execute(&circuit, &gates, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(-1.0, 0.0)));
    }

    #[test]
    fn test_configured_single_qubit_error_matrix_is_applied_after_gate() {
        let config: SimulatorConfig = toml::from_str(
            "[gates.x]\nerror_matrix = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-1.0, 0.0]]",
        )
        .unwrap();
        let simulator = Simulator::new(config).unwrap();
        let mut circuit = Circuit::new(1);
        circuit.x(0).unwrap();

        let state = simulator.execute(&circuit);

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(-1.0, 0.0)));
    }

    #[test]
    fn test_configured_single_qubit_error_matrices_are_applied_after_gates() {
        let error_matrix = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]];
        let mut config = SimulatorConfig::default();
        for name in ["y", "z", "h", "phase"] {
            config.gates.insert(
                name.to_string(),
                GatePropsConfig {
                    time_ns: None,
                    rot_angle_err_rad: None,
                    error_matrix: Some(error_matrix.clone()),
                },
            );
        }
        let simulator = Simulator::new(config).unwrap();

        let cases = [
            SingleQubitErrorCase {
                name: "y",
                expected: [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
                apply_gate: |circuit| {
                    circuit.y(0).unwrap();
                },
            },
            SingleQubitErrorCase {
                name: "z",
                expected: [Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0)],
                apply_gate: |circuit| {
                    circuit.z(0).unwrap();
                },
            },
            SingleQubitErrorCase {
                name: "h",
                expected: [
                    Complex64::new(-FRAC_1_SQRT_2, 0.0),
                    Complex64::new(FRAC_1_SQRT_2, 0.0),
                ],
                apply_gate: |circuit| {
                    circuit.h(0).unwrap();
                },
            },
            SingleQubitErrorCase {
                name: "phase",
                expected: [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
                apply_gate: |circuit| {
                    circuit.phase(0, PI / 2.0).unwrap();
                },
            },
        ];
        for SingleQubitErrorCase {
            name,
            expected,
            apply_gate,
        } in cases
        {
            let mut circuit = Circuit::new(1);
            circuit.x(0).unwrap();
            apply_gate(&mut circuit);

            let state = simulator.execute(&circuit);
            for (actual, expected) in state.amplitudes.iter().zip(expected) {
                assert!(
                    approx_eq(*actual, expected),
                    "{name}: expected {expected:?}, got {actual:?}"
                );
            }
        }
    }

    #[test]
    fn test_configured_error_matrix_applies_after_custom_unitary() {
        let config: SimulatorConfig = toml::from_str(
            "[gates.unitary]\n\
             error_matrix = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-1.0, 0.0]]",
        )
        .unwrap();
        let simulator = Simulator::new(config).unwrap();
        let mut circuit = Circuit::new(1);
        let x = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        circuit.unitary(0, x, None, 0.0).unwrap();

        let state = simulator.execute(&circuit);

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(-1.0, 0.0)));
    }

    #[test]
    fn test_configured_two_qubit_error_matrix_is_applied_after_gate() {
        let config: SimulatorConfig = toml::from_str(
            "[gates.cnot]\n\
             error_matrix = [\
               [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],\
               [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0],\
               [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0],\
               [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]\
             ]",
        )
        .unwrap();
        let simulator = Simulator::new(config).unwrap();
        let mut circuit = Circuit::new(2);
        circuit.x(1).unwrap().cnot(0, 1).unwrap();

        let state = simulator.execute(&circuit);

        assert!(approx_eq(state.amplitudes[1], Complex64::new(1.0, 0.0)));
        assert!(state.amplitudes.iter().enumerate().all(
            |(index, amplitude)| index == 1 || approx_eq(*amplitude, Complex64::new(0.0, 0.0))
        ));
    }

    #[test]
    fn test_configured_cz_and_swap_error_matrices_are_applied_after_gates() {
        let error_matrix = vec![
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ];

        let cases = [
            TwoQubitErrorCase {
                name: "cz",
                expected_index: 1,
                apply_gate: |circuit| {
                    circuit.cz(0, 1).unwrap();
                },
            },
            TwoQubitErrorCase {
                name: "swap",
                expected_index: 2,
                apply_gate: |circuit| {
                    circuit.swap(0, 1).unwrap();
                },
            },
        ];
        for TwoQubitErrorCase {
            name,
            expected_index,
            apply_gate,
        } in cases
        {
            let mut config = SimulatorConfig::default();
            config.gates.insert(
                name.to_string(),
                GatePropsConfig {
                    time_ns: None,
                    rot_angle_err_rad: None,
                    error_matrix: Some(error_matrix.clone()),
                },
            );
            let simulator = Simulator::new(config).unwrap();
            let mut circuit = Circuit::new(2);
            circuit.x(1).unwrap();
            apply_gate(&mut circuit);

            let state = simulator.execute(&circuit);
            assert!(
                state
                    .amplitudes
                    .iter()
                    .enumerate()
                    .all(|(index, amplitude)| index == expected_index
                        && approx_eq(*amplitude, Complex64::new(1.0, 0.0))
                        || index != expected_index
                            && approx_eq(*amplitude, Complex64::new(0.0, 0.0)))
            );
        }
    }

    #[test]
    fn test_execute_cnot_flips_target_qubit() {
        let mut circuit = Circuit::new(2);
        let gate_matrix_registry: GateMatrixRegistry = Default::default();
        // Put the first qubit in the state |1> so the CNOT can flip the second.
        circuit.x(0).unwrap().cnot(0, 1).unwrap();

        let state = execute(&circuit, &gate_matrix_registry, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[3], Complex64::new(1.0, 0.0)));
    }

    #[test]
    fn test_execute_cnot_applies_configured_rotation_error() {
        let mut config = SimulatorConfig::default();
        config.gates.insert(
            "cnot".to_string(),
            GatePropsConfig {
                time_ns: None,
                rot_angle_err_rad: Some(PI / 2.0),
                error_matrix: None,
            },
        );
        let gates = GateMatrixRegistry::from_config(&config);

        let mut circuit = Circuit::new(2);
        circuit.x(0).unwrap().cnot(0, 1).unwrap();

        let state = execute(&circuit, &gates, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(
            state.amplitudes[1],
            Complex64::new(0.0, -FRAC_1_SQRT_2)
        ));
        assert!(approx_eq(state.amplitudes[2], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(
            state.amplitudes[3],
            Complex64::new(FRAC_1_SQRT_2, 0.0)
        ));
    }

    #[test]
    fn test_execute_configured_cnot_leaves_control_zero_branch_unchanged() {
        let mut config = SimulatorConfig::default();
        config.gates.insert(
            "cnot".to_string(),
            GatePropsConfig {
                time_ns: None,
                rot_angle_err_rad: Some(PI / 2.0),
                error_matrix: None,
            },
        );
        let gates = GateMatrixRegistry::from_config(&config);

        let mut circuit = Circuit::new(2);
        circuit.h(1).unwrap().cnot(0, 1).unwrap();

        let state = execute(&circuit, &gates, EngineConfig::default());

        assert!(approx_eq(
            state.amplitudes[0],
            Complex64::new(FRAC_1_SQRT_2, 0.0)
        ));
        assert!(approx_eq(
            state.amplitudes[2],
            Complex64::new(FRAC_1_SQRT_2, 0.0)
        ));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[3], Complex64::new(0.0, 0.0)));
    }

    #[test]
    fn test_execute_cz_adds_phase_to_target_qubit() {
        let mut circuit = Circuit::new(2);
        let gate_matrix_registry: GateMatrixRegistry = Default::default();

        circuit.x(0).unwrap().x(1).unwrap().cz(0, 1).unwrap();

        let state = execute(&circuit, &gate_matrix_registry, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[3], Complex64::new(-1.0, 0.0)));
    }

    #[test]
    fn test_execute_cz_applies_configured_rotation_error() {
        let mut config = SimulatorConfig::default();
        config.gates.insert(
            "cz".to_string(),
            GatePropsConfig {
                time_ns: None,
                rot_angle_err_rad: Some(PI / 2.0),
                error_matrix: None,
            },
        );
        let gates = GateMatrixRegistry::from_config(&config);

        let mut circuit = Circuit::new(2);
        circuit.x(0).unwrap().x(1).unwrap().cz(0, 1).unwrap();

        let state = execute(&circuit, &gates, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[2], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(
            state.amplitudes[3],
            Complex64::new(-FRAC_1_SQRT_2, -FRAC_1_SQRT_2)
        ));
    }

    #[test]
    fn test_execute_swap_swaps_qubits() {
        let mut circuit = Circuit::new(2);
        let gate_matrix_registry: GateMatrixRegistry = Default::default();

        circuit.x(0).unwrap().swap(0, 1).unwrap();

        let state = execute(&circuit, &gate_matrix_registry, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[2], Complex64::new(1.0, 0.0)));
    }

    #[test]
    fn test_execute_swap_applies_configured_rotation_error() {
        let mut config = SimulatorConfig::default();
        config.gates.insert(
            "swap".to_string(),
            GatePropsConfig {
                time_ns: None,
                rot_angle_err_rad: Some(PI / 2.0),
                error_matrix: None,
            },
        );
        let gates = GateMatrixRegistry::from_config(&config);

        let mut circuit = Circuit::new(2);
        circuit.x(0).unwrap().swap(0, 1).unwrap();

        let state = execute(&circuit, &gates, EngineConfig::default());

        assert!(approx_eq(
            state.amplitudes[1],
            Complex64::new(0.0, -FRAC_1_SQRT_2)
        ));
        assert!(approx_eq(
            state.amplitudes[2],
            Complex64::new(FRAC_1_SQRT_2, 0.0)
        ));
        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, 0.0)));
        assert!(approx_eq(state.amplitudes[3], Complex64::new(0.0, 0.0)));
    }

    #[test]
    fn test_execute_bell_state_entanglement_entangles_qubits() {
        let mut circuit = Circuit::new(2);
        let gate_matrix_registry: GateMatrixRegistry = Default::default();

        // 1. Hadamard on Qubit 0
        circuit.h(0).unwrap();

        // 2. CNOT: Control = Qubit 0, Target = Qubit 1
        circuit.cnot(0, 1).unwrap();

        let state = execute(&circuit, &gate_matrix_registry, EngineConfig::default());

        // A Bell State creates a 50/50 superposition of |00> and |11>.
        // The states |01> and |10> should have exactly 0 probability.
        assert!(approx_eq(
            state.amplitudes[0],
            Complex64::new(FRAC_1_SQRT_2, 0.0)
        )); // |00>
        assert!(approx_eq(state.amplitudes[1], Complex64::new(0.0, 0.0))); // |01>
        assert!(approx_eq(state.amplitudes[2], Complex64::new(0.0, 0.0))); // |10>
        assert!(approx_eq(
            state.amplitudes[3],
            Complex64::new(FRAC_1_SQRT_2, 0.0)
        )); // |11>
    }

    #[test]
    fn test_execute_err_pauli_x_does_not_flip_qubit() {
        let mut circuit = Circuit::new(1);

        circuit.x(0).unwrap();

        let state = execute(&circuit, &ERR_GATE_MATRIX_REGISTRY, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, -1.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(0.0, 0.0)));
    }

    #[test]
    fn test_execute_err_pauli_y_does_not_flip_qubit() {
        let mut circuit = Circuit::new(1);

        circuit.y(0).unwrap();

        let state = execute(&circuit, &ERR_GATE_MATRIX_REGISTRY, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, -1.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(0.0, 0.0)));
    }

    #[test]
    fn test_execute_err_pauli_z() {
        let mut circuit = Circuit::new(1);

        circuit.z(0).unwrap();

        let state = execute(&circuit, &ERR_GATE_MATRIX_REGISTRY, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, -1.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(0.0, 0.0)));
    }

    #[test]
    fn test_execute_err_hadamard_does_nothing() {
        let mut circuit = Circuit::new(1);

        circuit.h(0).unwrap();

        let state = execute(&circuit, &ERR_GATE_MATRIX_REGISTRY, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(0.0, -1.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(0.0, 0.0)));
    }

    #[test]
    fn test_execute_err_phase_gate_does_nothing() {
        let mut circuit = Circuit::new(1);

        circuit.phase(0, PI).unwrap();

        let state = execute(&circuit, &ERR_GATE_MATRIX_REGISTRY, EngineConfig::default());

        assert!(approx_eq(state.amplitudes[0], Complex64::new(1.0, 0.0)));
        assert!(approx_eq(state.amplitudes[1], Complex64::new(0.0, 0.0)));
    }

    #[test]
    fn test_simulator_settings_are_isolated_per_instance() {
        let mut noisy_config = SimulatorConfig::default();
        noisy_config.engine.parallel_threshold = 0;
        noisy_config.engine.chunk_size = 1;
        noisy_config.gates.insert(
            "x".to_string(),
            GatePropsConfig {
                time_ns: Some(12.0),
                rot_angle_err_rad: Some(PI),
                error_matrix: None,
            },
        );

        let noisy_simulator = Simulator::new(noisy_config).unwrap();
        let ideal_simulator = Simulator::default();
        let mut circuit = Circuit::new(1);
        circuit.x(0).unwrap();

        let noisy_state = noisy_simulator.execute(&circuit);
        let ideal_state = ideal_simulator.execute(&circuit);

        assert!(approx_eq(
            noisy_state.amplitudes[0],
            Complex64::new(0.0, -1.0)
        ));
        assert!(approx_eq(
            noisy_state.amplitudes[1],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq(
            ideal_state.amplitudes[0],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq(
            ideal_state.amplitudes[1],
            Complex64::new(1.0, 0.0)
        ));
        assert_eq!(noisy_simulator.estimated_duration_ns(&circuit), 12.0);
        assert_eq!(ideal_simulator.estimated_duration_ns(&circuit), 0.0);
        assert_eq!(noisy_simulator.config().engine.parallel_threshold, 0);
        assert_eq!(noisy_simulator.config().engine.chunk_size, 1);
        assert_eq!(ideal_simulator.config().engine.parallel_threshold, 16_384);
    }

    #[test]
    fn test_simulation_result_exposes_amplitudes_and_probabilities() {
        let mut circuit = Circuit::new(1);
        circuit.h(0).unwrap();
        let result = Simulator::default().execute(&circuit);

        assert_eq!(result.num_qubits(), 1);
        assert!(approx_eq(
            result.amplitudes()[0],
            Complex64::new(FRAC_1_SQRT_2, 0.0)
        ));
        assert!((result.probability(0).unwrap() - 0.5).abs() < 1e-10);
        assert!((result.probabilities()[1] - 0.5).abs() < 1e-10);
        assert_eq!(result.probability(2), None);
    }
}
