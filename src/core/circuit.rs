//! Quantum circuit construction and operation management.
//!
//! This module provides the [`Circuit`] struct, which uses a fluent Builder Pattern
//! to allow users to construct complex quantum logic sequences ergonomically.

use std::fmt;

use num_complex::Complex64;

use crate::core::config::SimulatorConfig;
use crate::core::gate::Gate;
use crate::core::operation::Operation;

/// A quantum circuit representing a sequence of operations on a set of qubits.
///
/// The [`Circuit`] struct is the primary entry point for users building quantum algorithms.
/// It utilises a mutable reference builder pattern, allowing gate operations to be chained
/// fluently.
///
/// # Examples
///
/// ```
/// use quentangled::Circuit;
///
/// // Create a 2-qubit circuit and build a Bell state
/// let mut circuit = Circuit::new(2);
/// circuit
///     .h(0).unwrap()
///     .cnot(0, 1).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Circuit {
    /// The total number of qubits allocated for this circuit.
    num_qubits: usize,
    /// The chronological list of instructions to execute.
    operations: Vec<Operation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// An error encountered while constructing a validated quantum circuit.
pub enum CircuitError {
    /// A gate refers to a qubit outside the circuit's allocated range.
    QubitOutOfBounds {
        /// The invalid qubit index requested by the gate.
        qubit: usize,
        /// The number of qubits allocated to the circuit.
        num_qubits: usize,
    },
    /// A gate received a number of qubit operands different from its arity.
    InvalidArity {
        /// The number of operands required by the gate.
        expected: usize,
        /// The number of operands supplied to the gate.
        actual: usize,
    },
    /// A multi-qubit gate uses the same qubit for more than one operand.
    RepeatedQubit {
        /// The qubit index supplied for multiple operands.
        qubit: usize,
    },
    /// A custom gate matrix is non-finite or is not unitary.
    InvalidUnitary,
    /// A phase-gate angle is not finite.
    InvalidPhaseAngle,
    /// A custom gate duration is negative or not finite.
    InvalidGateDuration,
}

impl fmt::Display for CircuitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QubitOutOfBounds { qubit, num_qubits } => write!(
                formatter,
                "qubit index {qubit} is out of bounds for a {num_qubits}-qubit circuit"
            ),
            Self::InvalidArity { expected, actual } => {
                write!(formatter, "gate expects {expected} qubits, got {actual}")
            }
            Self::RepeatedQubit { qubit } => {
                write!(
                    formatter,
                    "qubit {qubit} cannot fill multiple gate operands"
                )
            }
            Self::InvalidUnitary => write!(formatter, "matrix is not a valid 2x2 unitary"),
            Self::InvalidPhaseAngle => write!(formatter, "phase angle must be finite"),
            Self::InvalidGateDuration => {
                write!(formatter, "gate duration must be finite and non-negative")
            }
        }
    }
}

impl std::error::Error for CircuitError {}

impl Circuit {
    /// Creates a new quantum circuit with each qubit initialised in the |0> state.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The total number of qubits allocated for this circuit.
    ///
    /// # Returns
    /// A circuit with no operations. Zero-qubit circuits can be constructed, but cannot accept gates.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            operations: Vec::new(),
        }
    }

    /// Returns the number of qubits allocated to this circuit.
    ///
    /// # Returns
    /// The number of qubits declared when this circuit was created.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the circuit's operations in insertion order.
    ///
    /// # Returns
    /// A read-only slice of the validated operations.
    pub(crate) fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Validates and appends a gate operation.
    ///
    /// # Arguments
    /// * `gate` - The gate to append.
    /// * `qubits` - The qubit operands, in the gate's defined operand order.
    /// * `duration_ns` - Optional per-operation duration override; required for custom unitaries.
    ///
    /// # Returns
    /// A mutable reference to this circuit for further construction.
    ///
    /// # Errors
    /// Returns [`CircuitError::InvalidArity`] for the wrong operand count,
    /// [`CircuitError::QubitOutOfBounds`] for an invalid index,
    /// [`CircuitError::RepeatedQubit`] when a two-qubit gate uses one qubit twice,
    /// [`CircuitError::InvalidUnitary`] for a non-unitary custom matrix, or
    /// [`CircuitError::InvalidPhaseAngle`] for a non-finite phase angle, or
    /// [`CircuitError::InvalidGateDuration`] for a negative or non-finite duration.
    pub(crate) fn apply_gate(
        &mut self,
        gate: Gate,
        qubits: &[usize],
        duration_ns: Option<f64>,
    ) -> Result<&mut Self, CircuitError> {
        let expected = gate.qubit_arity();
        if qubits.len() != expected {
            return Err(CircuitError::InvalidArity {
                expected,
                actual: qubits.len(),
            });
        }

        for &qubit in qubits {
            self.validate_qubit(qubit)?;
        }

        if let Gate::Phase(theta) = gate
            && !theta.is_finite()
        {
            return Err(CircuitError::InvalidPhaseAngle);
        }

        if let Gate::Unitary2x2(matrix) = &gate
            && !is_unitary_2x2(matrix)
        {
            return Err(CircuitError::InvalidUnitary);
        }

        if duration_ns.is_some_and(|duration| !duration.is_finite() || duration < 0.0) {
            return Err(CircuitError::InvalidGateDuration);
        }

        if expected == 2 && qubits[0] == qubits[1] {
            return Err(CircuitError::RepeatedQubit { qubit: qubits[0] });
        }

        self.operations.push(Operation::ApplyGate {
            gate,
            qubits: qubits.to_vec(),
            duration_ns,
        });
        Ok(self)
    }

    /// Applies a Pauli-X (NOT) gate to flip the target qubit.
    ///
    /// # Arguments
    ///
    /// * `target` - The index of the qubit to flip.
    ///
    /// # Returns
    /// A mutable reference to this circuit for further construction.
    ///
    /// # Errors
    /// Returns [`CircuitError::QubitOutOfBounds`] if `target` is outside the circuit,
    /// or [`CircuitError::InvalidPhaseAngle`] if `theta` is not finite.
    pub fn x(&mut self, target: usize) -> Result<&mut Self, CircuitError> {
        self.apply_gate(Gate::PauliX, &[target], None)
    }

    /// Applies a Pauli-Y gate to flip the amplitude of the target qubit
    /// and add a phase to it.
    ///
    /// # Arguments
    ///
    /// * `target` - The index of the qubit to flip and add a phase to.
    ///
    /// # Returns
    /// A mutable reference to this circuit for further construction.
    ///
    /// # Errors
    /// Returns [`CircuitError::QubitOutOfBounds`] if `target` is outside the circuit.
    pub fn y(&mut self, target: usize) -> Result<&mut Self, CircuitError> {
        self.apply_gate(Gate::PauliY, &[target], None)
    }

    /// Applies a Pauli-Z gate to add a phase to the amplitude of the target qubit.
    ///
    /// # Arguments
    ///
    /// * `target` - The index of the qubit to add a phase to.
    ///
    /// # Returns
    /// A mutable reference to this circuit for further construction.
    ///
    /// # Errors
    /// Returns [`CircuitError::QubitOutOfBounds`] if `target` is outside the circuit.
    pub fn z(&mut self, target: usize) -> Result<&mut Self, CircuitError> {
        self.apply_gate(Gate::PauliZ, &[target], None)
    }

    /// Applies a Hadamard gate to place the target qubit into an equal superposition.
    ///
    /// # Arguments
    ///
    /// * `target` - The index of the qubit to apply the gate to.
    ///
    /// # Returns
    /// A mutable reference to this circuit for further construction.
    ///
    /// # Errors
    /// Returns [`CircuitError::QubitOutOfBounds`] if `target` is outside the circuit.
    pub fn h(&mut self, target: usize) -> Result<&mut Self, CircuitError> {
        self.apply_gate(Gate::Hadamard, &[target], None)
    }

    /// Applies a parameterised phase rotation around the Z-axis of the Bloch sphere.
    ///
    /// # Arguments
    ///
    /// * `target` - The index of the qubit to rotate.
    /// * `theta` - The rotation angle in radians.
    ///
    /// # Returns
    /// A mutable reference to this circuit for further construction.
    ///
    /// # Errors
    /// Returns [`CircuitError::QubitOutOfBounds`] if `target` is outside the circuit.
    pub fn phase(&mut self, target: usize, theta: f64) -> Result<&mut Self, CircuitError> {
        self.apply_gate(Gate::Phase(theta), &[target], None)
    }

    /// Applies a custom single-qubit unitary, optionally followed by an error matrix.
    ///
    /// Both matrices use row-major element order `[m00, m01, m10, m11]`. When
    /// `error_matrix` is `Some`, it is applied after `matrix`, so the effective
    /// matrix is `error_matrix * matrix`. The duration is attached to this operation
    /// and is independent of the simulator's shared gate configuration.
    ///
    /// # Arguments
    /// * `target` - The qubit on which to apply the matrix.
    /// * `matrix` - The requested unitary matrix.
    /// * `error_matrix` - An optional unitary error matrix applied after `matrix`.
    /// * `duration_ns` - The duration of this unitary operation in nanoseconds.
    ///
    /// # Returns
    /// A mutable reference to this circuit for further construction.
    ///
    /// # Errors
    /// Returns [`CircuitError::QubitOutOfBounds`] if `target` is outside the circuit,
    /// [`CircuitError::InvalidUnitary`] if either supplied matrix is not a finite unitary matrix,
    /// or [`CircuitError::InvalidGateDuration`] if the duration is negative or not finite.
    pub fn unitary(
        &mut self,
        target: usize,
        matrix: [Complex64; 4],
        error_matrix: Option<[Complex64; 4]>,
        duration_ns: f64,
    ) -> Result<&mut Self, CircuitError> {
        if !is_unitary_2x2(&matrix)
            || error_matrix
                .as_ref()
                .is_some_and(|error| !is_unitary_2x2(error))
        {
            return Err(CircuitError::InvalidUnitary);
        }

        let effective_matrix = error_matrix
            .map(|error| multiply_2x2(error, matrix))
            .unwrap_or(matrix);
        self.apply_gate(
            Gate::Unitary2x2(effective_matrix),
            &[target],
            Some(duration_ns),
        )
    }

    /// Applies a Controlled-NOT (CNOT) gate to entangle two qubits.
    ///
    /// The target qubit is flipped only if the control qubit is in the |1> state.
    ///
    /// # Arguments
    ///
    /// * `control` - The index of the control qubit.
    /// * `target` - The index of the target qubit to flip.
    ///
    /// # Returns
    /// A mutable reference to this circuit for further construction.
    ///
    /// # Errors
    /// Returns [`CircuitError::QubitOutOfBounds`] if either qubit is outside the circuit,
    /// or [`CircuitError::RepeatedQubit`] if both operands are the same.
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<&mut Self, CircuitError> {
        self.apply_gate(Gate::CNOT, &[control, target], None)
    }

    /// Applies a Controlled-Z gate to two qubits.
    ///
    /// The target qubit gets a phase if the control qubit and target qubit are in the |1> state.
    ///
    /// # Arguments
    ///
    /// * `control` - The index of the control qubit.
    /// * `target` - The index of the target qubit to apply a phase to.
    ///
    /// # Returns
    /// A mutable reference to this circuit for further construction.
    ///
    /// # Errors
    /// Returns [`CircuitError::QubitOutOfBounds`] if either qubit is outside the circuit,
    /// or [`CircuitError::RepeatedQubit`] if both operands are the same.
    pub fn cz(&mut self, control: usize, target: usize) -> Result<&mut Self, CircuitError> {
        self.apply_gate(Gate::CZ, &[control, target], None)
    }

    /// Applies a SWAP gate to exchange the states of two qubits.
    ///
    /// # Arguments
    ///
    /// * `q1` - The index of the first qubit.
    /// * `q2` - The index of the second qubit.
    ///
    /// # Returns
    /// A mutable reference to this circuit for further construction.
    ///
    /// # Errors
    /// Returns [`CircuitError::QubitOutOfBounds`] if either qubit is outside the circuit,
    /// or [`CircuitError::RepeatedQubit`] if both operands are the same.
    pub fn swap(&mut self, q1: usize, q2: usize) -> Result<&mut Self, CircuitError> {
        self.apply_gate(Gate::SWAP, &[q1, q2], None)
    }

    /// Estimates circuit duration using the settings of a specific simulator.
    ///
    /// This sums the configured duration for every operation. It does not account for
    /// parallel scheduling or simulator runtime.
    ///
    /// # Arguments
    /// * `config` - The configuration whose gate durations should be used.
    ///
    /// # Returns
    /// The estimated circuit duration in nanoseconds.
    pub fn estimated_duration_ns(&self, config: &SimulatorConfig) -> f64 {
        self.operations
            .iter()
            .map(|operation| match operation {
                Operation::ApplyGate {
                    gate, duration_ns, ..
                } => duration_ns.unwrap_or_else(|| config.gate_duration_ns(gate)),
            })
            .sum()
    }

    /// Checks that a qubit index belongs to this circuit.
    ///
    /// # Arguments
    /// * `qubit` - The qubit index to validate.
    ///
    /// # Returns
    /// `Ok(())` if the index is valid.
    ///
    /// # Errors
    /// Returns [`CircuitError::QubitOutOfBounds`] if `qubit` is outside the circuit.
    fn validate_qubit(&self, qubit: usize) -> Result<(), CircuitError> {
        if qubit >= self.num_qubits {
            return Err(CircuitError::QubitOutOfBounds {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        Ok(())
    }
}

/// Checks whether a row-major 2x2 matrix is finite and unitary within tolerance.
///
/// # Arguments
/// * `matrix` - The matrix elements ordered as `[m00, m01, m10, m11]`.
///
/// # Returns
/// `true` if the columns are orthonormal within the validation tolerance; otherwise `false`.
fn is_unitary_2x2(matrix: &[Complex64; 4]) -> bool {
    let tolerance = 1e-10;
    if matrix
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return false;
    }

    let first_column_norm = matrix[0].norm_sqr() + matrix[2].norm_sqr();
    let second_column_norm = matrix[1].norm_sqr() + matrix[3].norm_sqr();
    let columns_inner_product = matrix[0].conj() * matrix[1] + matrix[2].conj() * matrix[3];

    (first_column_norm - 1.0).abs() <= tolerance
        && (second_column_norm - 1.0).abs() <= tolerance
        && columns_inner_product.norm() <= tolerance
}

fn multiply_2x2(lhs: [Complex64; 4], rhs: [Complex64; 4]) -> [Complex64; 4] {
    [
        lhs[0] * rhs[0] + lhs[1] * rhs[2],
        lhs[0] * rhs[1] + lhs[1] * rhs[3],
        lhs[2] * rhs[0] + lhs[3] * rhs[2],
        lhs[2] * rhs[1] + lhs[3] * rhs[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::super::gate::Gate;
    use super::super::operation::Operation;
    use super::*;
    use crate::core::config::GatePropsConfig;

    #[test]
    fn test_circuit_new_creates_new_circuit() {
        let circuit = Circuit::new(3);

        assert_eq!(circuit.num_qubits(), 3);
        assert!(
            circuit.operations().is_empty(),
            "Operations vector should be empty on initialisation"
        );
    }

    #[test]
    fn test_circuit_add_operation_adds_operation() {
        let mut circuit = Circuit::new(2);

        circuit.apply_gate(Gate::Hadamard, &[0], None).unwrap();

        assert_eq!(circuit.operations().len(), 1);

        assert_eq!(
            circuit.operations()[0],
            Operation::ApplyGate {
                gate: Gate::Hadamard,
                qubits: vec![0],
                duration_ns: None,
            }
        );
    }

    #[test]
    fn test_circuit_x_adds_x_gate() {
        let mut circuit = Circuit::new(1);

        circuit.x(0).unwrap();

        assert_eq!(circuit.operations().len(), 1);

        assert_eq!(
            circuit.operations()[0],
            Operation::ApplyGate {
                gate: Gate::PauliX,
                qubits: vec![0],
                duration_ns: None,
            }
        );
    }

    #[test]
    fn test_circuit_y_adds_y_gate() {
        let mut circuit = Circuit::new(1);

        circuit.y(0).unwrap();

        assert_eq!(circuit.operations().len(), 1);

        assert_eq!(
            circuit.operations()[0],
            Operation::ApplyGate {
                gate: Gate::PauliY,
                qubits: vec![0],
                duration_ns: None,
            }
        );
    }

    #[test]
    fn test_circuit_z_adds_y_gate() {
        let mut circuit = Circuit::new(1);

        circuit.z(0).unwrap();

        assert_eq!(circuit.operations().len(), 1);

        assert_eq!(
            circuit.operations()[0],
            Operation::ApplyGate {
                gate: Gate::PauliZ,
                qubits: vec![0],
                duration_ns: None,
            }
        );
    }

    #[test]
    fn test_circuit_h_adds_h_gate() {
        let mut circuit = Circuit::new(1);

        circuit.h(0).unwrap();

        assert_eq!(circuit.operations().len(), 1);

        assert_eq!(
            circuit.operations()[0],
            Operation::ApplyGate {
                gate: Gate::Hadamard,
                qubits: vec![0],
                duration_ns: None,
            }
        );
    }

    #[test]
    fn test_circuit_phase_adds_phase_gate() {
        let mut circuit = Circuit::new(1);

        circuit.phase(0, 0.0).unwrap();

        assert_eq!(circuit.operations().len(), 1);

        assert_eq!(
            circuit.operations()[0],
            Operation::ApplyGate {
                gate: Gate::Phase(0.0),
                qubits: vec![0],
                duration_ns: None,
            }
        );
    }

    #[test]
    fn test_phase_rejects_non_finite_angles_without_mutating_circuit() {
        for theta in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut circuit = Circuit::new(1);

            assert_eq!(
                circuit.phase(0, theta).unwrap_err(),
                CircuitError::InvalidPhaseAngle
            );
            assert!(circuit.operations().is_empty());
        }
    }

    #[test]
    fn test_circuit_cnot_adds_cnot_gate() {
        let mut circuit = Circuit::new(2);

        circuit.cnot(0, 1).unwrap();

        assert_eq!(circuit.operations().len(), 1);

        assert_eq!(
            circuit.operations()[0],
            Operation::ApplyGate {
                gate: Gate::CNOT,
                qubits: vec![0, 1],
                duration_ns: None,
            }
        );
    }

    #[test]
    fn test_circuit_cz_adds_cz_gate() {
        let mut circuit = Circuit::new(2);

        circuit.cz(0, 1).unwrap();

        assert_eq!(circuit.operations().len(), 1);

        assert_eq!(
            circuit.operations()[0],
            Operation::ApplyGate {
                gate: Gate::CZ,
                qubits: vec![0, 1],
                duration_ns: None,
            }
        );
    }

    #[test]
    fn test_circuit_swap_adds_swap_gate() {
        let mut circuit = Circuit::new(2);

        circuit.swap(0, 1).unwrap();

        assert_eq!(circuit.operations().len(), 1);

        assert_eq!(
            circuit.operations()[0],
            Operation::ApplyGate {
                gate: Gate::SWAP,
                qubits: vec![0, 1],
                duration_ns: None,
            }
        );
    }

    #[test]
    fn test_validated_builder_accepts_valid_indices() {
        let mut circuit = Circuit::new(3);

        circuit.x(0).unwrap().y(1).unwrap().z(2).unwrap();
        assert_eq!(circuit.operations().len(), 3);
    }

    #[test]
    fn test_invalid_gate_does_not_mutate_circuit() {
        let mut circuit = Circuit::new(2);

        assert_eq!(
            circuit.x(2).unwrap_err(),
            CircuitError::QubitOutOfBounds {
                qubit: 2,
                num_qubits: 2
            }
        );
        assert!(circuit.operations().is_empty());

        let mut zero_qubit_circuit = Circuit::new(0);
        assert_eq!(
            zero_qubit_circuit.x(0).unwrap_err(),
            CircuitError::QubitOutOfBounds {
                qubit: 0,
                num_qubits: 0
            }
        );
    }

    #[test]
    fn test_apply_gate_rejects_wrong_arity_and_repeated_qubits() {
        let mut circuit = Circuit::new(2);

        assert_eq!(
            circuit.apply_gate(Gate::PauliX, &[0, 1], None).unwrap_err(),
            CircuitError::InvalidArity {
                expected: 1,
                actual: 2
            }
        );
        assert_eq!(
            circuit.cnot(1, 1).unwrap_err(),
            CircuitError::RepeatedQubit { qubit: 1 }
        );
        assert!(circuit.operations().is_empty());
    }

    #[test]
    fn test_unitary_rejects_non_unitary_matrix() {
        let mut circuit = Circuit::new(1);
        let matrix = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
        ];

        assert_eq!(
            circuit.unitary(0, matrix, None, 4.0).unwrap_err(),
            CircuitError::InvalidUnitary
        );
        assert!(circuit.operations().is_empty());
    }

    #[test]
    fn test_apply_gate_rejects_non_unitary_or_non_finite_unitary_matrices() {
        let invalid_matrices = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            [
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ];

        for matrix in invalid_matrices {
            let mut circuit = Circuit::new(1);

            assert_eq!(
                circuit
                    .apply_gate(Gate::Unitary2x2(matrix), &[0], None)
                    .unwrap_err(),
                CircuitError::InvalidUnitary
            );
            assert!(circuit.operations().is_empty());
        }
    }

    #[test]
    fn test_unitary_rejects_invalid_duration_without_mutating_circuit() {
        let identity = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        for duration_ns in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut circuit = Circuit::new(1);

            assert_eq!(
                circuit.unitary(0, identity, None, duration_ns).unwrap_err(),
                CircuitError::InvalidGateDuration
            );
            assert!(circuit.operations().is_empty());
        }
    }

    #[test]
    fn test_estimated_duration_uses_per_unitary_duration() {
        let mut config = SimulatorConfig::default();
        config.gates.insert(
            "unitary".to_string(),
            GatePropsConfig {
                time_ns: Some(100.0),
                rot_angle_err_rad: None,
                error_matrix: None,
            },
        );
        config.gates.insert(
            "x".to_string(),
            GatePropsConfig {
                time_ns: Some(7.0),
                rot_angle_err_rad: None,
                error_matrix: None,
            },
        );

        let identity = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let mut circuit = Circuit::new(1);
        circuit
            .unitary(0, identity, None, 2.5)
            .unwrap()
            .unitary(0, identity, None, 3.5)
            .unwrap()
            .x(0)
            .unwrap();

        assert_eq!(circuit.estimated_duration_ns(&config), 13.0);
    }

    #[test]
    fn test_unitary_composes_optional_error_after_gate() {
        let mut circuit = Circuit::new(1);
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

        circuit.unitary(0, x, Some(z), 12.5).unwrap();

        assert_eq!(
            circuit.operations()[0],
            Operation::ApplyGate {
                gate: Gate::Unitary2x2([
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ]),
                qubits: vec![0],
                duration_ns: Some(12.5),
            }
        );
    }

    #[test]
    fn test_circuit_error_display_messages() {
        let cases = [
            (
                CircuitError::QubitOutOfBounds {
                    qubit: 3,
                    num_qubits: 2,
                },
                "qubit index 3 is out of bounds for a 2-qubit circuit",
            ),
            (
                CircuitError::InvalidArity {
                    expected: 2,
                    actual: 1,
                },
                "gate expects 2 qubits, got 1",
            ),
            (
                CircuitError::RepeatedQubit { qubit: 1 },
                "qubit 1 cannot fill multiple gate operands",
            ),
            (
                CircuitError::InvalidUnitary,
                "matrix is not a valid 2x2 unitary",
            ),
            (
                CircuitError::InvalidPhaseAngle,
                "phase angle must be finite",
            ),
            (
                CircuitError::InvalidGateDuration,
                "gate duration must be finite and non-negative",
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn test_unitary_rejects_invalid_error_matrix_without_mutating_circuit() {
        let mut circuit = Circuit::new(1);
        let identity = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let invalid_error = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
        ];

        assert_eq!(
            circuit
                .unitary(0, identity, Some(invalid_error), 4.0)
                .unwrap_err(),
            CircuitError::InvalidUnitary
        );
        assert!(circuit.operations().is_empty());
    }
}
