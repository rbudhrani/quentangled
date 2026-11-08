use super::gate::Gate;

/// A validated gate application stored in a circuit's execution order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Operation {
    /// Applies a gate to the listed qubit operands.
    ApplyGate {
        /// The logical gate to apply.
        gate: Gate,
        /// Qubit indices in the operand order defined by `gate`.
        ///
        /// Circuit construction validates the operand count, bounds, and distinctness
        /// requirements before storing this operation.
        qubits: Vec<usize>,
        /// Per-operation duration override, used for custom unitary gates.
        duration_ns: Option<f64>,
    },
}
