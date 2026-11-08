use num_complex::Complex64;
use std::str::FromStr;

/// Represents an abstract quantum logic gate.
///
/// This enum defines the pure mathematical operation. It does not contain
/// routing information (such as which qubits to target). Circuit operations provide
/// that information when a gate is added.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum Gate {
    /// The Pauli-X (NOT) gate. Flips the state of the target qubit.
    PauliX,

    /// The Pauli-Y gate. Applies a bit and phase flip to the target qubit.
    PauliY,

    /// The Pauli-Z (Phase Flip) gate. Flips the sign of the |1> state.
    PauliZ,

    /// The Hadamard gate. Places the target qubit into equal superposition.
    Hadamard,

    /// A parameterized phase-shift gate.
    ///
    /// # Contains
    /// * `f64` - The rotation angle (theta) in radians.
    Phase(f64),

    /// The Controlled-NOT (CNOT) gate.
    CNOT,

    /// The Controlled-Z (CZ) gate.
    CZ,

    /// The SWAP gate. Exchanges the states of two qubits.
    SWAP,

    /// A custom arbitrary single-qubit unitary gate.
    ///
    /// Its duration is provided for each application through the circuit API. A
    /// configured explicit error matrix can be applied afterward, but a
    /// rotation-angle error is undefined for an arbitrary unitary.
    ///
    /// # Contains
    /// * `[Complex64; 4]` - The 4 elements of a 2x2 unitary matrix.
    Unitary2x2([Complex64; 4]),
}

impl Gate {
    /// Returns the number of qubit operands required by this gate.
    ///
    /// # Returns
    /// `2` for controlled and swap gates, and `1` for all other gates.
    pub(crate) const fn qubit_arity(&self) -> usize {
        match self {
            Self::CNOT | Self::CZ | Self::SWAP => 2,
            _ => 1,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
/// Error returned when a string does not identify a supported gate name.
pub struct ParseGateError;

impl FromStr for Gate {
    type Err = ParseGateError;

    /// Parses a gate name without regard to letter case.
    ///
    /// Supported names are `x`, `y`, `z`, `h`, `phase`, `cnot`, `cz`, `swap`,
    /// and `unitary`. Parameterized gates use placeholder values; configuration
    /// loading uses this parser only to validate gate names.
    ///
    /// # Arguments
    /// * `name` - The gate name to parse.
    ///
    /// # Returns
    /// The corresponding gate variant, or [`ParseGateError`] if the name is unsupported.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name.to_lowercase().as_str() {
            "x" => Ok(Gate::PauliX),
            "y" => Ok(Gate::PauliY),
            "z" => Ok(Gate::PauliZ),
            "h" => Ok(Gate::Hadamard),
            "phase" => Ok(Gate::Phase(0.0)),
            "cnot" => Ok(Gate::CNOT),
            "cz" => Ok(Gate::CZ),
            "swap" => Ok(Gate::SWAP),
            "unitary" => Ok(Gate::Unitary2x2([Complex64::new(0.0, 0.0); 4])),
            _ => Err(ParseGateError),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_from_str_standard_gates() {
        assert_eq!("x".parse::<Gate>(), Ok(Gate::PauliX));
        assert_eq!("y".parse::<Gate>(), Ok(Gate::PauliY));
        assert_eq!("z".parse::<Gate>(), Ok(Gate::PauliZ));
        assert_eq!("h".parse::<Gate>(), Ok(Gate::Hadamard));
        assert_eq!("cnot".parse::<Gate>(), Ok(Gate::CNOT));
        assert_eq!("cz".parse::<Gate>(), Ok(Gate::CZ));
        assert_eq!("swap".parse::<Gate>(), Ok(Gate::SWAP));
    }

    #[test]
    fn test_from_str_parameterised_gates_use_dummy_values() {
        assert_eq!("phase".parse::<Gate>(), Ok(Gate::Phase(0.0)));
        assert_eq!(
            "unitary".parse::<Gate>(),
            Ok(Gate::Unitary2x2([Complex64::new(0.0, 0.0); 4]))
        );
    }

    #[test]
    fn test_from_str_is_case_insensitive() {
        assert_eq!("X".parse::<Gate>(), Ok(Gate::PauliX));
        assert_eq!("Cnot".parse::<Gate>(), Ok(Gate::CNOT));
        assert_eq!("pHaSe".parse::<Gate>(), Ok(Gate::Phase(0.0)));
        assert_eq!(
            "UNITARY".parse::<Gate>(),
            Ok(Gate::Unitary2x2([Complex64::new(0.0, 0.0); 4]))
        );
    }

    #[test]
    fn test_from_str_invalid_returns_none() {
        assert_eq!("Invalidgate".parse::<Gate>(), Err(ParseGateError));
        assert_eq!("".parse::<Gate>(), Err(ParseGateError));
        assert_eq!("hgate".parse::<Gate>(), Err(ParseGateError));
        assert_eq!("cx".parse::<Gate>(), Err(ParseGateError));
    }

    #[test]
    fn test_gate_qubit_arity_metadata() {
        assert_eq!(Gate::PauliX.qubit_arity(), 1);
        assert_eq!(Gate::Phase(0.0).qubit_arity(), 1);
        assert_eq!(
            Gate::Unitary2x2([Complex64::new(0.0, 0.0); 4]).qubit_arity(),
            1
        );
        assert_eq!(Gate::CNOT.qubit_arity(), 2);
        assert_eq!(Gate::CZ.qubit_arity(), 2);
        assert_eq!(Gate::SWAP.qubit_arity(), 2);
    }
}
