use crate::core::config::SimulatorConfig;
use crate::core::gate::Gate;
use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

/// A 2x2 complex matrix stored in row-major order: `[m00, m01, m10, m11]`.
pub type Matrix2x2 = [Complex64; 4];
/// A 4x4 complex matrix stored in row-major order.
pub type Matrix4x4 = [Complex64; 16];

/// The gate matrices and phase error configured for a simulator.
#[derive(Debug, Clone)]
pub struct GateMatrixRegistry {
    /// Pauli-X matrix including the configured rotation-angle error.
    pub x: Matrix2x2,
    /// Pauli-Y matrix including the configured rotation-angle error.
    pub y: Matrix2x2,
    /// Pauli-Z matrix including the configured rotation-angle error.
    pub z: Matrix2x2,
    /// Hadamard matrix including the configured rotation-angle error.
    pub h: Matrix2x2,
    /// CNOT target matrix including the configured rotation-angle error.
    pub cnot: Matrix2x2,
    /// CZ target matrix including the configured rotation-angle error.
    pub cz: Matrix2x2,
    /// SWAP matrix for the `|01⟩`/`|10⟩` subspace, including configured error.
    pub swap: Matrix2x2,
    /// Rotation-angle error added to each requested phase-gate angle, in radians.
    pub phase_angle_err_rad: f64,
    /// Optional explicit errors applied after each standard single-qubit gate.
    pub x_error_matrix: Option<Matrix2x2>,
    pub y_error_matrix: Option<Matrix2x2>,
    pub z_error_matrix: Option<Matrix2x2>,
    pub h_error_matrix: Option<Matrix2x2>,
    pub phase_error_matrix: Option<Matrix2x2>,
    /// Optional explicit errors applied after the corresponding two-qubit gate.
    pub cnot_error_matrix: Option<Matrix4x4>,
    pub cz_error_matrix: Option<Matrix4x4>,
    pub swap_error_matrix: Option<Matrix4x4>,
    /// Optional configured error applied after a per-operation custom unitary.
    pub unitary_error_matrix: Option<Matrix2x2>,
}

impl GateMatrixRegistry {
    /// Builds the Pauli-X matrix with a coherent rotation-angle error.
    ///
    /// # Arguments
    /// * `error_rad` - The configured rotation-angle error in radians.
    ///
    /// # Returns
    /// The error-adjusted Pauli-X matrix in row-major order.
    fn x_matrix(error_rad: f64) -> Matrix2x2 {
        let sin_e = (error_rad / 2.0).sin();
        let cos_e = (error_rad / 2.0).cos();
        [
            Complex64::new(0.0, -sin_e),
            Complex64::new(cos_e, 0.0),
            Complex64::new(cos_e, 0.0),
            Complex64::new(0.0, -sin_e),
        ]
    }

    /// Builds the Pauli-Y matrix with a coherent rotation-angle error.
    ///
    /// # Arguments
    /// * `error_rad` - The configured rotation-angle error in radians.
    ///
    /// # Returns
    /// The error-adjusted Pauli-Y matrix in row-major order.
    fn y_matrix(error_rad: f64) -> Matrix2x2 {
        let sin_e = (error_rad / 2.0).sin();
        let cos_e = (error_rad / 2.0).cos();
        [
            Complex64::new(0.0, -sin_e),
            Complex64::new(0.0, -cos_e),
            Complex64::new(0.0, cos_e),
            Complex64::new(0.0, -sin_e),
        ]
    }

    /// Builds the Pauli-Z matrix with a coherent rotation-angle error.
    ///
    /// # Arguments
    /// * `error_rad` - The configured rotation-angle error in radians.
    ///
    /// # Returns
    /// The error-adjusted Pauli-Z matrix in row-major order.
    fn z_matrix(error_rad: f64) -> Matrix2x2 {
        let sin_e = (error_rad / 2.0).sin();
        let cos_e = (error_rad / 2.0).cos();
        [
            Complex64::new(cos_e, -sin_e),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-cos_e, -sin_e),
        ]
    }

    /// Builds the Hadamard matrix with a coherent rotation-angle error.
    ///
    /// # Arguments
    /// * `error_rad` - The configured rotation-angle error in radians.
    ///
    /// # Returns
    /// The error-adjusted Hadamard matrix in row-major order.
    fn h_matrix(error_rad: f64) -> Matrix2x2 {
        let sin_e = (error_rad / 2.0).sin();
        let cos_e = (error_rad / 2.0).cos();
        [
            Complex64::new(cos_e * FRAC_1_SQRT_2, -sin_e),
            Complex64::new(cos_e * FRAC_1_SQRT_2, 0.0),
            Complex64::new(cos_e * FRAC_1_SQRT_2, 0.0),
            Complex64::new(-cos_e * FRAC_1_SQRT_2, -sin_e),
        ]
    }

    /// Builds the single-qubit matrices using a simulator's gate configuration.
    ///
    /// # Arguments
    /// * `config` - The configuration containing per-gate rotation-angle errors.
    ///
    /// # Returns
    /// A registry of gate matrices plus the phase-angle error.
    pub fn from_config(config: &SimulatorConfig) -> Self {
        Self {
            x: Self::x_matrix(config.gate_rotation_error_rad(&Gate::PauliX)),
            y: Self::y_matrix(config.gate_rotation_error_rad(&Gate::PauliY)),
            z: Self::z_matrix(config.gate_rotation_error_rad(&Gate::PauliZ)),
            h: Self::h_matrix(config.gate_rotation_error_rad(&Gate::Hadamard)),
            cnot: Self::x_matrix(config.gate_rotation_error_rad(&Gate::CNOT)),
            cz: Self::z_matrix(config.gate_rotation_error_rad(&Gate::CZ)),
            swap: Self::x_matrix(config.gate_rotation_error_rad(&Gate::SWAP)),
            phase_angle_err_rad: config.gate_rotation_error_rad(&Gate::Phase(0.0)),
            x_error_matrix: configured_matrix::<4>(config, &Gate::PauliX),
            y_error_matrix: configured_matrix::<4>(config, &Gate::PauliY),
            z_error_matrix: configured_matrix::<4>(config, &Gate::PauliZ),
            h_error_matrix: configured_matrix::<4>(config, &Gate::Hadamard),
            phase_error_matrix: configured_matrix::<4>(config, &Gate::Phase(0.0)),
            cnot_error_matrix: configured_matrix::<16>(config, &Gate::CNOT),
            cz_error_matrix: configured_matrix::<16>(config, &Gate::CZ),
            swap_error_matrix: configured_matrix::<16>(config, &Gate::SWAP),
            unitary_error_matrix: configured_matrix::<4>(
                config,
                &Gate::Unitary2x2([Complex64::new(0.0, 0.0); 4]),
            ),
        }
    }

    /// Generates a phase gate for the requested angle, including the configured error.
    ///
    /// # Arguments
    /// * `theta` - The requested phase angle in radians.
    ///
    /// # Returns
    /// The phase matrix for `theta + phase_angle_err_rad`, in row-major order.
    pub fn phase(&self, theta: f64) -> Matrix2x2 {
        let angle = theta + self.phase_angle_err_rad;

        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(angle.cos(), angle.sin()),
        ]
    }
}

impl Default for GateMatrixRegistry {
    /// Creates a registry of ideal single-qubit gates with zero phase error.
    fn default() -> Self {
        Self {
            x: [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            y: [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
            z: [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
            h: [
                Complex64::new(FRAC_1_SQRT_2, 0.0),
                Complex64::new(FRAC_1_SQRT_2, 0.0),
                Complex64::new(FRAC_1_SQRT_2, 0.0),
                Complex64::new(-FRAC_1_SQRT_2, 0.0),
            ],
            cnot: [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            cz: [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
            swap: [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            phase_angle_err_rad: 0.0,
            x_error_matrix: None,
            y_error_matrix: None,
            z_error_matrix: None,
            h_error_matrix: None,
            phase_error_matrix: None,
            cnot_error_matrix: None,
            cz_error_matrix: None,
            swap_error_matrix: None,
            unitary_error_matrix: None,
        }
    }
}

fn configured_matrix<const N: usize>(
    config: &SimulatorConfig,
    gate: &Gate,
) -> Option<[Complex64; N]> {
    config.gate_error_matrix(gate).map(|elements| {
        std::array::from_fn(|index| Complex64::new(elements[index][0], elements[index][1]))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper to compare complex matrices within a small margin of error.
    ///
    /// # Arguments
    /// * `a` - The first row-major matrix to compare.
    /// * `b` - The second row-major matrix to compare.
    ///
    /// # Returns
    /// `true` if every real and imaginary component differs by less than `1e-10`.
    fn approx_eq_matrix(a: Matrix2x2, b: Matrix2x2) -> bool {
        (a[0].re - b[0].re).abs() < 1e-10
            && (a[0].im - b[0].im).abs() < 1e-10
            && (a[1].re - b[1].re).abs() < 1e-10
            && (a[1].im - b[1].im).abs() < 1e-10
            && (a[2].re - b[2].re).abs() < 1e-10
            && (a[2].im - b[2].im).abs() < 1e-10
            && (a[3].re - b[3].re).abs() < 1e-10
            && (a[3].im - b[3].im).abs() < 1e-10
    }

    #[test]
    fn test_err_gate_matrix_registry() {
        let config: SimulatorConfig = toml::from_str(
            "[gates.x]\nrot_angle_err_rad = 1.5707963267948966\n\
             [gates.y]\nrot_angle_err_rad = 1.5707963267948966\n\
             [gates.z]\nrot_angle_err_rad = 1.5707963267948966\n\
             [gates.h]\nrot_angle_err_rad = 1.5707963267948966\n\
             [gates.cnot]\nrot_angle_err_rad = 1.5707963267948966\n\
             [gates.cz]\nrot_angle_err_rad = 1.5707963267948966\n\
             [gates.swap]\nrot_angle_err_rad = 1.5707963267948966\n\
             [gates.phase]\nrot_angle_err_rad = 1.5707963267948966\n",
        )
        .unwrap();
        let err_gate_matrix_registry = GateMatrixRegistry::from_config(&config);
        let expected_err_x_matrix = [
            Complex64::new(0.0, -FRAC_1_SQRT_2),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(0.0, -FRAC_1_SQRT_2),
        ];
        let expected_err_y_matrix = [
            Complex64::new(0.0, -FRAC_1_SQRT_2),
            Complex64::new(0.0, -FRAC_1_SQRT_2),
            Complex64::new(0.0, FRAC_1_SQRT_2),
            Complex64::new(0.0, -FRAC_1_SQRT_2),
        ];
        let expected_err_z_matrix = [
            Complex64::new(FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
        ];
        let expected_err_h_matrix = [
            Complex64::new(0.5, -FRAC_1_SQRT_2),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(-0.5, -FRAC_1_SQRT_2),
        ];
        let expected_err_cnot_matrix = [
            Complex64::new(0.0, -FRAC_1_SQRT_2),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(0.0, -FRAC_1_SQRT_2),
        ];
        let expected_err_phase_matrix = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ];

        assert!(approx_eq_matrix(
            expected_err_x_matrix,
            err_gate_matrix_registry.x
        ));
        assert!(approx_eq_matrix(
            expected_err_y_matrix,
            err_gate_matrix_registry.y
        ));
        assert!(approx_eq_matrix(
            expected_err_z_matrix,
            err_gate_matrix_registry.z
        ));
        assert!(approx_eq_matrix(
            expected_err_h_matrix,
            err_gate_matrix_registry.h
        ));
        assert!(approx_eq_matrix(
            expected_err_cnot_matrix,
            err_gate_matrix_registry.cnot
        ));
        assert!(approx_eq_matrix(
            expected_err_z_matrix,
            err_gate_matrix_registry.cz
        ));
        assert!(approx_eq_matrix(
            expected_err_cnot_matrix,
            err_gate_matrix_registry.swap
        ));
        assert!(approx_eq_matrix(
            expected_err_phase_matrix,
            err_gate_matrix_registry.phase(PI / 2.0)
        ));
    }
}
