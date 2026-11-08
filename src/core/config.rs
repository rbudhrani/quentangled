use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::Path;

use crate::core::gate::Gate;

const DEFAULT_PARALLELISM_THRESHOLD: usize = 16_384;
const DEFAULT_CHUNK_SIZE: usize = 4_096;

#[derive(Debug, Clone, PartialEq, Eq)]
/// An error encountered while loading or validating simulator configuration.
pub enum ConfigError {
    /// The parallel chunk size is zero.
    InvalidChunkSize,
    /// A gate settings table names an unsupported gate.
    UnknownGate(String),
    /// Multiple gate settings tables name the same gate, ignoring case.
    DuplicateGate(String),
    /// A gate duration is negative or not finite.
    InvalidGateDuration {
        /// The gate whose duration is invalid.
        gate: String,
    },
    /// A gate rotation error is not finite.
    InvalidGateRotationError {
        /// The gate whose rotation error is invalid.
        gate: String,
    },
    /// A rotation-angle error is not defined for arbitrary custom unitaries.
    UnsupportedGateRotationError {
        /// The gate whose rotation error is unsupported.
        gate: String,
    },
    /// A gate config specifies both supported error models.
    ConflictingGateErrors {
        /// The gate whose error settings conflict.
        gate: String,
    },
    /// An error matrix has the wrong number of elements for the gate.
    InvalidGateErrorMatrixSize {
        /// The gate whose matrix has the wrong size.
        gate: String,
        /// The number of complex elements required.
        expected: usize,
        /// The number of complex elements supplied.
        actual: usize,
    },
    /// An error matrix contains non-finite elements or is not unitary.
    InvalidGateErrorMatrix {
        /// The gate whose matrix is invalid.
        gate: String,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidChunkSize => write!(formatter, "chunk_size must be greater than 0"),
            Self::UnknownGate(name) => write!(formatter, "unknown gate configuration `{name}`"),
            Self::DuplicateGate(name) => {
                write!(formatter, "duplicate gate configuration `{name}`")
            }
            Self::InvalidGateDuration { gate } => write!(
                formatter,
                "gate duration for `{gate}` must be finite and non-negative"
            ),
            Self::InvalidGateRotationError { gate } => {
                write!(formatter, "rotation error for `{gate}` must be finite")
            }
            Self::UnsupportedGateRotationError { gate } => write!(
                formatter,
                "rotation-angle errors are not supported for arbitrary gate `{gate}`"
            ),
            Self::ConflictingGateErrors { gate } => write!(
                formatter,
                "gate `{gate}` cannot configure both rot_angle_err_rad and error_matrix"
            ),
            Self::InvalidGateErrorMatrixSize {
                gate,
                expected,
                actual,
            } => write!(
                formatter,
                "error matrix for `{gate}` must contain {expected} complex elements, got {actual}"
            ),
            Self::InvalidGateErrorMatrix { gate } => {
                write!(
                    formatter,
                    "error matrix for `{gate}` must be finite and unitary"
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Configuration owned by one simulator instance.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SimulatorConfig {
    /// CPU parallelism settings used by this simulator instance.
    #[serde(default)]
    pub engine: EngineConfig,
    /// Optional physical properties keyed by gate name (for example, `x` or `cnot`).
    #[serde(default)]
    pub gates: HashMap<String, GatePropsConfig>,
}

impl SimulatorConfig {
    /// Loads and validates simulator settings from a TOML file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read, contains invalid TOML, or has invalid settings.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Validates engine settings, gate names, and gate property values.
    ///
    /// # Errors
    /// Returns a [`ConfigError`] describing the first invalid setting.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.engine.chunk_size == 0 {
            return Err(ConfigError::InvalidChunkSize);
        }

        let mut configured_names = HashSet::with_capacity(self.gates.len());
        for (name, properties) in &self.gates {
            if name.parse::<Gate>().is_err() {
                return Err(ConfigError::UnknownGate(name.clone()));
            }

            let normalized_name = name.to_ascii_lowercase();
            if !configured_names.insert(normalized_name.clone()) {
                return Err(ConfigError::DuplicateGate(normalized_name));
            }

            if properties
                .time_ns
                .is_some_and(|duration| !duration.is_finite() || duration < 0.0)
            {
                return Err(ConfigError::InvalidGateDuration { gate: name.clone() });
            }

            if properties
                .rot_angle_err_rad
                .is_some_and(|error| !error.is_finite())
            {
                return Err(ConfigError::InvalidGateRotationError { gate: name.clone() });
            }

            if properties.rot_angle_err_rad.is_some() && properties.error_matrix.is_some() {
                return Err(ConfigError::ConflictingGateErrors { gate: name.clone() });
            }

            if properties.rot_angle_err_rad.is_some()
                && matches!(name.to_ascii_lowercase().as_str(), "unitary")
            {
                return Err(ConfigError::UnsupportedGateRotationError { gate: name.clone() });
            }

            if let Some(matrix) = &properties.error_matrix {
                let dimension = 1
                    << name
                        .parse::<Gate>()
                        .expect("gate name validated")
                        .qubit_arity();
                let expected_elements = dimension * dimension;
                if matrix.len() != expected_elements {
                    return Err(ConfigError::InvalidGateErrorMatrixSize {
                        gate: name.clone(),
                        expected: expected_elements,
                        actual: matrix.len(),
                    });
                }

                if !is_unitary_error_matrix(matrix, dimension) {
                    return Err(ConfigError::InvalidGateErrorMatrix { gate: name.clone() });
                }
            }
        }

        Ok(())
    }

    pub(crate) fn gate_properties(&self, gate: &Gate) -> Option<&GatePropsConfig> {
        let name = gate_name(gate);
        self.gates.get(name).or_else(|| {
            self.gates
                .iter()
                .find(|(configured_name, _)| configured_name.eq_ignore_ascii_case(name))
                .map(|(_, properties)| properties)
        })
    }

    pub(crate) fn gate_duration_ns(&self, gate: &Gate) -> f64 {
        self.gate_properties(gate)
            .and_then(|properties| properties.time_ns)
            .unwrap_or(0.0)
    }

    pub(crate) fn gate_rotation_error_rad(&self, gate: &Gate) -> f64 {
        self.gate_properties(gate)
            .and_then(|properties| properties.rot_angle_err_rad)
            .unwrap_or(0.0)
    }

    pub(crate) fn gate_error_matrix(&self, gate: &Gate) -> Option<&[[f64; 2]]> {
        self.gate_properties(gate)
            .and_then(|properties| properties.error_matrix.as_deref())
    }
}

fn is_unitary_error_matrix(matrix: &[[f64; 2]], dimension: usize) -> bool {
    let values: Vec<_> = matrix
        .iter()
        .map(|[real, imaginary]| num_complex::Complex64::new(*real, *imaginary))
        .collect();
    let tolerance = 1e-10;

    if values
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return false;
    }

    for column_a in 0..dimension {
        for column_b in 0..dimension {
            let inner_product: num_complex::Complex64 = (0..dimension)
                .map(|row| {
                    values[row * dimension + column_a].conj() * values[row * dimension + column_b]
                })
                .sum();
            let expected = if column_a == column_b { 1.0 } else { 0.0 };
            if (inner_product.re - expected).abs() > tolerance || inner_product.im.abs() > tolerance
            {
                return false;
            }
        }
    }

    true
}

fn gate_name(gate: &Gate) -> &'static str {
    match gate {
        Gate::PauliX => "x",
        Gate::PauliY => "y",
        Gate::PauliZ => "z",
        Gate::Hadamard => "h",
        Gate::Phase(_) => "phase",
        Gate::CNOT => "cnot",
        Gate::CZ => "cz",
        Gate::SWAP => "swap",
        Gate::Unitary2x2(_) => "unitary",
    }
}

/// CPU and memory tuning parameters for one simulator.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default)]
/// CPU parallelism and chunking settings for one simulator instance.
pub struct EngineConfig {
    /// Minimum state-vector length at which gates use parallel execution.
    pub parallel_threshold: usize,
    /// Number of state-vector elements assigned to each parallel task; must be nonzero.
    pub chunk_size: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            parallel_threshold: DEFAULT_PARALLELISM_THRESHOLD,
            chunk_size: DEFAULT_CHUNK_SIZE,
        }
    }
}

/// Optional properties parsed for each gate from TOML.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GatePropsConfig {
    /// Gate duration in nanoseconds; if present, must be finite and non-negative.
    pub time_ns: Option<f64>,
    /// Rotation-angle error in radians; if present, must be finite.
    pub rot_angle_err_rad: Option<f64>,
    /// Optional row-major unitary error matrix, each complex element as `[real, imaginary]`.
    /// Use 4 elements for a single-qubit gate and 16 for a two-qubit gate.
    pub error_matrix: Option<Vec<[f64; 2]>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_config_uses_defaults() {
        let config: SimulatorConfig = toml::from_str("").unwrap();
        assert_eq!(
            config.engine.parallel_threshold,
            DEFAULT_PARALLELISM_THRESHOLD
        );
        assert_eq!(config.engine.chunk_size, DEFAULT_CHUNK_SIZE);
        assert!(config.gates.is_empty());
    }

    #[test]
    fn test_partial_engine_config_uses_field_defaults() {
        let config: SimulatorConfig =
            toml::from_str("[engine]\nparallel_threshold = 100\n").unwrap();

        assert_eq!(config.engine.parallel_threshold, 100);
        assert_eq!(config.engine.chunk_size, DEFAULT_CHUNK_SIZE);
    }

    #[test]
    fn test_config_keeps_gate_settings_instance_local() {
        let configured: SimulatorConfig =
            toml::from_str("[gates.x]\ntime_ns = 25.0\nrot_angle_err_rad = 0.2\n").unwrap();
        let defaults = SimulatorConfig::default();

        assert_eq!(configured.gate_duration_ns(&Gate::PauliX), 25.0);
        assert_eq!(configured.gate_rotation_error_rad(&Gate::PauliX), 0.2);
        assert_eq!(defaults.gate_duration_ns(&Gate::PauliX), 0.0);
    }

    #[test]
    fn test_from_file_returns_owned_config() {
        let path =
            std::env::temp_dir().join(format!("quentangled-config-{}.toml", std::process::id()));
        fs::write(
            &path,
            "[engine]\nparallel_threshold = 32\n[gates.x]\ntime_ns = 7.0\n",
        )
        .unwrap();

        let result = SimulatorConfig::from_file(&path);
        fs::remove_file(path).unwrap();
        let config = result.unwrap();

        assert_eq!(config.engine.parallel_threshold, 32);
        assert_eq!(config.engine.chunk_size, DEFAULT_CHUNK_SIZE);
        assert_eq!(config.gate_duration_ns(&Gate::PauliX), 7.0);
        assert_eq!(
            SimulatorConfig::default().engine.parallel_threshold,
            DEFAULT_PARALLELISM_THRESHOLD
        );
    }

    #[test]
    fn test_from_file_returns_error_when_file_cannot_be_read() {
        let path = std::env::temp_dir().join(format!(
            "quentangled-missing-config-{}-{}.toml",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));

        let error = SimulatorConfig::from_file(&path).unwrap_err();

        assert!(error.downcast_ref::<std::io::Error>().is_some());
    }

    #[test]
    fn test_from_file_returns_error_for_invalid_toml() {
        let path = std::env::temp_dir().join(format!(
            "quentangled-malformed-config-{}.toml",
            std::process::id()
        ));
        fs::write(&path, "[engine\nchunk_size = 4").unwrap();

        let result = SimulatorConfig::from_file(&path);
        fs::remove_file(path).unwrap();
        let error = result.unwrap_err();

        assert!(error.downcast_ref::<toml::de::Error>().is_some());
    }

    #[test]
    fn test_from_file_rejects_unknown_gate_name() {
        let path = std::env::temp_dir().join(format!(
            "quentangled-invalid-config-{}.toml",
            std::process::id()
        ));
        fs::write(&path, "[gates.cnnot]\ntime_ns = 5.0\n").unwrap();

        let result = SimulatorConfig::from_file(&path);
        fs::remove_file(path).unwrap();
        let error = result.unwrap_err();

        assert_eq!(
            error.downcast_ref::<ConfigError>(),
            Some(&ConfigError::UnknownGate("cnnot".to_string()))
        );
    }

    #[test]
    fn test_zero_chunk_size_is_rejected() {
        let config: SimulatorConfig = toml::from_str("[engine]\nchunk_size = 0\n").unwrap();
        assert_eq!(config.validate(), Err(ConfigError::InvalidChunkSize));
    }

    #[test]
    fn test_unknown_gate_name_is_rejected() {
        let config: SimulatorConfig = toml::from_str("[gates.cnnot]\ntime_ns = 5.0\n").unwrap();

        assert_eq!(
            config.validate(),
            Err(ConfigError::UnknownGate("cnnot".to_string()))
        );
    }

    #[test]
    fn test_gate_config_names_are_case_insensitive() {
        let config: SimulatorConfig = toml::from_str("[gates.X]\ntime_ns = 5.0\n").unwrap();

        assert!(config.validate().is_ok());
        assert_eq!(config.gate_duration_ns(&Gate::PauliX), 5.0);
    }

    #[test]
    fn test_duplicate_case_insensitive_gate_names_are_rejected() {
        let config: SimulatorConfig =
            toml::from_str("[gates.x]\ntime_ns = 5.0\n[gates.X]\ntime_ns = 6.0\n").unwrap();

        assert_eq!(
            config.validate(),
            Err(ConfigError::DuplicateGate("x".to_string()))
        );
    }

    #[test]
    fn test_gate_durations_must_be_finite_and_non_negative() {
        for duration in [-1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let config = config_with_gate_properties(GatePropsConfig {
                time_ns: Some(duration),
                rot_angle_err_rad: None,
                error_matrix: None,
            });

            assert_eq!(
                config.validate(),
                Err(ConfigError::InvalidGateDuration {
                    gate: "x".to_string()
                })
            );
        }
    }

    #[test]
    fn test_gate_rotation_errors_must_be_finite() {
        for error in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let config = config_with_gate_properties(GatePropsConfig {
                time_ns: None,
                rot_angle_err_rad: Some(error),
                error_matrix: None,
            });

            assert_eq!(
                config.validate(),
                Err(ConfigError::InvalidGateRotationError {
                    gate: "x".to_string()
                })
            );
        }
    }

    #[test]
    fn test_error_matrix_must_be_unitary() {
        let config: SimulatorConfig = toml::from_str(
            "[gates.x]\nerror_matrix = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]",
        )
        .unwrap();

        assert_eq!(
            config.validate(),
            Err(ConfigError::InvalidGateErrorMatrix {
                gate: "x".to_string()
            })
        );
    }

    #[test]
    fn test_error_matrix_must_be_finite() {
        let config = config_with_gate_properties(GatePropsConfig {
            time_ns: None,
            rot_angle_err_rad: None,
            error_matrix: Some(vec![[f64::NAN, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        });

        assert_eq!(
            config.validate(),
            Err(ConfigError::InvalidGateErrorMatrix {
                gate: "x".to_string()
            })
        );
    }

    #[test]
    fn test_config_error_display_messages() {
        let cases = [
            (
                ConfigError::InvalidChunkSize,
                "chunk_size must be greater than 0",
            ),
            (
                ConfigError::UnknownGate("foo".to_string()),
                "unknown gate configuration `foo`",
            ),
            (
                ConfigError::DuplicateGate("x".to_string()),
                "duplicate gate configuration `x`",
            ),
            (
                ConfigError::InvalidGateDuration {
                    gate: "x".to_string(),
                },
                "gate duration for `x` must be finite and non-negative",
            ),
            (
                ConfigError::InvalidGateRotationError {
                    gate: "x".to_string(),
                },
                "rotation error for `x` must be finite",
            ),
            (
                ConfigError::UnsupportedGateRotationError {
                    gate: "unitary".to_string(),
                },
                "rotation-angle errors are not supported for arbitrary gate `unitary`",
            ),
            (
                ConfigError::ConflictingGateErrors {
                    gate: "x".to_string(),
                },
                "gate `x` cannot configure both rot_angle_err_rad and error_matrix",
            ),
            (
                ConfigError::InvalidGateErrorMatrixSize {
                    gate: "cnot".to_string(),
                    expected: 16,
                    actual: 4,
                },
                "error matrix for `cnot` must contain 16 complex elements, got 4",
            ),
            (
                ConfigError::InvalidGateErrorMatrix {
                    gate: "x".to_string(),
                },
                "error matrix for `x` must be finite and unitary",
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn test_error_matrix_size_depends_on_gate_arity() {
        let config: SimulatorConfig = toml::from_str(
            "[gates.cnot]\nerror_matrix = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]",
        )
        .unwrap();

        assert_eq!(
            config.validate(),
            Err(ConfigError::InvalidGateErrorMatrixSize {
                gate: "cnot".to_string(),
                expected: 16,
                actual: 4
            })
        );
    }

    #[test]
    fn test_rotation_and_matrix_error_models_are_mutually_exclusive() {
        let config: SimulatorConfig = toml::from_str(
            "[gates.x]\nrot_angle_err_rad = 0.2\n\
             error_matrix = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]",
        )
        .unwrap();

        assert_eq!(
            config.validate(),
            Err(ConfigError::ConflictingGateErrors {
                gate: "x".to_string()
            })
        );
    }

    #[test]
    fn test_rotation_error_is_not_defined_for_arbitrary_custom_unitary() {
        let config: SimulatorConfig =
            toml::from_str("[gates.unitary]\nrot_angle_err_rad = 0.2\n").unwrap();

        assert_eq!(
            config.validate(),
            Err(ConfigError::UnsupportedGateRotationError {
                gate: "unitary".to_string()
            })
        );
    }

    fn config_with_gate_properties(properties: GatePropsConfig) -> SimulatorConfig {
        SimulatorConfig {
            gates: HashMap::from([("x".to_string(), properties)]),
            ..SimulatorConfig::default()
        }
    }
}
