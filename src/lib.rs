//! Quentangled is a Rust library for constructing and simulating quantum circuits.
//!
//! Build circuits with [`Circuit`], configure an instance with [`SimulatorConfig`],
//! and execute it with [`Simulator`]. The resulting [`SimulationResult`] provides
//! amplitudes and probabilities without exposing the internal state-vector backend.

mod backend;
mod core;

pub use backend::executor::{SimulationResult, Simulator};
pub use core::circuit::{Circuit, CircuitError};
pub use core::config::{ConfigError, EngineConfig, GatePropsConfig, SimulatorConfig};
pub use num_complex::Complex64;
