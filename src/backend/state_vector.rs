//! The state vector simulation engine using a double-buffered state vector.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::core::config::EngineConfig;

/// Represents the quantum state of the system using a state vector.
///
/// This structure uses a double-buffer architecture. During gate application,
/// it reads coefficients from `amplitudes`, calculates the new state, writes
/// the results to `buffer`, and then swaps the two vectors. This guarantees
/// zero new memory allocations during circuit execution.
#[derive(Debug, Clone)]
pub(crate) struct StateVector {
    /// The number of qubits in the simulated system.
    pub(crate) num_qubits: usize,
    /// The active probability amplitudes of the quantum states.
    pub(crate) amplitudes: Vec<Complex64>,
    /// A pre-allocated secondary buffer used for gate transformations.
    buffer: Vec<Complex64>,
}

impl StateVector {
    /// Initialises a new state vector in the |0...0> state.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits to simulate.
    ///
    /// State-vector storage is an internal backend detail. Use [`crate::Simulator`] and
    /// [`crate::SimulationResult`] from the public API.
    ///
    /// ```compile_fail
    /// use quentangled::backend::state_vector::StateVector;
    /// ```
    pub(crate) fn new(num_qubits: usize) -> Self {
        // A system of n qubits has 2^n possible states
        let len = 1 << num_qubits;

        let mut amplitudes = vec![Complex64::new(0.0, 0.0); len];
        // Initialise to state |0...0>
        amplitudes[0] = Complex64::new(1.0, 0.0);

        Self {
            num_qubits,
            amplitudes,
            buffer: vec![Complex64::new(0.0, 0.0); len],
        }
    }

    /// Applies a 2x2 unitary matrix representing a single-qubit gate.
    ///
    /// # Arguments
    ///
    /// * `target` - The index of the qubit being operated on.
    /// * `matrix` - The unitary matrix flattened as `[m00, m01, m10, m11]`.
    /// * `engine` - The engine settings used for parallel execution.
    pub(crate) fn apply_single_qubit_gate(
        &mut self,
        target: usize,
        matrix: [Complex64; 4],
        engine: EngineConfig,
    ) {
        let [m00, m01, m10, m11] = matrix;

        // Iterating over half the states is enough since single qubit gates work on pairs of states.
        let half_len = 1 << (self.num_qubits - 1);

        if self.amplitudes.len() >= engine.parallel_threshold {
            let target_mask = 1 << target;
            self.buffer
                .par_chunks_mut(engine.chunk_size)
                .enumerate()
                .for_each(|(chunk_id, chunk)| {
                    // Calculate the starting index of this specific chunk
                    let base_idx = chunk_id * engine.chunk_size;

                    // Process the chunk sequentially.
                    for (local_idx, buf_val) in chunk.iter_mut().enumerate() {
                        let global_idx = base_idx + local_idx;

                        let is_target_one = (global_idx & target_mask) != 0;
                        let idx0 = global_idx & !target_mask;
                        let idx1 = global_idx | target_mask;

                        let amp0 = self.amplitudes[idx0];
                        let amp1 = self.amplitudes[idx1];

                        if is_target_one {
                            *buf_val = m10 * amp0 + m11 * amp1;
                        } else {
                            *buf_val = m00 * amp0 + m01 * amp1;
                        }
                    }
                });
        } else {
            let target_mask = (1 << target) - 1;
            for i in 0..half_len {
                // Find the pairs of states that are affected
                let idx0 = (i & target_mask) | ((i & !target_mask) << 1);
                let idx1 = idx0 | (1 << target);

                // Input state
                let amp0 = self.amplitudes[idx0];
                let amp1 = self.amplitudes[idx1];

                // Apply the gate and store results in the buffer vector
                self.buffer[idx0] = m00 * amp0 + m01 * amp1;
                self.buffer[idx1] = m10 * amp0 + m11 * amp1;
            }
        }

        // Swap the active vector and the buffer pointers
        std::mem::swap(&mut self.amplitudes, &mut self.buffer);
    }

    /// Applies a single-qubit matrix to the target when the control is |1>.
    ///
    /// Amplitude pairs for which the control is |0> are left unchanged.
    ///
    /// # Arguments
    ///
    /// * `control` - The index of the controlling qubit.
    /// * `target` - The index of the qubit to which the matrix is applied.
    /// * `matrix` - The matrix flattened as `[m00, m01, m10, m11]`.
    /// * `engine` - The engine settings used for parallel execution.
    ///
    /// # Panics
    ///
    /// Panics if `control` and `target` are the same index.
    pub(crate) fn apply_controlled_gate(
        &mut self,
        control: usize,
        target: usize,
        matrix: [Complex64; 4],
        engine: EngineConfig,
    ) {
        assert_ne!(
            control, target,
            "Control and target qubits cannot be the same"
        );

        let [m00, m01, m10, m11] = matrix;
        let control_mask = 1 << control;
        let target_mask = 1 << target;

        if self.amplitudes.len() >= engine.parallel_threshold {
            self.buffer
                .par_chunks_mut(engine.chunk_size)
                .enumerate()
                .for_each(|(chunk_id, chunk)| {
                    let base_idx = chunk_id * engine.chunk_size;

                    for (local_idx, buf_val) in chunk.iter_mut().enumerate() {
                        let idx = base_idx + local_idx;
                        if idx & control_mask == 0 {
                            *buf_val = self.amplitudes[idx];
                            continue;
                        }

                        let partner_idx = idx ^ target_mask;
                        if idx & target_mask == 0 {
                            *buf_val =
                                m00 * self.amplitudes[idx] + m01 * self.amplitudes[partner_idx];
                        } else {
                            *buf_val =
                                m10 * self.amplitudes[partner_idx] + m11 * self.amplitudes[idx];
                        }
                    }
                });
        } else {
            self.buffer.copy_from_slice(&self.amplitudes);

            let half_len = 1 << (self.num_qubits - 1);
            let mask = (1 << target) - 1;

            for i in 0..half_len {
                let idx0 = (i & mask) | ((i & !mask) << 1);
                let idx1 = idx0 | target_mask;

                if (idx0 & control_mask) != 0 {
                    let amp0 = self.amplitudes[idx0];
                    let amp1 = self.amplitudes[idx1];
                    self.buffer[idx0] = m00 * amp0 + m01 * amp1;
                    self.buffer[idx1] = m10 * amp0 + m11 * amp1;
                }
            }
        }

        // Swap the pointers
        std::mem::swap(&mut self.amplitudes, &mut self.buffer);
    }

    /// Applies a matrix to the `|01⟩`/`|10⟩` subspace of two qubits.
    ///
    /// # Arguments
    ///
    /// * `q1` - The index of the first qubit.
    /// * `q2` - The index of the second qubit.
    /// * `matrix` - The matrix flattened as `[m00, m01, m10, m11]`.
    /// * `engine` - The engine settings used for parallel execution.
    pub(crate) fn apply_swap(
        &mut self,
        q1: usize,
        q2: usize,
        matrix: [Complex64; 4],
        engine: EngineConfig,
    ) {
        assert_ne!(q1, q2, "Cannot swap a qubit with itself");

        let [m00, m01, m10, m11] = matrix;
        let q1_mask = 1 << q1;
        let q2_mask = 1 << q2;
        let swap_mask = q1_mask | q2_mask;

        if self.amplitudes.len() >= engine.parallel_threshold {
            self.buffer
                .par_chunks_mut(engine.chunk_size)
                .enumerate()
                .for_each(|(chunk_id, chunk)| {
                    let base_idx = chunk_id * engine.chunk_size;

                    for (local_idx, buf_val) in chunk.iter_mut().enumerate() {
                        let idx = base_idx + local_idx;
                        let q1_is_one = idx & q1_mask != 0;
                        let q2_is_one = idx & q2_mask != 0;

                        if q1_is_one == q2_is_one {
                            *buf_val = self.amplitudes[idx];
                            continue;
                        }

                        let partner_idx = idx ^ swap_mask;
                        if !q1_is_one {
                            *buf_val =
                                m00 * self.amplitudes[idx] + m01 * self.amplitudes[partner_idx];
                        } else {
                            *buf_val =
                                m10 * self.amplitudes[partner_idx] + m11 * self.amplitudes[idx];
                        }
                    }
                });
        } else {
            self.buffer.copy_from_slice(&self.amplitudes);
            let n = self.amplitudes.len();

            for i in 0..n {
                let bit1 = (i >> q1) & 1;
                let bit2 = (i >> q2) & 1;

                if bit1 == 0 && bit2 == 1 {
                    let j = i ^ swap_mask;
                    let amp01 = self.amplitudes[i];
                    let amp10 = self.amplitudes[j];
                    self.buffer[i] = m00 * amp01 + m01 * amp10;
                    self.buffer[j] = m10 * amp01 + m11 * amp10;
                }
            }
        }

        std::mem::swap(&mut self.amplitudes, &mut self.buffer);
    }

    /// Applies a full 4x4 unitary to two qubits in operand order.
    ///
    /// The matrix basis is `|q1 q2>`: `q1` is the most-significant local bit and
    /// `q2` is the least-significant local bit.
    pub(crate) fn apply_two_qubit_gate(
        &mut self,
        q1: usize,
        q2: usize,
        matrix: [Complex64; 16],
        engine: EngineConfig,
    ) {
        assert_ne!(q1, q2, "Two-qubit gate operands must be distinct");

        let q1_mask = 1 << q1;
        let q2_mask = 1 << q2;
        let operands_mask = q1_mask | q2_mask;

        if self.amplitudes.len() >= engine.parallel_threshold {
            self.buffer
                .par_chunks_mut(engine.chunk_size)
                .enumerate()
                .for_each(|(chunk_id, chunk)| {
                    let base_idx = chunk_id * engine.chunk_size;

                    for (local_idx, output) in chunk.iter_mut().enumerate() {
                        let output_idx = base_idx + local_idx;
                        let row = usize::from(output_idx & q1_mask != 0) * 2
                            + usize::from(output_idx & q2_mask != 0);
                        let base = output_idx & !operands_mask;
                        let mut amplitude = Complex64::new(0.0, 0.0);

                        for column in 0..4 {
                            let input_idx = base
                                | (if column & 0b10 != 0 { q1_mask } else { 0 })
                                | (if column & 0b01 != 0 { q2_mask } else { 0 });
                            amplitude += matrix[row * 4 + column] * self.amplitudes[input_idx];
                        }

                        *output = amplitude;
                    }
                });
        } else {
            for output_idx in 0..self.amplitudes.len() {
                let row = usize::from(output_idx & q1_mask != 0) * 2
                    + usize::from(output_idx & q2_mask != 0);
                let base = output_idx & !operands_mask;
                let mut amplitude = Complex64::new(0.0, 0.0);

                for column in 0..4 {
                    let input_idx = base
                        | (if column & 0b10 != 0 { q1_mask } else { 0 })
                        | (if column & 0b01 != 0 { q2_mask } else { 0 });
                    amplitude += matrix[row * 4 + column] * self.amplitudes[input_idx];
                }

                self.buffer[output_idx] = amplitude;
            }
        }

        std::mem::swap(&mut self.amplitudes, &mut self.buffer);
    }

    /// Computes the measurement probabilities from the current amplitudes.
    ///
    /// # Returns
    ///
    /// Returns the probabilities for each state.
    pub(crate) fn into_parts(self) -> (usize, Vec<Complex64>) {
        (self.num_qubits, self.amplitudes)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the state vector and its mathematical operations.

    use super::*;

    /// A helper function to compare floating-point complex numbers safely.
    fn approx_eq_complex(a: Complex64, b: Complex64) -> bool {
        (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10
    }

    fn assert_amplitudes_approx_eq(left: &[Complex64], right: &[Complex64]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert!(approx_eq_complex(*left, *right), "{left:?} != {right:?}");
        }
    }

    #[test]
    fn test_initialisation_initialises_to_ground_state() {
        let state = StateVector::new(3); // 3 qubits = 8 states

        assert_eq!(state.amplitudes.len(), 8);
        assert_eq!(state.buffer.len(), 8);

        // State |000> should have 100% probability
        assert!(approx_eq_complex(
            state.amplitudes[0],
            Complex64::new(1.0, 0.0)
        ));

        // All other states should be 0
        for i in 1..8 {
            assert!(approx_eq_complex(
                state.amplitudes[i],
                Complex64::new(0.0, 0.0)
            ));
        }
    }

    #[test]
    fn test_pauli_x_gate_qubit_0_flips_qubit_0() {
        let mut state = StateVector::new(2); // 2 qubits = 4 states: |00>, |01>, |10>, |11>

        let pauli_x = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        // Apply X gate to Qubit 0.
        // This should flip |00> (Index 0) to |01> (Index 1)
        state.apply_single_qubit_gate(0, pauli_x, EngineConfig::default());

        assert!(approx_eq_complex(
            state.amplitudes[0],
            Complex64::new(0.0, 0.0)
        )); // |00> is empty
        assert!(approx_eq_complex(
            state.amplitudes[1],
            Complex64::new(1.0, 0.0)
        )); // |01> is active
    }

    #[test]
    fn test_pauli_x_gate_qubit_0_flips_qubit_0_parallel() {
        let mut state = StateVector::new(2); // 2 qubits = 4 states: |00>, |01>, |10>, |11>

        let pauli_x = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        // Apply X gate to Qubit 0.
        // This should flip |00> (Index 0) to |01> (Index 1)
        state.apply_single_qubit_gate(
            0,
            pauli_x,
            EngineConfig {
                parallel_threshold: 0,
                chunk_size: 1,
            },
        );

        assert!(approx_eq_complex(
            state.amplitudes[0],
            Complex64::new(0.0, 0.0)
        )); // |00> is empty
        assert!(approx_eq_complex(
            state.amplitudes[1],
            Complex64::new(1.0, 0.0)
        )); // |01> is active
    }

    #[test]
    fn test_cnot_gate_with_active_control_qubit_flips_target_qubit() {
        let mut state = StateVector::new(2);

        // Force the state to |10> (Qubit 0 is 0, Qubit 1 is 1)
        state.amplitudes[0] = Complex64::new(0.0, 0.0);
        state.amplitudes[2] = Complex64::new(1.0, 0.0); // Index 2 is binary 10

        // Apply CNOT: Control = Qubit 1, Target = Qubit 0
        // Because Qubit 1 is 1, Qubit 0 should flip. Result should be |11> (Index 3)
        state.apply_controlled_gate(
            1,
            0,
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            EngineConfig::default(),
        );

        assert!(approx_eq_complex(
            state.amplitudes[0],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[1],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[2],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[3],
            Complex64::new(1.0, 0.0)
        ));
    }

    #[test]
    fn test_cnot_gate_with_inactive_control_qubit_does_not_flip_target_qubit() {
        let mut state = StateVector::new(2);

        // Quantum state is |00>
        state.amplitudes[0] = Complex64::new(1.0, 0.0);

        // Apply CNOT: Control = Qubit 1, Target = Qubit 0
        state.apply_controlled_gate(
            1,
            0,
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            EngineConfig::default(),
        );

        assert!(approx_eq_complex(
            state.amplitudes[0],
            Complex64::new(1.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[1],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[2],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[3],
            Complex64::new(0.0, 0.0)
        ));
    }

    #[test]
    fn test_cz_gate_with_active_control_qubit_adds_phase_to_target_qubit() {
        let mut state = StateVector::new(2);

        // Force the state to |11> (Qubit 0 is 1, Qubit 1 is 1)
        state.amplitudes[0] = Complex64::new(0.0, 0.0);
        state.amplitudes[3] = Complex64::new(1.0, 0.0); // Index 3 is binary 11

        // Apply CZ: Control = Qubit 1, Target = Qubit 0
        state.apply_controlled_gate(
            1,
            0,
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
            EngineConfig::default(),
        );

        assert!(approx_eq_complex(
            state.amplitudes[0],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[1],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[2],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[3],
            Complex64::new(-1.0, 0.0)
        ));
    }

    #[test]
    fn test_cz_gate_with_inactive_target_qubit_does_not_add_phase_to_target_qubit() {
        let mut state = StateVector::new(2);

        // Force the state to |10> (Qubit 0 is 0, Qubit 1 is 1)
        state.amplitudes[0] = Complex64::new(0.0, 0.0);
        state.amplitudes[2] = Complex64::new(1.0, 0.0); // Index 2 is binary 10

        // Apply CZ: Control = Qubit 1, Target = Qubit 0
        state.apply_controlled_gate(
            1,
            0,
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
            EngineConfig::default(),
        );

        assert!(approx_eq_complex(
            state.amplitudes[0],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[1],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[2],
            Complex64::new(1.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[3],
            Complex64::new(0.0, 0.0)
        ));
    }

    #[test]
    fn test_cz_gate_with_inactive_control_qubit_does_not_add_phase_to_target_qubit() {
        let mut state = StateVector::new(2);

        // Force the state to |01> (Qubit 0 is 1, Qubit 1 is 0)
        state.amplitudes[0] = Complex64::new(0.0, 0.0);
        state.amplitudes[1] = Complex64::new(1.0, 0.0); // Index 1 is binary 01

        // Apply CZ: Control = Qubit 1, Target = Qubit
        state.apply_controlled_gate(
            1,
            0,
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
            EngineConfig::default(),
        );

        assert!(approx_eq_complex(
            state.amplitudes[0],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[1],
            Complex64::new(1.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[2],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[3],
            Complex64::new(0.0, 0.0)
        ));
    }

    #[test]
    fn test_controlled_gate_parallel_matches_sequential() {
        let matrix = [
            Complex64::new(0.5, 0.25),
            Complex64::new(-0.75, 0.5),
            Complex64::new(0.25, -0.5),
            Complex64::new(0.75, 0.125),
        ];
        let mut sequential = StateVector::new(4);
        for (i, amplitude) in sequential.amplitudes.iter_mut().enumerate() {
            *amplitude = Complex64::new(i as f64 * 0.125, i as f64 * -0.0625);
        }
        let mut parallel = sequential.clone();

        sequential.apply_controlled_gate(
            3,
            1,
            matrix,
            EngineConfig {
                parallel_threshold: usize::MAX,
                chunk_size: 3,
            },
        );
        parallel.apply_controlled_gate(
            3,
            1,
            matrix,
            EngineConfig {
                parallel_threshold: 0,
                chunk_size: 3,
            },
        );

        assert_amplitudes_approx_eq(&sequential.amplitudes, &parallel.amplitudes);
    }

    #[test]
    fn test_two_qubit_matrix_parallel_matches_sequential() {
        let swap = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let mut sequential = StateVector::new(2);
        sequential.amplitudes.fill(Complex64::new(0.0, 0.0));
        sequential.amplitudes[2] = Complex64::new(1.0, 0.0);
        let mut parallel = sequential.clone();

        sequential.apply_two_qubit_gate(
            1,
            0,
            swap,
            EngineConfig {
                parallel_threshold: usize::MAX,
                chunk_size: 2,
            },
        );
        parallel.apply_two_qubit_gate(
            1,
            0,
            swap,
            EngineConfig {
                parallel_threshold: 0,
                chunk_size: 1,
            },
        );

        assert_amplitudes_approx_eq(&sequential.amplitudes, &parallel.amplitudes);
        assert!(approx_eq_complex(
            sequential.amplitudes[1],
            Complex64::new(1.0, 0.0)
        ));
    }

    #[test]
    fn test_swap_gate_swaps_qubits() {
        let mut state = StateVector::new(2);

        // Force the state to |10> (Qubit 0 is 0, Qubit 1 is 1)
        state.amplitudes[0] = Complex64::new(0.0, 0.0);
        state.amplitudes[1] = Complex64::new(1.0, 0.0); // Index 1 is binary 10

        // Apply CZ: Control = Qubit 1, Target = Qubit 0
        state.apply_swap(
            1,
            0,
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            EngineConfig::default(),
        );

        assert!(approx_eq_complex(
            state.amplitudes[0],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[1],
            Complex64::new(0.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[2],
            Complex64::new(1.0, 0.0)
        ));
        assert!(approx_eq_complex(
            state.amplitudes[3],
            Complex64::new(0.0, 0.0)
        ));
    }

    #[test]
    fn test_swap_gate_parallel_matches_sequential() {
        let matrix = [
            Complex64::new(0.5, 0.25),
            Complex64::new(-0.75, 0.5),
            Complex64::new(0.25, -0.5),
            Complex64::new(0.75, 0.125),
        ];
        let mut sequential = StateVector::new(4);
        for (i, amplitude) in sequential.amplitudes.iter_mut().enumerate() {
            *amplitude = Complex64::new(i as f64 * 0.125, i as f64 * -0.0625);
        }
        let mut parallel = sequential.clone();

        sequential.apply_swap(
            3,
            0,
            matrix,
            EngineConfig {
                parallel_threshold: usize::MAX,
                chunk_size: 3,
            },
        );
        parallel.apply_swap(
            3,
            0,
            matrix,
            EngineConfig {
                parallel_threshold: 0,
                chunk_size: 3,
            },
        );

        assert_amplitudes_approx_eq(&sequential.amplitudes, &parallel.amplitudes);
    }
}
