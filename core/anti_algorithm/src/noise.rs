//! Multi-Domain Noise Generation System
//! 
//! This module implements the core noise generation capabilities across four domains:
//! - Deterministic: Structured, predictable failure patterns
//! - Fuzzy: Continuous-valued, context-aware perturbations  
//! - Quantum: Superposition-based parallel exploration
//! - Molecular: Thermal fluctuation-driven exploration
//! 
//! The system generates wrong solutions at rates exceeding 10^15 per second
//! to enable statistical emergence of correct solutions.

use crate::types::{AntiAlgorithmResult, AntiAlgorithmError, NoisePattern, FemtosecondTimestamp};
use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_distr::{Normal, Uniform, Gamma, Beta};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use rayon::prelude::*;

/// Trait defining noise generation capabilities
#[async_trait]
pub trait NoiseGenerator: Send + Sync {
    /// Generate a batch of noise patterns at femtosecond precision
    async fn generate_batch(
        &mut self,
        batch_size: usize,
        precision: f64,
    ) -> AntiAlgorithmResult<Vec<NoisePattern>>;
    
    /// Get the current generation rate (patterns per second)
    fn generation_rate(&self) -> f64;
    
    /// Get the noise domain type
    fn domain_type(&self) -> NoiseType;
    
    /// Adjust generation parameters based on performance feedback
    async fn adapt_parameters(&mut self, performance_feedback: &PerformanceFeedback);
    
    /// Reset the generator to initial state
    async fn reset(&mut self);
}

/// Types of noise domains available
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NoiseType {
    Deterministic,
    Fuzzy,
    Quantum,
    Molecular,
}

/// Performance feedback for adaptive noise generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceFeedback {
    /// Success rate of this noise type in generating good candidates
    pub success_rate: f64,
    
    /// Average fitness of candidates from this noise type
    pub average_fitness: f64,
    
    /// Computational cost per generated pattern
    pub cost_per_pattern: f64,
    
    /// Time since last successful generation
    pub time_since_success: f64,
    
    /// Statistical significance of generated patterns
    pub average_significance: f64,
}

/// Comprehensive noise portfolio managing all noise types
#[derive(Debug)]
pub struct NoisePortfolio {
    /// Individual noise generators by type
    generators: HashMap<NoiseType, Box<dyn NoiseGenerator>>,
    
    /// Mixing weights for composite noise generation
    mixing_weights: HashMap<NoiseType, f64>,
    
    /// Total generation rate across all generators
    total_rate: f64,
    
    /// Performance tracking per noise type
    performance_history: HashMap<NoiseType, Vec<PerformanceFeedback>>,
    
    /// Adaptive weighting enabled
    adaptive_weighting: bool,
}

impl NoisePortfolio {
    /// Create a portfolio with all noise types at optimal rates
    pub fn full_spectrum() -> AntiAlgorithmResult<Self> {
        let mut generators: HashMap<NoiseType, Box<dyn NoiseGenerator>> = HashMap::new();
        let mut mixing_weights = HashMap::new();
        
        // Initialize all noise generators
        generators.insert(NoiseType::Deterministic, Box::new(DeterministicNoise::new()?));
        generators.insert(NoiseType::Fuzzy, Box::new(FuzzyNoise::new()?));
        generators.insert(NoiseType::Quantum, Box::new(QuantumNoise::new()?));
        generators.insert(NoiseType::Molecular, Box::new(MolecularNoise::new()?));
        
        // Equal initial weights
        for noise_type in &[NoiseType::Deterministic, NoiseType::Fuzzy, NoiseType::Quantum, NoiseType::Molecular] {
            mixing_weights.insert(noise_type.clone(), 0.25);
        }
        
        Ok(Self {
            generators,
            mixing_weights,
            total_rate: crate::OPTIMAL_NOISE_RATE,
            performance_history: HashMap::new(),
            adaptive_weighting: true,
        })
    }
    
    /// Generate composite noise pattern using all generators
    pub async fn generate_composite_batch(
        &mut self,
        batch_size: usize,
        precision: f64,
    ) -> AntiAlgorithmResult<Vec<NoisePattern>> {
        let mut all_patterns = Vec::new();
        
        // Generate from each noise type based on mixing weights
        for (noise_type, weight) in &self.mixing_weights {
            let type_batch_size = (batch_size as f64 * weight) as usize;
            if type_batch_size > 0 {
                if let Some(generator) = self.generators.get_mut(noise_type) {
                    let mut patterns = generator.generate_batch(type_batch_size, precision).await?;
                    all_patterns.append(&mut patterns);
                }
            }
        }
        
        Ok(all_patterns)
    }
    
    /// Update mixing weights based on performance
    pub async fn adapt_weights(&mut self, feedback: &HashMap<NoiseType, PerformanceFeedback>) {
        if !self.adaptive_weighting {
            return;
        }
        
        let mut total_effectiveness = 0.0;
        let mut effectiveness_scores = HashMap::new();
        
        // Calculate effectiveness score for each noise type
        for (noise_type, perf) in feedback {
            let effectiveness = perf.success_rate * perf.average_fitness / perf.cost_per_pattern;
            effectiveness_scores.insert(noise_type.clone(), effectiveness);
            total_effectiveness += effectiveness;
        }
        
        // Update mixing weights based on relative effectiveness
        if total_effectiveness > 0.0 {
            for (noise_type, effectiveness) in effectiveness_scores {
                let new_weight = effectiveness / total_effectiveness;
                self.mixing_weights.insert(noise_type, new_weight);
            }
        }
        
        // Update individual generator parameters
        for (noise_type, perf) in feedback {
            if let Some(generator) = self.generators.get_mut(noise_type) {
                generator.adapt_parameters(perf).await;
            }
        }
    }
}

/// Deterministic noise generator for structured failure patterns
#[derive(Debug)]
pub struct DeterministicNoise {
    /// Random number generator with fixed seed for reproducibility
    rng: StdRng,
    
    /// Base amplitude for noise generation
    amplitude: f64,
    
    /// Frequency of oscillatory patterns
    frequency: f64,
    
    /// Phase offset for pattern variation
    phase: f64,
    
    /// Systematic bias in one direction
    systematic_bias: f64,
    
    /// Current generation rate
    generation_rate: f64,
    
    /// Parameter adaptation rate
    adaptation_rate: f64,
}

impl DeterministicNoise {
    /// Create new deterministic noise generator
    pub fn new() -> AntiAlgorithmResult<Self> {
        Ok(Self {
            rng: StdRng::seed_from_u64(42), // Fixed seed for deterministic patterns
            amplitude: 1.0,
            frequency: 1.0,
            phase: 0.0,
            systematic_bias: 0.1,
            generation_rate: crate::OPTIMAL_NOISE_RATE / 4.0, // 25% of total rate
            adaptation_rate: 0.01,
        })
    }
    
    /// Generate deterministic noise value at specific time
    fn generate_value(&mut self, t: f64) -> f64 {
        self.amplitude * (self.frequency * t + self.phase).sin() + self.systematic_bias
    }
}

#[async_trait]
impl NoiseGenerator for DeterministicNoise {
    async fn generate_batch(
        &mut self,
        batch_size: usize,
        precision: f64,
    ) -> AntiAlgorithmResult<Vec<NoisePattern>> {
        let mut patterns = Vec::with_capacity(batch_size);
        let time_step = precision;
        
        for i in 0..batch_size {
            let t = i as f64 * time_step;
            let noise_value = self.generate_value(t);
            
            let pattern = NoisePattern::Deterministic {
                amplitude: self.amplitude,
                frequency: self.frequency,
                phase: self.phase + t,
                systematic_bias: self.systematic_bias + noise_value * 0.1,
            };
            
            patterns.push(pattern);
        }
        
        Ok(patterns)
    }
    
    fn generation_rate(&self) -> f64 {
        self.generation_rate
    }
    
    fn domain_type(&self) -> NoiseType {
        NoiseType::Deterministic
    }
    
    async fn adapt_parameters(&mut self, feedback: &PerformanceFeedback) {
        // Increase amplitude if success rate is low
        if feedback.success_rate < 0.1 {
            self.amplitude *= 1.0 + self.adaptation_rate;
        } else if feedback.success_rate > 0.3 {
            self.amplitude *= 1.0 - self.adaptation_rate;
        }
        
        // Adjust frequency based on cost efficiency
        if feedback.cost_per_pattern > 1.0 {
            self.frequency *= 1.0 - self.adaptation_rate;
        }
        
        // Modify systematic bias based on average fitness
        if feedback.average_fitness < 0.5 {
            self.systematic_bias += self.adaptation_rate;
        }
    }
    
    async fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(42);
        self.amplitude = 1.0;
        self.frequency = 1.0;
        self.phase = 0.0;
        self.systematic_bias = 0.1;
    }
}

/// Fuzzy noise generator for continuous-valued perturbations
#[derive(Debug)]
pub struct FuzzyNoise {
    /// Random number generator
    rng: StdRng,
    
    /// Membership function parameters
    membership_centers: Vec<f64>,
    membership_widths: Vec<f64>,
    
    /// Context-aware weights
    context_weights: HashMap<String, f64>,
    
    /// Temporal noise amplitude
    temporal_amplitude: f64,
    
    /// Generation rate
    generation_rate: f64,
    
    /// Fuzzy logic rules
    fuzzy_rules: Vec<FuzzyRule>,
}

#[derive(Debug, Clone)]
struct FuzzyRule {
    antecedent: String,
    consequent: f64,
    weight: f64,
}

impl FuzzyNoise {
    /// Create new fuzzy noise generator
    pub fn new() -> AntiAlgorithmResult<Self> {
        let mut context_weights = HashMap::new();
        context_weights.insert("exploration".to_string(), 0.7);
        context_weights.insert("exploitation".to_string(), 0.3);
        context_weights.insert("convergence".to_string(), 0.5);
        
        let fuzzy_rules = vec![
            FuzzyRule {
                antecedent: "high_performance".to_string(),
                consequent: 0.8,
                weight: 1.0,
            },
            FuzzyRule {
                antecedent: "low_cost".to_string(),
                consequent: 0.6,
                weight: 0.8,
            },
        ];
        
        Ok(Self {
            rng: StdRng::from_entropy(),
            membership_centers: vec![0.0, 0.5, 1.0],
            membership_widths: vec![0.2, 0.3, 0.2],
            context_weights,
            temporal_amplitude: 1.0,
            generation_rate: crate::OPTIMAL_NOISE_RATE / 4.0,
            fuzzy_rules,
        })
    }
    
    /// Calculate fuzzy membership value
    fn membership(&self, x: f64, center: f64, width: f64) -> f64 {
        let diff = (x - center) / width;
        (-diff * diff).exp()
    }
    
    /// Generate fuzzy noise based on context
    fn generate_fuzzy_value(&mut self, context: &str) -> f64 {
        let base_noise: f64 = self.rng.gen_range(-1.0..1.0);
        let context_weight = self.context_weights.get(context).unwrap_or(&0.5);
        
        let membership_value = self.membership_centers.iter()
            .zip(self.membership_widths.iter())
            .map(|(center, width)| self.membership(base_noise, *center, *width))
            .fold(0.0, f64::max);
        
        base_noise * context_weight * membership_value * self.temporal_amplitude
    }
}

#[async_trait]
impl NoiseGenerator for FuzzyNoise {
    async fn generate_batch(
        &mut self,
        batch_size: usize,
        _precision: f64,
    ) -> AntiAlgorithmResult<Vec<NoisePattern>> {
        let mut patterns = Vec::with_capacity(batch_size);
        
        // Generate temporal noise array
        let temporal_noise: Array1<f64> = Array1::from_vec(
            (0..batch_size)
                .map(|_| self.generate_fuzzy_value("exploration"))
                .collect()
        );
        
        for i in 0..batch_size {
            let membership_function: Vec<f64> = self.membership_centers.iter()
                .zip(self.membership_widths.iter())
                .map(|(center, width)| self.membership(temporal_noise[i], *center, *width))
                .collect();
            
            let pattern = NoisePattern::Fuzzy {
                membership_function,
                temporal_noise: temporal_noise.slice(s![i..i+1]).to_owned(),
                context_weights: self.context_weights.clone(),
            };
            
            patterns.push(pattern);
        }
        
        Ok(patterns)
    }
    
    fn generation_rate(&self) -> f64 {
        self.generation_rate
    }
    
    fn domain_type(&self) -> NoiseType {
        NoiseType::Fuzzy
    }
    
    async fn adapt_parameters(&mut self, feedback: &PerformanceFeedback) {
        // Adjust temporal amplitude based on success rate
        if feedback.success_rate < 0.1 {
            self.temporal_amplitude *= 1.1;
        } else if feedback.success_rate > 0.4 {
            self.temporal_amplitude *= 0.9;
        }
        
        // Update context weights based on performance
        if feedback.average_fitness > 0.7 {
            if let Some(weight) = self.context_weights.get_mut("exploitation") {
                *weight = (*weight + 0.1).min(1.0);
            }
        }
    }
    
    async fn reset(&mut self) {
        self.temporal_amplitude = 1.0;
        self.membership_centers = vec![0.0, 0.5, 1.0];
        self.membership_widths = vec![0.2, 0.3, 0.2];
    }
}

/// Quantum noise generator using superposition-based exploration
#[derive(Debug)]
pub struct QuantumNoise {
    /// Random number generator
    rng: StdRng,
    
    /// Number of qubits for superposition
    qubit_count: usize,
    
    /// Decoherence time (femtoseconds)
    decoherence_time: f64,
    
    /// Entanglement strength
    entanglement_strength: f64,
    
    /// Generation rate
    generation_rate: f64,
    
    /// Quantum state evolution time
    evolution_time: f64,
}

impl QuantumNoise {
    /// Create new quantum noise generator
    pub fn new() -> AntiAlgorithmResult<Self> {
        Ok(Self {
            rng: StdRng::from_entropy(),
            qubit_count: 10,
            decoherence_time: 1000.0, // 1000 femtoseconds
            entanglement_strength: 0.5,
            generation_rate: crate::OPTIMAL_NOISE_RATE / 4.0,
            evolution_time: 0.0,
        })
    }
    
    /// Generate quantum superposition coefficients
    fn generate_superposition(&mut self) -> Vec<Complex64> {
        let state_count = 1 << self.qubit_count; // 2^n quantum states
        let mut coefficients = Vec::with_capacity(state_count);
        
        for _ in 0..state_count {
            let real = self.rng.gen_range(-1.0..1.0);
            let imag = self.rng.gen_range(-1.0..1.0);
            coefficients.push(Complex64::new(real, imag));
        }
        
        // Normalize the state vector
        let norm_squared: f64 = coefficients.iter().map(|c| c.norm_sqr()).sum();
        let norm = norm_squared.sqrt();
        
        if norm > 0.0 {
            for coeff in &mut coefficients {
                *coeff /= norm;
            }
        }
        
        coefficients
    }
    
    /// Generate entanglement matrix
    fn generate_entanglement_matrix(&mut self) -> Array2<Complex64> {
        let size = self.qubit_count;
        let mut matrix = Array2::zeros((size, size));
        
        for i in 0..size {
            for j in 0..size {
                let real = self.rng.gen_range(-1.0..1.0) * self.entanglement_strength;
                let imag = self.rng.gen_range(-1.0..1.0) * self.entanglement_strength;
                matrix[[i, j]] = Complex64::new(real, imag);
            }
        }
        
        matrix
    }
}

#[async_trait]
impl NoiseGenerator for QuantumNoise {
    async fn generate_batch(
        &mut self,
        batch_size: usize,
        precision: f64,
    ) -> AntiAlgorithmResult<Vec<NoisePattern>> {
        // Check decoherence constraints
        if precision > self.decoherence_time {
            return Err(AntiAlgorithmError::QuantumDecoherence {
                decoherence: precision,
                required: self.decoherence_time,
            });
        }
        
        let mut patterns = Vec::with_capacity(batch_size);
        
        for _ in 0..batch_size {
            let superposition_coefficients = self.generate_superposition();
            let entanglement_matrix = self.generate_entanglement_matrix();
            
            let pattern = NoisePattern::Quantum {
                superposition_coefficients,
                entanglement_matrix,
                decoherence_time: self.decoherence_time,
            };
            
            patterns.push(pattern);
            self.evolution_time += precision;
        }
        
        Ok(patterns)
    }
    
    fn generation_rate(&self) -> f64 {
        self.generation_rate
    }
    
    fn domain_type(&self) -> NoiseType {
        NoiseType::Quantum
    }
    
    async fn adapt_parameters(&mut self, feedback: &PerformanceFeedback) {
        // Adjust entanglement strength based on performance
        if feedback.average_significance < 1.0 {
            self.entanglement_strength = (self.entanglement_strength + 0.1).min(1.0);
        } else if feedback.average_significance > 3.0 {
            self.entanglement_strength = (self.entanglement_strength - 0.1).max(0.1);
        }
        
        // Modify qubit count based on cost efficiency
        if feedback.cost_per_pattern > 2.0 && self.qubit_count > 5 {
            self.qubit_count -= 1;
        } else if feedback.cost_per_pattern < 0.5 && self.qubit_count < 15 {
            self.qubit_count += 1;
        }
    }
    
    async fn reset(&mut self) {
        self.qubit_count = 10;
        self.entanglement_strength = 0.5;
        self.evolution_time = 0.0;
    }
}

/// Molecular noise generator driven by thermal fluctuations
#[derive(Debug)]
pub struct MolecularNoise {
    /// Random number generator
    rng: StdRng,
    
    /// Thermal energy (kT units)
    thermal_energy: f64,
    
    /// Boltzmann constant (scaled for computation)
    boltzmann_constant: f64,
    
    /// Temperature (computational units)
    temperature: f64,
    
    /// Number of conformational states
    state_count: usize,
    
    /// Generation rate
    generation_rate: f64,
    
    /// Molecular dynamics time step
    time_step: f64,
}

impl MolecularNoise {
    /// Create new molecular noise generator
    pub fn new() -> AntiAlgorithmResult<Self> {
        let temperature = 300.0; // Room temperature equivalent
        let boltzmann_constant = 1.0; // Scaled for computational convenience
        let thermal_energy = boltzmann_constant * temperature;
        
        Ok(Self {
            rng: StdRng::from_entropy(),
            thermal_energy,
            boltzmann_constant,
            temperature,
            state_count: 100,
            generation_rate: crate::OPTIMAL_NOISE_RATE / 4.0,
            time_step: 1e-12, // Picosecond time steps
        })
    }
    
    /// Generate conformational states
    fn generate_conformational_states(&mut self) -> Vec<Vec<f64>> {
        let mut states = Vec::with_capacity(self.state_count);
        
        for _ in 0..self.state_count {
            let state: Vec<f64> = (0..10) // 10 degrees of freedom per state
                .map(|_| self.rng.gen_range(-1.0..1.0))
                .collect();
            states.push(state);
        }
        
        states
    }
    
    /// Calculate Boltzmann transition probabilities
    fn calculate_transition_probabilities(&mut self, states: &[Vec<f64>]) -> Array2<f64> {
        let n = states.len();
        let mut matrix = Array2::zeros((n, n));
        
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // Calculate energy difference (simplified)
                    let energy_diff: f64 = states[i].iter().zip(states[j].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    
                    // Boltzmann probability
                    let prob = (-energy_diff / self.thermal_energy).exp();
                    matrix[[i, j]] = prob;
                }
            }
        }
        
        // Normalize rows to make it a proper transition matrix
        for i in 0..n {
            let row_sum: f64 = matrix.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n {
                    matrix[[i, j]] /= row_sum;
                }
            }
        }
        
        matrix
    }
}

#[async_trait]
impl NoiseGenerator for MolecularNoise {
    async fn generate_batch(
        &mut self,
        batch_size: usize,
        _precision: f64,
    ) -> AntiAlgorithmResult<Vec<NoisePattern>> {
        // Check thermal equilibrium
        let min_thermal_energy = 0.1;
        if self.thermal_energy < min_thermal_energy {
            return Err(AntiAlgorithmError::ThermalEquilibrium {
                energy: self.thermal_energy,
                threshold: min_thermal_energy,
            });
        }
        
        let mut patterns = Vec::with_capacity(batch_size);
        
        for _ in 0..batch_size {
            let conformational_states = self.generate_conformational_states();
            let transition_probabilities = self.calculate_transition_probabilities(&conformational_states);
            
            let pattern = NoisePattern::Molecular {
                thermal_energy: self.thermal_energy,
                boltzmann_factor: (-1.0 / self.thermal_energy).exp(),
                conformational_states,
                transition_probabilities,
            };
            
            patterns.push(pattern);
        }
        
        Ok(patterns)
    }
    
    fn generation_rate(&self) -> f64 {
        self.generation_rate
    }
    
    fn domain_type(&self) -> NoiseType {
        NoiseType::Molecular
    }
    
    async fn adapt_parameters(&mut self, feedback: &PerformanceFeedback) {
        // Adjust temperature based on exploration success
        if feedback.success_rate < 0.05 {
            self.temperature *= 1.1; // Increase temperature for more exploration
        } else if feedback.success_rate > 0.3 {
            self.temperature *= 0.95; // Cool down for more exploitation
        }
        
        self.thermal_energy = self.boltzmann_constant * self.temperature;
        
        // Modify state count based on computational cost
        if feedback.cost_per_pattern > 1.5 && self.state_count > 50 {
            self.state_count -= 10;
        } else if feedback.cost_per_pattern < 0.8 && self.state_count < 200 {
            self.state_count += 10;
        }
    }
    
    async fn reset(&mut self) {
        self.temperature = 300.0;
        self.thermal_energy = self.boltzmann_constant * self.temperature;
        self.state_count = 100;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_deterministic_noise_generation() {
        let mut generator = DeterministicNoise::new().unwrap();
        let patterns = generator.generate_batch(10, 1e-15).await.unwrap();
        
        assert_eq!(patterns.len(), 10);
        
        for pattern in patterns {
            match pattern {
                NoisePattern::Deterministic { amplitude, frequency, phase, systematic_bias } => {
                    assert!(amplitude > 0.0);
                    assert!(frequency > 0.0);
                },
                _ => panic!("Expected deterministic pattern"),
            }
        }
    }
    
    #[tokio::test]
    async fn test_noise_portfolio_creation() {
        let portfolio = NoisePortfolio::full_spectrum().unwrap();
        assert_eq!(portfolio.generators.len(), 4);
        assert!(portfolio.generators.contains_key(&NoiseType::Deterministic));
        assert!(portfolio.generators.contains_key(&NoiseType::Fuzzy));
        assert!(portfolio.generators.contains_key(&NoiseType::Quantum));
        assert!(portfolio.generators.contains_key(&NoiseType::Molecular));
    }
    
    #[tokio::test]
    async fn test_fuzzy_noise_generation() {
        let mut generator = FuzzyNoise::new().unwrap();
        let patterns = generator.generate_batch(5, 1e-15).await.unwrap();
        
        assert_eq!(patterns.len(), 5);
        
        for pattern in patterns {
            match pattern {
                NoisePattern::Fuzzy { membership_function, temporal_noise, context_weights } => {
                    assert!(!membership_function.is_empty());
                    assert!(temporal_noise.len() > 0);
                    assert!(!context_weights.is_empty());
                },
                _ => panic!("Expected fuzzy pattern"),
            }
        }
    }
    
    #[tokio::test]
    async fn test_quantum_noise_generation() {
        let mut generator = QuantumNoise::new().unwrap();
        let patterns = generator.generate_batch(3, 100.0).await.unwrap(); // Within decoherence time
        
        assert_eq!(patterns.len(), 3);
        
        for pattern in patterns {
            match pattern {
                NoisePattern::Quantum { superposition_coefficients, entanglement_matrix, decoherence_time } => {
                    assert!(!superposition_coefficients.is_empty());
                    assert!(entanglement_matrix.len() > 0);
                    assert!(decoherence_time > 0.0);
                },
                _ => panic!("Expected quantum pattern"),
            }
        }
    }
    
    #[tokio::test]
    async fn test_molecular_noise_generation() {
        let mut generator = MolecularNoise::new().unwrap();
        let patterns = generator.generate_batch(2, 1e-15).await.unwrap();
        
        assert_eq!(patterns.len(), 2);
        
        for pattern in patterns {
            match pattern {
                NoisePattern::Molecular { thermal_energy, boltzmann_factor, conformational_states, transition_probabilities } => {
                    assert!(thermal_energy > 0.0);
                    assert!(boltzmann_factor > 0.0);
                    assert!(!conformational_states.is_empty());
                    assert!(transition_probabilities.len() > 0);
                },
                _ => panic!("Expected molecular pattern"),
            }
        }
    }
    
    #[test]
    fn test_performance_feedback() {
        let feedback = PerformanceFeedback {
            success_rate: 0.15,
            average_fitness: 0.8,
            cost_per_pattern: 0.5,
            time_since_success: 100.0,
            average_significance: 2.5,
        };
        
        assert_eq!(feedback.success_rate, 0.15);
        assert_eq!(feedback.average_fitness, 0.8);
    }
} 