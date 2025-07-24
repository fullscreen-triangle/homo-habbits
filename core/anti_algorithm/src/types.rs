//! Core types and data structures for the Anti-Algorithm computational engine
//! 
//! This module defines the fundamental data types used throughout the system,
//! including error types, solution representations, noise patterns, and 
//! statistical signatures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Result type for Anti-Algorithm operations
pub type AntiAlgorithmResult<T> = Result<T, AntiAlgorithmError>;

/// Comprehensive error types for Anti-Algorithm operations
#[derive(Debug, thiserror::Error)]
pub enum AntiAlgorithmError {
    #[error("Noise generation rate too low: {rate:.2e} < {min:.2e} required")]
    InsufficientNoiseRate { rate: f64, min: f64 },
    
    #[error("Statistical convergence failed after {attempts} attempts")]
    ConvergenceFailure { attempts: u64 },
    
    #[error("Femtosecond precision not achievable: {achieved:.2e} > {required:.2e}")]
    TemporalPrecisionError { achieved: f64, required: f64 },
    
    #[error("Noise portfolio incomplete: missing {missing_types:?}")]
    IncompleteNoisePortfolio { missing_types: Vec<String> },
    
    #[error("Solution space overflow: {candidates} candidates exceed capacity {max_capacity}")]
    SolutionSpaceOverflow { candidates: usize, max_capacity: usize },
    
    #[error("Pattern recognition failed: {reason}")]
    PatternRecognitionFailure { reason: String },
    
    #[error("Quantum coherence lost: decoherence time {decoherence:.2e} < required {required:.2e}")]
    QuantumDecoherence { decoherence: f64, required: f64 },
    
    #[error("Molecular thermal equilibrium: insufficient energy {energy:.2e} < {threshold:.2e}")]
    ThermalEquilibrium { energy: f64, threshold: f64 },
    
    #[error("Statistical anomaly detection failed: {details}")]
    AnomalyDetectionFailure { details: String },
    
    #[error("Zero-infinite computation binary collapse: {state}")]
    ComputationBinaryCollapse { state: String },
    
    #[error("IO operation failed: {source}")]
    IoError { #[from] source: std::io::Error },
    
    #[error("Serialization failed: {source}")]
    SerializationError { #[from] source: serde_json::Error },
}

/// Represents a candidate solution generated through the anti-algorithm process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionCandidate {
    /// Unique identifier for the solution candidate
    pub id: u64,
    
    /// The actual solution data (problem-dependent representation)
    pub data: Vec<f64>,
    
    /// Performance score of this candidate (higher = better)
    pub performance: f64,
    
    /// Cost of generating this candidate (lower = better)
    pub generation_cost: f64,
    
    /// Fitness score: performance / generation_cost
    pub fitness: f64,
    
    /// The noise pattern that generated this candidate
    pub generating_pattern: NoisePattern,
    
    /// Timestamp when this candidate was generated (femtosecond precision)
    pub timestamp: FemtosecondTimestamp,
    
    /// Statistical significance of this candidate
    pub significance: f64,
    
    /// Confidence interval for this solution
    pub confidence_interval: (f64, f64),
    
    /// Whether this candidate shows signs of emergence
    pub is_emergent: bool,
}

impl SolutionCandidate {
    /// Create a new solution candidate
    pub fn new(
        id: u64,
        data: Vec<f64>,
        performance: f64,
        generation_cost: f64,
        generating_pattern: NoisePattern,
    ) -> Self {
        let fitness = if generation_cost > 0.0 {
            performance / generation_cost
        } else {
            performance
        };
        
        Self {
            id,
            data,
            performance,
            generation_cost,
            fitness,
            generating_pattern,
            timestamp: FemtosecondTimestamp::now(),
            significance: 0.0,
            confidence_interval: (0.0, 0.0),
            is_emergent: false,
        }
    }
    
    /// Calculate statistical significance relative to baseline
    pub fn calculate_significance(&mut self, baseline_mean: f64, baseline_std: f64) {
        if baseline_std > 0.0 {
            self.significance = (self.performance - baseline_mean) / baseline_std;
        }
    }
    
    /// Update emergence status based on significance threshold
    pub fn update_emergence_status(&mut self, threshold: f64) {
        self.is_emergent = self.significance > threshold;
    }
}

/// Represents different types of noise patterns used in generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoisePattern {
    /// Deterministic noise with structured, predictable patterns
    Deterministic {
        amplitude: f64,
        frequency: f64,
        phase: f64,
        systematic_bias: f64,
    },
    
    /// Fuzzy noise with continuous-valued, context-aware perturbations
    Fuzzy {
        membership_function: Vec<f64>,
        temporal_noise: Array1<f64>,
        context_weights: HashMap<String, f64>,
    },
    
    /// Quantum noise using superposition-based parallel exploration
    Quantum {
        superposition_coefficients: Vec<Complex64>,
        entanglement_matrix: Array2<Complex64>,
        decoherence_time: f64,
    },
    
    /// Molecular noise driven by thermal fluctuations
    Molecular {
        thermal_energy: f64,
        boltzmann_factor: f64,
        conformational_states: Vec<Vec<f64>>,
        transition_probabilities: Array2<f64>,
    },
    
    /// Combined noise from multiple domains
    Composite {
        components: Vec<NoisePattern>,
        mixing_weights: Vec<f64>,
    },
}

impl NoisePattern {
    /// Get the effective amplitude of this noise pattern
    pub fn amplitude(&self) -> f64 {
        match self {
            NoisePattern::Deterministic { amplitude, .. } => *amplitude,
            NoisePattern::Fuzzy { temporal_noise, .. } => {
                temporal_noise.iter().map(|x| x.abs()).sum::<f64>() / temporal_noise.len() as f64
            },
            NoisePattern::Quantum { superposition_coefficients, .. } => {
                superposition_coefficients.iter().map(|c| c.norm()).sum::<f64>()
            },
            NoisePattern::Molecular { thermal_energy, .. } => *thermal_energy,
            NoisePattern::Composite { components, mixing_weights } => {
                components.iter().zip(mixing_weights.iter())
                    .map(|(c, w)| c.amplitude() * w)
                    .sum()
            },
        }
    }
    
    /// Get the domain type of this noise pattern
    pub fn domain_type(&self) -> String {
        match self {
            NoisePattern::Deterministic { .. } => "Deterministic".to_string(),
            NoisePattern::Fuzzy { .. } => "Fuzzy".to_string(),
            NoisePattern::Quantum { .. } => "Quantum".to_string(),
            NoisePattern::Molecular { .. } => "Molecular".to_string(),
            NoisePattern::Composite { .. } => "Composite".to_string(),
        }
    }
}

/// Statistical signature for monitoring noise distribution and emergence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSignature {
    /// Mean value of the distribution
    pub mean: f64,
    
    /// Standard deviation of the distribution
    pub std_dev: f64,
    
    /// Variance of the distribution
    pub variance: f64,
    
    /// Skewness of the distribution
    pub skewness: f64,
    
    /// Kurtosis of the distribution
    pub kurtosis: f64,
    
    /// Entropy of the distribution
    pub entropy: f64,
    
    /// Number of samples in this signature
    pub sample_count: usize,
    
    /// Time window this signature covers
    pub time_window: Duration,
    
    /// Confidence level of statistical measurements
    pub confidence_level: f64,
}

impl StatisticalSignature {
    /// Create a new statistical signature from sample data
    pub fn from_samples(samples: &[f64], confidence_level: f64) -> Self {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        
        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        
        let std_dev = variance.sqrt();
        
        let skewness = samples.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>() / n;
        
        let kurtosis = samples.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>() / n - 3.0;
        
        // Calculate entropy (approximation for continuous distribution)
        let entropy = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * variance).ln();
        
        Self {
            mean,
            std_dev,
            variance,
            skewness,
            kurtosis,
            entropy,
            sample_count: samples.len(),
            time_window: Duration::default(),
            confidence_level,
        }
    }
    
    /// Check if another signature shows statistical emergence relative to this baseline
    pub fn detect_emergence(&self, other: &StatisticalSignature, threshold: f64) -> bool {
        if self.std_dev == 0.0 {
            return false;
        }
        
        let z_score = (other.mean - self.mean) / self.std_dev;
        z_score.abs() > threshold
    }
}

/// Metrics for tracking emergence detection performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceMetrics {
    /// Number of emergent solutions detected
    pub emergent_count: u64,
    
    /// Total number of candidates evaluated
    pub total_candidates: u64,
    
    /// Emergence detection rate (emergent / total)
    pub detection_rate: f64,
    
    /// Average significance of emergent solutions
    pub average_significance: f64,
    
    /// Maximum significance observed
    pub max_significance: f64,
    
    /// Time to first emergence detection
    pub time_to_first_emergence: Option<Duration>,
    
    /// Average time between emergent detections
    pub average_emergence_interval: Option<Duration>,
    
    /// Statistical confidence in emergence detections
    pub detection_confidence: f64,
}

impl EmergenceMetrics {
    /// Create new emergence metrics
    pub fn new() -> Self {
        Self {
            emergent_count: 0,
            total_candidates: 0,
            detection_rate: 0.0,
            average_significance: 0.0,
            max_significance: 0.0,
            time_to_first_emergence: None,
            average_emergence_interval: None,
            detection_confidence: 0.0,
        }
    }
    
    /// Update metrics with new candidate
    pub fn update(&mut self, candidate: &SolutionCandidate, start_time: Instant) {
        self.total_candidates += 1;
        
        if candidate.is_emergent {
            self.emergent_count += 1;
            
            // Update time to first emergence
            if self.time_to_first_emergence.is_none() {
                self.time_to_first_emergence = Some(start_time.elapsed());
            }
            
            // Update significance tracking
            let prev_avg = self.average_significance;
            let count = self.emergent_count as f64;
            self.average_significance = (prev_avg * (count - 1.0) + candidate.significance) / count;
            
            if candidate.significance > self.max_significance {
                self.max_significance = candidate.significance;
            }
        }
        
        // Update detection rate
        self.detection_rate = self.emergent_count as f64 / self.total_candidates as f64;
    }
}

impl Default for EmergenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// High-precision timestamp for femtosecond-level timing
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FemtosecondTimestamp {
    /// Seconds since epoch
    pub seconds: u64,
    
    /// Femtoseconds within the second (0 to 10^15 - 1)
    pub femtoseconds: u64,
}

impl FemtosecondTimestamp {
    /// Create timestamp for current time
    pub fn now() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards");
        
        Self {
            seconds: now.as_secs(),
            femtoseconds: (now.subsec_nanos() as u64) * 1_000_000, // Convert ns to fs
        }
    }
    
    /// Create timestamp with specific femtosecond precision
    pub fn with_femtoseconds(seconds: u64, femtoseconds: u64) -> Self {
        Self { seconds, femtoseconds }
    }
    
    /// Calculate duration since another timestamp
    pub fn duration_since(&self, other: &FemtosecondTimestamp) -> Duration {
        let sec_diff = self.seconds - other.seconds;
        let fs_diff = self.femtoseconds - other.femtoseconds;
        
        Duration::new(sec_diff, (fs_diff / 1_000_000) as u32) // Convert fs to ns
    }
    
    /// Convert to total femtoseconds since epoch
    pub fn total_femtoseconds(&self) -> u128 {
        (self.seconds as u128) * 1_000_000_000_000_000 + (self.femtoseconds as u128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution_candidate_creation() {
        let pattern = NoisePattern::Deterministic {
            amplitude: 1.0,
            frequency: 2.0,
            phase: 0.0,
            systematic_bias: 0.1,
        };
        
        let candidate = SolutionCandidate::new(
            1,
            vec![1.0, 2.0, 3.0],
            0.8,
            0.4,
            pattern,
        );
        
        assert_eq!(candidate.id, 1);
        assert_eq!(candidate.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(candidate.performance, 0.8);
        assert_eq!(candidate.generation_cost, 0.4);
        assert_eq!(candidate.fitness, 2.0); // 0.8 / 0.4
        assert!(!candidate.is_emergent);
    }
    
    #[test]
    fn test_noise_pattern_amplitude() {
        let pattern = NoisePattern::Deterministic {
            amplitude: 2.5,
            frequency: 1.0,
            phase: 0.0,
            systematic_bias: 0.0,
        };
        
        assert_eq!(pattern.amplitude(), 2.5);
        assert_eq!(pattern.domain_type(), "Deterministic");
    }
    
    #[test]
    fn test_statistical_signature() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let signature = StatisticalSignature::from_samples(&samples, 0.95);
        
        assert_eq!(signature.mean, 3.0);
        assert_eq!(signature.sample_count, 5);
        assert_eq!(signature.confidence_level, 0.95);
    }
    
    #[test]
    fn test_femtosecond_timestamp() {
        let ts1 = FemtosecondTimestamp::with_femtoseconds(1000, 500);
        let ts2 = FemtosecondTimestamp::with_femtoseconds(1001, 1000);
        
        let duration = ts2.duration_since(&ts1);
        assert!(duration.as_secs() >= 1);
    }
    
    #[test]
    fn test_emergence_metrics() {
        let mut metrics = EmergenceMetrics::new();
        let start_time = Instant::now();
        
        let mut candidate = SolutionCandidate::new(
            1,
            vec![1.0],
            0.9,
            0.1,
            NoisePattern::Deterministic {
                amplitude: 1.0,
                frequency: 1.0,
                phase: 0.0,
                systematic_bias: 0.0,
            },
        );
        
        candidate.is_emergent = true;
        candidate.significance = 3.5;
        
        metrics.update(&candidate, start_time);
        
        assert_eq!(metrics.emergent_count, 1);
        assert_eq!(metrics.total_candidates, 1);
        assert_eq!(metrics.detection_rate, 1.0);
        assert_eq!(metrics.average_significance, 3.5);
        assert_eq!(metrics.max_significance, 3.5);
    }
} 