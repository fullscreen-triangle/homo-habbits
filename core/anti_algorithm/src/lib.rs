//! # Anti-Algorithm Computational Engine
//! 
//! This module implements the Anti-Algorithm Principle: achieving computational success
//! through intentional massive failure generation at femtosecond temporal precision.
//! 
//! ## Core Concept
//! 
//! Rather than optimizing for correctness, the system generates wrong solutions at rates
//! exceeding 10^15 attempts per second across multiple noise domains, enabling correct
//! solutions to emerge through statistical convergence.
//! 
//! ## Architecture
//! 
//! ```
//! Multi-Domain Noise Generation → Statistical Emergence Detection → Solution Extraction
//! ```
//! 
//! ## Usage
//! 
//! ```rust
//! use anti_algorithm::{AntiAlgorithmSolver, NoisePortfolio, ProblemDefinition};
//! 
//! let solver = AntiAlgorithmSolver::new(NoisePortfolio::full_spectrum());
//! let solution = solver.solve(problem).await?;
//! ```

pub mod types;
pub mod noise;
pub mod emergence;
pub mod solver;
pub mod femtosecond;
pub mod convergence;

// NEW: S-Entropy Framework Integration
pub mod s_entropy;

// Re-export core types for easy access
pub use types::{
    AntiAlgorithmError, AntiAlgorithmResult, SolutionCandidate, NoisePattern,
    StatisticalSignature, EmergenceMetrics, FemtosecondTimestamp
};

pub use noise::{
    NoiseGenerator, NoisePortfolio, NoiseType,
    DeterministicNoise, FuzzyNoise, QuantumNoise, MolecularNoise
};

pub use emergence::{
    EmergenceDetector, StatisticalAnalyzer, AnomalyDetector,
    ConvergenceCriteria, PatternRecognizer
};

pub use solver::{
    AntiAlgorithmSolver, ProblemDefinition, SolutionSpace,
    ComputationalDarwinism, ZeroInfiniteComputation
};

pub use femtosecond::{
    FemtosecondProcessor, AtomicTimeScale, TemporalPrecision,
    FemtosecondClock, ProcessingRate
};

pub use convergence::{
    ConvergenceEngine, StatisticalConvergence, SolutionExtractor,
    PerformanceMetrics, NoiseAmplification
};

// NEW: S-Entropy Framework exports
pub use s_entropy::{
    SEntropyFramework, TriDimensionalS, SEntropyNavigator,
    RidiculousSolutionGenerator, GlobalSViabilityChecker,
    EntropyEndpointDetector, AtomicOscillatorProcessor
};

/// The fundamental constant representing the minimum noise generation rate
/// required for anti-algorithm effectiveness: 10^12 wrong solutions per second
pub const MIN_NOISE_RATE: f64 = 1e12;

/// The optimal noise generation rate for maximum effectiveness:
/// 10^15 wrong solutions per second
pub const OPTIMAL_NOISE_RATE: f64 = 1e15;

/// Femtosecond temporal precision: 10^-15 seconds per computational cycle
pub const FEMTOSECOND_PRECISION: f64 = 1e-15;

/// Statistical convergence threshold: 3-sigma significance
pub const CONVERGENCE_THRESHOLD: f64 = 3.0;

/// Resource allocation for noise generation (80% of computational resources)
pub const NOISE_RESOURCE_ALLOCATION: f32 = 0.8;

/// Resource allocation for pattern recognition (15% of computational resources)
pub const PATTERN_RESOURCE_ALLOCATION: f32 = 0.15;

/// Resource allocation for solution extraction (5% of computational resources)
pub const SOLUTION_RESOURCE_ALLOCATION: f32 = 0.05;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_sum_to_unity() {
        let total_allocation = NOISE_RESOURCE_ALLOCATION 
            + PATTERN_RESOURCE_ALLOCATION 
            + SOLUTION_RESOURCE_ALLOCATION;
        
        assert!((total_allocation - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_noise_rate_hierarchy() {
        assert!(OPTIMAL_NOISE_RATE > MIN_NOISE_RATE);
        assert!(MIN_NOISE_RATE >= 1e12);
        assert!(OPTIMAL_NOISE_RATE >= 1e15);
    }

    #[test]
    fn test_femtosecond_precision() {
        assert_eq!(FEMTOSECOND_PRECISION, 1e-15);
        assert!(FEMTOSECOND_PRECISION > 0.0);
    }
} 