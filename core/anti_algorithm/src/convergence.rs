//! Statistical Convergence Engine
//! 
//! This module implements convergence detection and solution extraction
//! from the massive noise generation process of the Anti-Algorithm.

use crate::types::{AntiAlgorithmResult, AntiAlgorithmError, SolutionCandidate, EmergenceMetrics};
use crate::emergence::EmergentSolution;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Main convergence engine for solution extraction
pub struct ConvergenceEngine {
    /// Convergence criteria
    criteria: ConvergenceCriteria,
    
    /// Statistical convergence tracker
    convergence_tracker: StatisticalConvergence,
    
    /// Solution extraction system
    solution_extractor: SolutionExtractor,
    
    /// Performance metrics
    performance_metrics: PerformanceMetrics,
}

/// Convergence criteria configuration
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Minimum statistical significance required
    pub significance_threshold: f64,
    
    /// Stability window size
    pub stability_window: usize,
    
    /// Maximum variance allowed for convergence
    pub variance_threshold: f64,
    
    /// Minimum improvement rate
    pub improvement_threshold: f64,
    
    /// Maximum time without improvement
    pub stagnation_timeout: Duration,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            significance_threshold: 3.0,
            stability_window: 100,
            variance_threshold: 1e-6,
            improvement_threshold: 0.01,
            stagnation_timeout: Duration::from_secs(30),
        }
    }
}

/// Statistical convergence tracking
pub struct StatisticalConvergence {
    /// Recent solution quality history
    quality_history: VecDeque<f64>,
    
    /// Convergence detection state
    convergence_state: ConvergenceState,
    
    /// Last improvement timestamp
    last_improvement: Option<Instant>,
    
    /// Current convergence rate
    convergence_rate: f64,
}

/// Convergence detection states
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceState {
    Exploring,
    Converging { rate: f64 },
    Converged { final_quality: f64 },
    Stagnated,
}

/// Solution extraction system
pub struct SolutionExtractor {
    /// Best solutions archive
    best_solutions: Vec<EmergentSolution>,
    
    /// Archive size limit
    archive_size: usize,
    
    /// Diversity maintenance threshold
    diversity_threshold: f64,
    
    /// Quality improvement tracking
    quality_improvements: Vec<QualityImprovement>,
}

/// Quality improvement record
#[derive(Debug, Clone)]
pub struct QualityImprovement {
    /// Timestamp of improvement
    pub timestamp: Instant,
    
    /// Previous best quality
    pub previous_quality: f64,
    
    /// New best quality
    pub new_quality: f64,
    
    /// Improvement magnitude
    pub improvement: f64,
}

/// Performance metrics for convergence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Time to first convergence
    pub time_to_convergence: Option<Duration>,
    
    /// Total solutions evaluated
    pub solutions_evaluated: u64,
    
    /// Convergence efficiency (quality/time)
    pub convergence_efficiency: f64,
    
    /// Final solution quality
    pub final_quality: f64,
    
    /// Number of quality improvements
    pub improvement_count: u64,
    
    /// Average improvement magnitude
    pub average_improvement: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            time_to_convergence: None,
            solutions_evaluated: 0,
            convergence_efficiency: 0.0,
            final_quality: 0.0,
            improvement_count: 0,
            average_improvement: 0.0,
        }
    }
}

impl ConvergenceEngine {
    /// Create new convergence engine
    pub fn new() -> Self {
        Self {
            criteria: ConvergenceCriteria::default(),
            convergence_tracker: StatisticalConvergence::new(),
            solution_extractor: SolutionExtractor::new(1000), // Archive top 1000 solutions
            performance_metrics: PerformanceMetrics::default(),
        }
    }
    
    /// Update convergence tracking with new solutions
    pub async fn update_convergence(&mut self, emergent_solutions: &[EmergentSolution]) -> AntiAlgorithmResult<()> {
        // Update solution archive
        self.solution_extractor.update_archive(emergent_solutions).await;
        
        // Get current best quality
        let current_quality = self.solution_extractor.get_best_quality();
        
        // Update convergence tracking
        self.convergence_tracker.update_quality(current_quality);
        
        // Check convergence criteria
        let convergence_state = self.check_convergence().await?;
        self.convergence_tracker.convergence_state = convergence_state;
        
        // Update performance metrics
        self.update_performance_metrics(emergent_solutions.len()).await;
        
        Ok(())
    }
    
    /// Check if convergence criteria are met
    async fn check_convergence(&self) -> AntiAlgorithmResult<ConvergenceState> {
        let quality_history = &self.convergence_tracker.quality_history;
        
        if quality_history.len() < self.criteria.stability_window {
            return Ok(ConvergenceState::Exploring);
        }
        
        // Check for stagnation
        if let Some(last_improvement) = self.convergence_tracker.last_improvement {
            if last_improvement.elapsed() > self.criteria.stagnation_timeout {
                return Ok(ConvergenceState::Stagnated);
            }
        }
        
        // Calculate variance of recent quality values
        let recent_values: Vec<f64> = quality_history.iter()
            .rev()
            .take(self.criteria.stability_window)
            .copied()
            .collect();
        
        let mean = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
        let variance = recent_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / recent_values.len() as f64;
        
        // Check convergence criteria
        if variance < self.criteria.variance_threshold {
            Ok(ConvergenceState::Converged { final_quality: mean })
        } else {
            // Calculate convergence rate
            let rate = self.calculate_convergence_rate(&recent_values);
            Ok(ConvergenceState::Converging { rate })
        }
    }
    
    /// Calculate convergence rate from quality history
    fn calculate_convergence_rate(&self, quality_values: &[f64]) -> f64 {
        if quality_values.len() < 2 {
            return 0.0;
        }
        
        // Linear regression slope as convergence rate
        let n = quality_values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = quality_values.iter().sum::<f64>() / n;
        
        let numerator: f64 = quality_values.iter()
            .enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();
        
        let denominator: f64 = quality_values.iter()
            .enumerate()
            .map(|(i, _)| (i as f64 - x_mean).powi(2))
            .sum();
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    /// Update performance metrics
    async fn update_performance_metrics(&mut self, new_solutions_count: usize) {
        self.performance_metrics.solutions_evaluated += new_solutions_count as u64;
        
        if let Some(best_solution) = self.solution_extractor.get_best_solution() {
            let new_quality = best_solution.candidate.fitness;
            
            // Check for quality improvement
            if new_quality > self.performance_metrics.final_quality {
                let improvement = new_quality - self.performance_metrics.final_quality;
                
                self.solution_extractor.quality_improvements.push(QualityImprovement {
                    timestamp: Instant::now(),
                    previous_quality: self.performance_metrics.final_quality,
                    new_quality,
                    improvement,
                });
                
                self.performance_metrics.improvement_count += 1;
                self.performance_metrics.final_quality = new_quality;
                
                // Update average improvement
                let total_improvement: f64 = self.solution_extractor.quality_improvements.iter()
                    .map(|qi| qi.improvement)
                    .sum();
                self.performance_metrics.average_improvement = 
                    total_improvement / self.performance_metrics.improvement_count as f64;
            }
        }
        
        // Update convergence efficiency
        if let ConvergenceState::Converged { .. } = self.convergence_tracker.convergence_state {
            if let Some(first_improvement) = self.solution_extractor.quality_improvements.first() {
                let time_to_convergence = first_improvement.timestamp.elapsed();
                self.performance_metrics.time_to_convergence = Some(time_to_convergence);
                
                if time_to_convergence.as_secs_f64() > 0.0 {
                    self.performance_metrics.convergence_efficiency = 
                        self.performance_metrics.final_quality / time_to_convergence.as_secs_f64();
                }
            }
        }
    }
    
    /// Get current convergence state
    pub fn convergence_state(&self) -> &ConvergenceState {
        &self.convergence_tracker.convergence_state
    }
    
    /// Get best solution found so far
    pub fn get_best_solution(&self) -> Option<&EmergentSolution> {
        self.solution_extractor.get_best_solution()
    }
    
    /// Get convergence metrics
    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }
    
    /// Check if convergence has been achieved
    pub fn is_converged(&self) -> bool {
        matches!(self.convergence_tracker.convergence_state, ConvergenceState::Converged { .. })
    }
}

impl StatisticalConvergence {
    /// Create new statistical convergence tracker
    pub fn new() -> Self {
        Self {
            quality_history: VecDeque::with_capacity(10000),
            convergence_state: ConvergenceState::Exploring,
            last_improvement: None,
            convergence_rate: 0.0,
        }
    }
    
    /// Update quality tracking
    pub fn update_quality(&mut self, quality: f64) {
        // Check for improvement
        if let Some(&last_quality) = self.quality_history.back() {
            if quality > last_quality {
                self.last_improvement = Some(Instant::now());
            }
        } else {
            self.last_improvement = Some(Instant::now());
        }
        
        self.quality_history.push_back(quality);
        
        // Limit history size
        if self.quality_history.len() > 10000 {
            self.quality_history.pop_front();
        }
    }
    
    /// Get current convergence rate
    pub fn current_convergence_rate(&self) -> f64 {
        self.convergence_rate
    }
}

impl SolutionExtractor {
    /// Create new solution extractor
    pub fn new(archive_size: usize) -> Self {
        Self {
            best_solutions: Vec::with_capacity(archive_size),
            archive_size,
            diversity_threshold: 0.01,
            quality_improvements: Vec::new(),
        }
    }
    
    /// Update solution archive with new emergent solutions
    pub async fn update_archive(&mut self, emergent_solutions: &[EmergentSolution]) {
        for solution in emergent_solutions {
            self.add_solution_to_archive(solution.clone());
        }
        
        // Maintain archive size limit
        if self.best_solutions.len() > self.archive_size {
            // Sort by fitness and keep best solutions
            self.best_solutions.sort_by(|a, b| 
                b.candidate.fitness.partial_cmp(&a.candidate.fitness).unwrap()
            );
            self.best_solutions.truncate(self.archive_size);
        }
    }
    
    /// Add solution to archive with diversity checking
    fn add_solution_to_archive(&mut self, solution: EmergentSolution) {
        // Check if solution is diverse enough
        if self.is_diverse_enough(&solution) {
            self.best_solutions.push(solution);
        }
    }
    
    /// Check if solution is diverse enough compared to existing solutions
    fn is_diverse_enough(&self, solution: &EmergentSolution) -> bool {
        for existing in &self.best_solutions {
            let similarity = self.calculate_similarity(&solution.candidate, &existing.candidate);
            if similarity > (1.0 - self.diversity_threshold) {
                // Too similar to existing solution
                if solution.candidate.fitness <= existing.candidate.fitness {
                    return false;
                }
            }
        }
        true
    }
    
    /// Calculate similarity between two solution candidates
    fn calculate_similarity(&self, sol1: &SolutionCandidate, sol2: &SolutionCandidate) -> f64 {
        if sol1.data.len() != sol2.data.len() {
            return 0.0;
        }
        
        // Cosine similarity
        let dot_product: f64 = sol1.data.iter()
            .zip(sol2.data.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let norm1 = sol1.data.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm2 = sol2.data.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        
        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }
    
    /// Get best solution from archive
    pub fn get_best_solution(&self) -> Option<&EmergentSolution> {
        self.best_solutions.iter()
            .max_by(|a, b| a.candidate.fitness.partial_cmp(&b.candidate.fitness).unwrap())
    }
    
    /// Get best quality value
    pub fn get_best_quality(&self) -> f64 {
        self.get_best_solution()
            .map(|sol| sol.candidate.fitness)
            .unwrap_or(0.0)
    }
    
    /// Get all solutions in archive
    pub fn get_all_solutions(&self) -> &[EmergentSolution] {
        &self.best_solutions
    }
    
    /// Get solution count in archive
    pub fn solution_count(&self) -> usize {
        self.best_solutions.len()
    }
}

/// Noise amplification system for boosting successful patterns
pub struct NoiseAmplification {
    /// Amplification factors by pattern type
    amplification_factors: std::collections::HashMap<String, f64>,
    
    /// Success history tracking
    success_history: Vec<AmplificationRecord>,
    
    /// Maximum amplification factor
    max_amplification: f64,
    
    /// Amplification decay rate
    decay_rate: f64,
}

/// Record of noise pattern amplification
#[derive(Debug, Clone)]
pub struct AmplificationRecord {
    /// Pattern identifier
    pub pattern_id: String,
    
    /// Amplification factor applied
    pub amplification_factor: f64,
    
    /// Success rate after amplification
    pub success_rate: f64,
    
    /// Timestamp of amplification
    pub timestamp: Instant,
}

impl NoiseAmplification {
    /// Create new noise amplification system
    pub fn new() -> Self {
        Self {
            amplification_factors: std::collections::HashMap::new(),
            success_history: Vec::new(),
            max_amplification: 10.0,
            decay_rate: 0.95,
        }
    }
    
    /// Amplify successful noise patterns
    pub async fn amplify_pattern(&mut self, pattern_id: String, success_rate: f64) {
        let current_factor = self.amplification_factors.get(&pattern_id).copied().unwrap_or(1.0);
        
        // Increase amplification for successful patterns
        let new_factor = if success_rate > 0.5 {
            (current_factor * 1.1).min(self.max_amplification)
        } else {
            (current_factor * self.decay_rate).max(0.1)
        };
        
        self.amplification_factors.insert(pattern_id.clone(), new_factor);
        
        // Record amplification
        self.success_history.push(AmplificationRecord {
            pattern_id,
            amplification_factor: new_factor,
            success_rate,
            timestamp: Instant::now(),
        });
        
        // Limit history size
        if self.success_history.len() > 1000 {
            self.success_history.drain(0..100);
        }
    }
    
    /// Get amplification factor for pattern
    pub fn get_amplification_factor(&self, pattern_id: &str) -> f64 {
        self.amplification_factors.get(pattern_id).copied().unwrap_or(1.0)
    }
    
    /// Decay all amplification factors over time
    pub fn decay_amplification(&mut self) {
        for factor in self.amplification_factors.values_mut() {
            *factor *= self.decay_rate;
            if *factor < 0.1 {
                *factor = 0.1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SolutionCandidate, NoisePattern, FemtosecondTimestamp};
    use crate::emergence::EmergentSolution;

    #[tokio::test]
    async fn test_convergence_engine_creation() {
        let engine = ConvergenceEngine::new();
        assert!(matches!(engine.convergence_state(), ConvergenceState::Exploring));
    }
    
    #[tokio::test]
    async fn test_solution_archive_update() {
        let mut extractor = SolutionExtractor::new(10);
        
        let candidate = SolutionCandidate::new(
            1,
            vec![1.0, 2.0, 3.0],
            0.8,
            0.1,
            NoisePattern::Deterministic {
                amplitude: 1.0,
                frequency: 1.0,
                phase: 0.0,
                systematic_bias: 0.0,
            },
        );
        
        let emergent = EmergentSolution {
            candidate,
            significance_level: 3.5,
            detection_confidence: 0.9,
            emergence_type: crate::emergence::EmergenceType::PerformanceSpike {
                magnitude: 0.8,
                baseline_deviation: 3.5,
            },
            emergence_time: std::time::Duration::from_secs(1),
            convergence_rate: 0.1,
        };
        
        extractor.update_archive(&[emergent]).await;
        assert_eq!(extractor.solution_count(), 1);
        assert!(extractor.get_best_solution().is_some());
    }
    
    #[test]
    fn test_convergence_criteria() {
        let criteria = ConvergenceCriteria::default();
        assert_eq!(criteria.significance_threshold, 3.0);
        assert_eq!(criteria.stability_window, 100);
    }
    
    #[test]
    fn test_statistical_convergence() {
        let mut tracker = StatisticalConvergence::new();
        tracker.update_quality(0.5);
        tracker.update_quality(0.7);
        tracker.update_quality(0.9);
        
        assert_eq!(tracker.quality_history.len(), 3);
        assert!(tracker.last_improvement.is_some());
    }
    
    #[test]
    fn test_noise_amplification() {
        let mut amplifier = NoiseAmplification::new();
        assert_eq!(amplifier.get_amplification_factor("test_pattern"), 1.0);
        
        // Test amplification doesn't require async for this simple case
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            amplifier.amplify_pattern("test_pattern".to_string(), 0.8).await;
        });
        
        assert!(amplifier.get_amplification_factor("test_pattern") > 1.0);
    }
    
    #[test]
    fn test_similarity_calculation() {
        let extractor = SolutionExtractor::new(10);
        
        let sol1 = SolutionCandidate::new(
            1,
            vec![1.0, 0.0, 0.0],
            0.8,
            0.1,
            NoisePattern::Deterministic {
                amplitude: 1.0,
                frequency: 1.0,
                phase: 0.0,
                systematic_bias: 0.0,
            },
        );
        
        let sol2 = SolutionCandidate::new(
            2,
            vec![0.0, 1.0, 0.0],
            0.7,
            0.1,
            NoisePattern::Deterministic {
                amplitude: 1.0,
                frequency: 1.0,
                phase: 0.0,
                systematic_bias: 0.0,
            },
        );
        
        let similarity = extractor.calculate_similarity(&sol1, &sol2);
        assert_eq!(similarity, 0.0); // Orthogonal vectors have zero similarity
    }
} 