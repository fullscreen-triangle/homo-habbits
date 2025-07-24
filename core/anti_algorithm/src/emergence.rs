//! Statistical Emergence Detection System
//! 
//! This module implements pattern recognition and anomaly detection within massive
//! noise streams to identify emergent solutions. Solutions appear as statistical
//! outliers when the Anti-Algorithm generates them through computational natural selection.
//! 
//! The system monitors noise distributions and detects when solution candidates
//! exhibit statistical significance exceeding baseline chaos levels.

use crate::types::{
    AntiAlgorithmResult, AntiAlgorithmError, SolutionCandidate, StatisticalSignature,
    EmergenceMetrics, NoisePattern, FemtosecondTimestamp,
};
use async_trait::async_trait;
use ndarray::{Array1, Array2};
use statrs::distribution::{Normal, ContinuousCDF};
use statrs::statistics::{Statistics, OrderStatistics};
use std::collections::{VecDeque, HashMap};
use std::time::{Instant, Duration};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Trait for detecting emergence in noise streams
#[async_trait]
pub trait EmergenceDetector: Send + Sync {
    /// Analyze a batch of solution candidates for emergence
    async fn detect_emergence(
        &mut self,
        candidates: &[SolutionCandidate],
        baseline: &StatisticalSignature,
    ) -> AntiAlgorithmResult<Vec<EmergentSolution>>;
    
    /// Update detection parameters based on recent performance
    async fn update_detection_parameters(&mut self, performance_metrics: &EmergenceMetrics);
    
    /// Get current detection sensitivity
    fn detection_threshold(&self) -> f64;
    
    /// Reset detector to initial state
    async fn reset(&mut self);
}

/// Represents a solution that has emerged from noise
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentSolution {
    /// The solution candidate that emerged
    pub candidate: SolutionCandidate,
    
    /// Statistical significance level
    pub significance_level: f64,
    
    /// Confidence in emergence detection
    pub detection_confidence: f64,
    
    /// Type of emergence pattern detected
    pub emergence_type: EmergenceType,
    
    /// Time since noise generation started
    pub emergence_time: Duration,
    
    /// Convergence rate leading to this emergence
    pub convergence_rate: f64,
}

/// Types of emergence patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergenceType {
    /// Sudden spike in performance
    PerformanceSpike {
        magnitude: f64,
        baseline_deviation: f64,
    },
    
    /// Gradual convergence to solution
    GradualConvergence {
        rate: f64,
        stability: f64,
    },
    
    /// Oscillatory approach to solution
    OscillatoryConvergence {
        frequency: f64,
        damping: f64,
        amplitude: f64,
    },
    
    /// Quantum superposition collapse
    QuantumCollapse {
        coherence_loss: f64,
        state_reduction: f64,
    },
    
    /// Molecular configuration optimization
    MolecularOptimization {
        energy_reduction: f64,
        stability_increase: f64,
    },
}

/// Comprehensive statistical analyzer for noise streams
pub struct StatisticalAnalyzer {
    /// Sliding window of recent performance values
    performance_window: VecDeque<f64>,
    
    /// Window size for statistical analysis
    window_size: usize,
    
    /// Baseline statistical signature
    baseline_signature: Option<StatisticalSignature>,
    
    /// Detection threshold (sigma levels)
    detection_threshold: f64,
    
    /// Minimum samples required for reliable detection
    min_samples: usize,
    
    /// Adaptive threshold adjustment rate
    adaptation_rate: f64,
    
    /// Performance history for trend analysis
    performance_history: Vec<f64>,
    
    /// Emergence detection count
    emergence_count: u64,
    
    /// False positive tracking
    false_positive_count: u64,
}

impl StatisticalAnalyzer {
    /// Create new statistical analyzer
    pub fn new(window_size: usize, detection_threshold: f64) -> Self {
        Self {
            performance_window: VecDeque::with_capacity(window_size),
            window_size,
            baseline_signature: None,
            detection_threshold,
            min_samples: window_size / 4,
            adaptation_rate: 0.01,
            performance_history: Vec::new(),
            emergence_count: 0,
            false_positive_count: 0,
        }
    }
    
    /// Update baseline from noise samples
    pub fn update_baseline(&mut self, noise_samples: &[f64]) {
        if noise_samples.len() >= self.min_samples {
            self.baseline_signature = Some(StatisticalSignature::from_samples(noise_samples, 0.95));
        }
    }
    
    /// Detect statistical anomalies in performance data
    pub fn detect_anomalies(&mut self, values: &[f64]) -> Vec<AnomalyDetection> {
        let mut anomalies = Vec::new();
        
        if let Some(baseline) = &self.baseline_signature {
            for (i, &value) in values.iter().enumerate() {
                let z_score = if baseline.std_dev > 0.0 {
                    (value - baseline.mean) / baseline.std_dev
                } else {
                    0.0
                };
                
                if z_score.abs() > self.detection_threshold {
                    anomalies.push(AnomalyDetection {
                        index: i,
                        value,
                        z_score,
                        significance: z_score.abs(),
                        anomaly_type: if z_score > 0.0 {
                            AnomalyType::PositiveSpike
                        } else {
                            AnomalyType::NegativeSpike
                        },
                    });
                }
            }
        }
        
        anomalies
    }
    
    /// Calculate convergence rate from recent data
    pub fn calculate_convergence_rate(&self) -> f64 {
        if self.performance_history.len() < 3 {
            return 0.0;
        }
        
        let recent_samples = &self.performance_history[self.performance_history.len().saturating_sub(10)..];
        
        // Calculate linear regression slope as convergence rate
        let n = recent_samples.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = recent_samples.iter().sum::<f64>() / n;
        
        let numerator: f64 = recent_samples.iter()
            .enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();
        
        let denominator: f64 = recent_samples.iter()
            .enumerate()
            .map(|(i, _)| (i as f64 - x_mean).powi(2))
            .sum();
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    /// Update performance tracking
    pub fn update_performance(&mut self, performance: f64) {
        self.performance_window.push_back(performance);
        if self.performance_window.len() > self.window_size {
            self.performance_window.pop_front();
        }
        
        self.performance_history.push(performance);
        
        // Limit history size to prevent memory growth
        if self.performance_history.len() > 10000 {
            self.performance_history.drain(0..1000);
        }
    }
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyDetection {
    /// Index in the analyzed array
    pub index: usize,
    
    /// The anomalous value
    pub value: f64,
    
    /// Z-score relative to baseline
    pub z_score: f64,
    
    /// Statistical significance
    pub significance: f64,
    
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,
}

/// Types of anomalies
#[derive(Debug, Clone)]
pub enum AnomalyType {
    PositiveSpike,
    NegativeSpike,
    TrendChange,
    Oscillation,
    Outlier,
}

/// Main anomaly detector implementation
#[derive(Debug)]
pub struct AnomalyDetector {
    /// Statistical analyzer
    analyzer: StatisticalAnalyzer,
    
    /// Pattern recognizer for different emergence types
    pattern_recognizer: PatternRecognizer,
    
    /// Convergence criteria checker
    convergence_checker: ConvergenceCriteria,
    
    /// Start time for emergence timing
    start_time: Instant,
    
    /// Metrics tracking
    metrics: EmergenceMetrics,
}

impl AnomalyDetector {
    /// Create new anomaly detector
    pub fn new(detection_threshold: f64) -> Self {
        Self {
            analyzer: StatisticalAnalyzer::new(1000, detection_threshold),
            pattern_recognizer: PatternRecognizer::new(),
            convergence_checker: ConvergenceCriteria::new(detection_threshold),
            start_time: Instant::now(),
            metrics: EmergenceMetrics::new(),
        }
    }
    
    /// Process a candidate for emergence detection
    async fn process_candidate(
        &mut self,
        candidate: &SolutionCandidate,
        baseline: &StatisticalSignature,
    ) -> Option<EmergentSolution> {
        // Update performance tracking
        self.analyzer.update_performance(candidate.performance);
        
        // Calculate significance relative to baseline
        let z_score = if baseline.std_dev > 0.0 {
            (candidate.performance - baseline.mean) / baseline.std_dev
        } else {
            0.0
        };
        
        // Check if significance exceeds threshold
        if z_score.abs() > self.analyzer.detection_threshold {
            // Determine emergence type
            let emergence_type = self.pattern_recognizer.classify_emergence(
                candidate,
                &self.analyzer.performance_history,
            ).await;
            
            // Calculate convergence rate
            let convergence_rate = self.analyzer.calculate_convergence_rate();
            
            // Calculate detection confidence
            let detection_confidence = self.calculate_detection_confidence(z_score, convergence_rate);
            
            Some(EmergentSolution {
                candidate: candidate.clone(),
                significance_level: z_score.abs(),
                detection_confidence,
                emergence_type,
                emergence_time: self.start_time.elapsed(),
                convergence_rate,
            })
        } else {
            None
        }
    }
    
    /// Calculate confidence in emergence detection
    fn calculate_detection_confidence(&self, z_score: f64, convergence_rate: f64) -> f64 {
        // Base confidence from statistical significance
        let significance_confidence = (1.0 - (-z_score.abs().powi(2) / 2.0).exp()).min(0.99);
        
        // Convergence rate bonus
        let convergence_bonus = if convergence_rate > 0.0 {
            (convergence_rate * 10.0).tanh() * 0.1
        } else {
            0.0
        };
        
        // Historical accuracy adjustment
        let accuracy_ratio = if self.metrics.total_candidates > 0 {
            self.metrics.emergent_count as f64 / self.metrics.total_candidates as f64
        } else {
            0.5
        };
        
        let historical_adjustment = (accuracy_ratio - 0.1).max(0.0) * 0.1;
        
        (significance_confidence + convergence_bonus + historical_adjustment).min(0.99)
    }
}

#[async_trait]
impl EmergenceDetector for AnomalyDetector {
    async fn detect_emergence(
        &mut self,
        candidates: &[SolutionCandidate],
        baseline: &StatisticalSignature,
    ) -> AntiAlgorithmResult<Vec<EmergentSolution>> {
        let mut emergent_solutions = Vec::new();
        
        // Update baseline in analyzer
        let baseline_samples: Vec<f64> = candidates.iter()
            .take(100) // Use first 100 as baseline samples
            .map(|c| c.performance)
            .collect();
        
        if !baseline_samples.is_empty() {
            self.analyzer.update_baseline(&baseline_samples);
        }
        
        // Process each candidate
        for candidate in candidates {
            self.metrics.update(candidate, self.start_time);
            
            if let Some(emergent) = self.process_candidate(candidate, baseline).await {
                emergent_solutions.push(emergent);
            }
        }
        
        Ok(emergent_solutions)
    }
    
    async fn update_detection_parameters(&mut self, performance_metrics: &EmergenceMetrics) {
        // Adjust threshold based on false positive rate
        let false_positive_rate = if performance_metrics.total_candidates > 0 {
            self.analyzer.false_positive_count as f64 / performance_metrics.total_candidates as f64
        } else {
            0.0
        };
        
        if false_positive_rate > 0.1 {
            // Too many false positives, increase threshold
            self.analyzer.detection_threshold *= 1.0 + self.analyzer.adaptation_rate;
        } else if false_positive_rate < 0.01 && performance_metrics.detection_rate < 0.05 {
            // Too few detections, decrease threshold
            self.analyzer.detection_threshold *= 1.0 - self.analyzer.adaptation_rate;
        }
        
        // Ensure threshold stays in reasonable range
        self.analyzer.detection_threshold = self.analyzer.detection_threshold.clamp(1.5, 5.0);
    }
    
    fn detection_threshold(&self) -> f64 {
        self.analyzer.detection_threshold
    }
    
    async fn reset(&mut self) {
        self.analyzer = StatisticalAnalyzer::new(1000, crate::CONVERGENCE_THRESHOLD);
        self.metrics = EmergenceMetrics::new();
        self.start_time = Instant::now();
    }
}

/// Pattern recognition for different types of emergence
#[derive(Debug)]
pub struct PatternRecognizer {
    /// FFT analyzer for frequency domain patterns
    fft_analyzer: FftAnalyzer,
    
    /// Trend analyzer for gradual changes
    trend_analyzer: TrendAnalyzer,
    
    /// Oscillation detector
    oscillation_detector: OscillationDetector,
}

impl PatternRecognizer {
    /// Create new pattern recognizer
    pub fn new() -> Self {
        Self {
            fft_analyzer: FftAnalyzer::new(),
            trend_analyzer: TrendAnalyzer::new(),
            oscillation_detector: OscillationDetector::new(),
        }
    }
    
    /// Classify the type of emergence pattern
    pub async fn classify_emergence(
        &self,
        candidate: &SolutionCandidate,
        performance_history: &[f64],
    ) -> EmergenceType {
        // Analyze performance pattern based on generating noise type
        match &candidate.generating_pattern {
            NoisePattern::Quantum { .. } => {
                EmergenceType::QuantumCollapse {
                    coherence_loss: 0.8,
                    state_reduction: 0.9,
                }
            },
            
            NoisePattern::Molecular { thermal_energy, .. } => {
                EmergenceType::MolecularOptimization {
                    energy_reduction: thermal_energy * 0.1,
                    stability_increase: 0.7,
                }
            },
            
            _ => {
                // Analyze performance history to determine pattern type
                if let Some(oscillation) = self.oscillation_detector.detect(performance_history) {
                    EmergenceType::OscillatoryConvergence {
                        frequency: oscillation.frequency,
                        damping: oscillation.damping,
                        amplitude: oscillation.amplitude,
                    }
                } else if self.trend_analyzer.is_gradual_convergence(performance_history) {
                    let convergence_rate = self.trend_analyzer.calculate_rate(performance_history);
                    EmergenceType::GradualConvergence {
                        rate: convergence_rate,
                        stability: 0.8,
                    }
                } else {
                    // Default to performance spike
                    EmergenceType::PerformanceSpike {
                        magnitude: candidate.performance,
                        baseline_deviation: candidate.significance,
                    }
                }
            }
        }
    }
}

/// FFT analysis for frequency domain patterns
#[derive(Debug)]
struct FftAnalyzer {
    /// Window size for FFT analysis
    window_size: usize,
}

impl FftAnalyzer {
    fn new() -> Self {
        Self { window_size: 64 }
    }
}

/// Trend analysis for gradual convergence detection
#[derive(Debug)]
struct TrendAnalyzer {
    /// Minimum trend length
    min_trend_length: usize,
}

impl TrendAnalyzer {
    fn new() -> Self {
        Self { min_trend_length: 10 }
    }
    
    /// Check if performance shows gradual convergence
    fn is_gradual_convergence(&self, performance_history: &[f64]) -> bool {
        if performance_history.len() < self.min_trend_length {
            return false;
        }
        
        // Check for consistent upward trend in recent history
        let recent = &performance_history[performance_history.len().saturating_sub(self.min_trend_length)..];
        let mut increasing_count = 0;
        
        for window in recent.windows(2) {
            if window[1] > window[0] {
                increasing_count += 1;
            }
        }
        
        increasing_count as f64 / (recent.len() - 1) as f64 > 0.7
    }
    
    /// Calculate convergence rate
    fn calculate_rate(&self, performance_history: &[f64]) -> f64 {
        if performance_history.len() < 2 {
            return 0.0;
        }
        
        let recent = &performance_history[performance_history.len().saturating_sub(10)..];
        let start = recent[0];
        let end = recent[recent.len() - 1];
        
        (end - start) / recent.len() as f64
    }
}

/// Oscillation detection for periodic patterns
#[derive(Debug)]
struct OscillationDetector {
    /// Minimum oscillation periods to detect
    min_periods: usize,
}

impl OscillationDetector {
    fn new() -> Self {
        Self { min_periods: 3 }
    }
    
    /// Detect oscillatory patterns in performance data
    fn detect(&self, performance_history: &[f64]) -> Option<OscillationPattern> {
        if performance_history.len() < self.min_periods * 4 {
            return None;
        }
        
        // Simple peak detection for oscillation
        let recent = &performance_history[performance_history.len().saturating_sub(20)..];
        let peaks = self.find_peaks(recent);
        
        if peaks.len() >= self.min_periods {
            let frequency = self.estimate_frequency(&peaks);
            let amplitude = self.estimate_amplitude(recent, &peaks);
            let damping = self.estimate_damping(recent, &peaks);
            
            Some(OscillationPattern {
                frequency,
                amplitude,
                damping,
            })
        } else {
            None
        }
    }
    
    /// Find peaks in the data
    fn find_peaks(&self, data: &[f64]) -> Vec<usize> {
        let mut peaks = Vec::new();
        
        for i in 1..data.len()-1 {
            if data[i] > data[i-1] && data[i] > data[i+1] {
                peaks.push(i);
            }
        }
        
        peaks
    }
    
    /// Estimate oscillation frequency from peak positions
    fn estimate_frequency(&self, peaks: &[usize]) -> f64 {
        if peaks.len() < 2 {
            return 0.0;
        }
        
        let intervals: Vec<f64> = peaks.windows(2)
            .map(|w| (w[1] - w[0]) as f64)
            .collect();
        
        let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        
        if mean_interval > 0.0 {
            1.0 / mean_interval
        } else {
            0.0
        }
    }
    
    /// Estimate oscillation amplitude
    fn estimate_amplitude(&self, data: &[f64], peaks: &[usize]) -> f64 {
        if peaks.is_empty() {
            return 0.0;
        }
        
        let mean_value = data.iter().sum::<f64>() / data.len() as f64;
        let peak_values: Vec<f64> = peaks.iter().map(|&i| data[i] - mean_value).collect();
        
        peak_values.iter().sum::<f64>() / peak_values.len() as f64
    }
    
    /// Estimate damping coefficient
    fn estimate_damping(&self, data: &[f64], peaks: &[usize]) -> f64 {
        if peaks.len() < 2 {
            return 0.0;
        }
        
        let first_peak = data[peaks[0]];
        let last_peak = data[peaks[peaks.len() - 1]];
        
        if first_peak > 0.0 {
            (first_peak - last_peak) / first_peak
        } else {
            0.0
        }
    }
}

/// Detected oscillation pattern
#[derive(Debug)]
struct OscillationPattern {
    frequency: f64,
    amplitude: f64,
    damping: f64,
}

/// Convergence criteria checker
#[derive(Debug)]
pub struct ConvergenceCriteria {
    /// Statistical significance threshold
    significance_threshold: f64,
    
    /// Minimum stability duration
    stability_duration: Duration,
    
    /// Convergence tolerance
    tolerance: f64,
}

impl ConvergenceCriteria {
    /// Create new convergence criteria
    pub fn new(significance_threshold: f64) -> Self {
        Self {
            significance_threshold,
            stability_duration: Duration::from_millis(100),
            tolerance: 1e-6,
        }
    }
    
    /// Check if convergence criteria are met
    pub fn is_converged(&self, recent_values: &[f64]) -> bool {
        if recent_values.len() < 10 {
            return false;
        }
        
        // Check for stability (low variance in recent values)
        let mean = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
        let variance = recent_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / recent_values.len() as f64;
        
        variance < self.tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NoisePattern;

    #[tokio::test]
    async fn test_anomaly_detector_creation() {
        let detector = AnomalyDetector::new(3.0);
        assert_eq!(detector.detection_threshold(), 3.0);
    }
    
    #[test]
    fn test_statistical_analyzer() {
        let mut analyzer = StatisticalAnalyzer::new(100, 2.0);
        
        // Update baseline with normal distribution
        let baseline_samples: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 10.0).collect();
        analyzer.update_baseline(&baseline_samples);
        
        // Test anomaly detection
        let test_values = vec![0.0, 0.1, 5.0, 0.2]; // 5.0 should be anomalous
        let anomalies = analyzer.detect_anomalies(&test_values);
        
        assert!(anomalies.len() > 0);
        assert!(anomalies[0].significance > 2.0);
    }
    
    #[test]
    fn test_pattern_recognizer() {
        let recognizer = PatternRecognizer::new();
        
        // Test trend analysis
        let increasing_trend = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        assert!(recognizer.trend_analyzer.is_gradual_convergence(&increasing_trend));
        
        let random_data = vec![1.0, 0.5, 1.2, 0.8, 1.1, 0.9];
        assert!(!recognizer.trend_analyzer.is_gradual_convergence(&random_data));
    }
    
    #[test]
    fn test_oscillation_detector() {
        let detector = OscillationDetector::new();
        
        // Create oscillatory data
        let oscillatory_data: Vec<f64> = (0..20)
            .map(|i| (i as f64 * 0.5).sin() * 2.0 + 5.0)
            .collect();
        
        let pattern = detector.detect(&oscillatory_data);
        assert!(pattern.is_some());
        
        if let Some(p) = pattern {
            assert!(p.frequency > 0.0);
            assert!(p.amplitude > 0.0);
        }
    }
    
    #[test]
    fn test_convergence_criteria() {
        let criteria = ConvergenceCriteria::new(3.0);
        
        // Test converged data (low variance)
        let converged_data = vec![1.0001, 1.0002, 1.0001, 1.0003, 1.0001, 1.0002, 1.0001, 1.0001, 1.0002, 1.0001];
        assert!(criteria.is_converged(&converged_data));
        
        // Test non-converged data (high variance)
        let divergent_data = vec![1.0, 2.0, 0.5, 3.0, 0.1, 2.5, 0.8, 3.2, 0.3, 2.8];
        assert!(!criteria.is_converged(&divergent_data));
    }
} 