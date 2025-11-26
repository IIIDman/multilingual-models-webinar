"""
Data Drift Detection
====================
Monitor for distribution shifts between training and production data.
"""

from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np

try:
    from scipy.stats import wasserstein_distance, ks_2samp
except ImportError:
    wasserstein_distance = None
    ks_2samp = None
    print("Warning: scipy not installed. Run: pip install scipy")


class DriftDetector:
    """
    Detect data drift between baseline (training) and current (production) data.
    
    Example:
        >>> detector = DriftDetector()
        >>> detector.set_baseline(training_texts)
        >>> drift_report = detector.check_drift(production_texts)
        >>> if drift_report['drift_detected']:
        ...     print("Retraining recommended!")
    """
    
    def __init__(self, drift_threshold: float = 0.15):
        """
        Initialize drift detector.
        
        Args:
            drift_threshold: Wasserstein distance threshold for drift alert
        """
        self.drift_threshold = drift_threshold
        self.baseline_stats: Optional[Dict] = None
    
    def _compute_stats(self, texts: List[str]) -> Dict[str, Any]:
        """Compute statistical features from texts."""
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        # Character type distributions
        has_numbers = [any(c.isdigit() for c in text) for text in texts]
        has_special = [any(not c.isalnum() and not c.isspace() for c in text) for text in texts]
        
        return {
            'lengths': np.array(lengths),
            'word_counts': np.array(word_counts),
            'pct_has_numbers': np.mean(has_numbers),
            'pct_has_special': np.mean(has_special),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
        }
    
    def set_baseline(self, texts: List[str]) -> None:
        """
        Set baseline statistics from training data.
        
        Args:
            texts: List of training texts
        """
        self.baseline_stats = self._compute_stats(texts)
        print(f"Baseline set with {len(texts)} texts")
        print(f"  Mean length: {self.baseline_stats['mean_length']:.1f}")
        print(f"  % with numbers: {self.baseline_stats['pct_has_numbers']:.1%}")
    
    def check_drift(self, texts: List[str]) -> Dict[str, Any]:
        """
        Check for drift against baseline.
        
        Args:
            texts: List of current/production texts
            
        Returns:
            Dictionary with drift metrics and recommendations
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")
        
        if wasserstein_distance is None:
            raise ImportError("scipy is required. Install with: pip install scipy")
        
        current_stats = self._compute_stats(texts)
        
        # Calculate Wasserstein distance for length distribution
        length_drift = wasserstein_distance(
            self.baseline_stats['lengths'],
            current_stats['lengths']
        )
        
        # Normalize by baseline mean for interpretability
        normalized_drift = length_drift / (self.baseline_stats['mean_length'] + 1e-6)
        
        # Check other metrics
        number_pct_change = abs(
            current_stats['pct_has_numbers'] - self.baseline_stats['pct_has_numbers']
        )
        
        drift_detected = normalized_drift > self.drift_threshold
        
        report = {
            'drift_detected': drift_detected,
            'drift_score': normalized_drift,
            'threshold': self.drift_threshold,
            'metrics': {
                'length_drift': length_drift,
                'normalized_drift': normalized_drift,
                'baseline_mean_length': self.baseline_stats['mean_length'],
                'current_mean_length': current_stats['mean_length'],
                'number_pct_change': number_pct_change,
            },
            'recommendation': self._get_recommendation(normalized_drift, number_pct_change)
        }
        
        return report
    
    def _get_recommendation(self, drift_score: float, number_change: float) -> str:
        """Generate recommendation based on drift metrics."""
        if drift_score > self.drift_threshold * 2:
            return "CRITICAL: Significant drift detected. Immediate retraining recommended."
        elif drift_score > self.drift_threshold:
            return "WARNING: Moderate drift detected. Schedule retraining soon."
        elif number_change > 0.1:
            return "NOTICE: Input characteristics changed. Monitor closely."
        else:
            return "OK: No significant drift detected."


def calculate_wasserstein_drift(
    baseline_values: List[float], 
    current_values: List[float]
) -> float:
    """
    Calculate Wasserstein distance between two distributions.
    
    Higher values indicate more drift.
    
    Args:
        baseline_values: Values from training/baseline period
        current_values: Values from current/production period
        
    Returns:
        Wasserstein distance (Earth Mover's Distance)
    """
    if wasserstein_distance is None:
        raise ImportError("scipy is required. Install with: pip install scipy")
    
    return wasserstein_distance(baseline_values, current_values)


def perform_ks_test(
    baseline_values: List[float], 
    current_values: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform Kolmogorov-Smirnov test for distribution difference.
    
    Args:
        baseline_values: Values from training/baseline period
        current_values: Values from current/production period
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    if ks_2samp is None:
        raise ImportError("scipy is required. Install with: pip install scipy")
    
    statistic, p_value = ks_2samp(baseline_values, current_values)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant_drift': p_value < alpha,
        'alpha': alpha
    }


# Example usage
if __name__ == "__main__":
    print("Data Drift Detection Example")
    print("=" * 40)
    
    # Simulate baseline data (training)
    np.random.seed(42)
    baseline_texts = [
        "This is a sample product description " * np.random.randint(1, 5)
        for _ in range(1000)
    ]
    
    # Simulate current data (production) - slightly different distribution
    current_texts = [
        "Short desc " * np.random.randint(1, 3)  # Shorter on average
        for _ in range(500)
    ]
    
    # Detect drift
    detector = DriftDetector(drift_threshold=0.15)
    detector.set_baseline(baseline_texts)
    
    report = detector.check_drift(current_texts)
    
    print(f"\nDrift Report:")
    print(f"  Drift detected: {report['drift_detected']}")
    print(f"  Drift score: {report['drift_score']:.3f} (threshold: {report['threshold']})")
    print(f"  {report['recommendation']}")
