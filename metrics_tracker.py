"""
Multilingual Metrics Tracker
============================
Track and monitor per-language performance metrics.
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import json


class MultilingualMetricsTracker:
    """
    Track predictions and performance metrics per language.
    
    Example:
        >>> tracker = MultilingualMetricsTracker()
        >>> tracker.log_prediction('en', 'positive', 'positive', 45.2)
        >>> tracker.log_prediction('ru', 'positive', 'negative', 68.1)
        >>> print(tracker.get_accuracy('en'))  # 1.0
        >>> print(tracker.get_accuracy('ru'))  # 0.0
    """
    
    def __init__(self):
        """Initialize the metrics tracker."""
        self.metrics: Dict[str, Dict] = defaultdict(lambda: {
            'predictions': 0,
            'correct': 0,
            'total_latency_ms': 0,
            'latencies': [],
            'errors': [],
            'first_seen': None,
            'last_seen': None,
        })
    
    def log_prediction(
        self,
        language: str,
        predicted: Any,
        actual: Any,
        latency_ms: float,
        text_sample: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log a single prediction for tracking.
        
        Args:
            language: Language code (e.g., 'en', 'ru', 'es')
            predicted: Model's prediction
            actual: Ground truth label
            latency_ms: Inference latency in milliseconds
            text_sample: Optional sample of input text (for error analysis)
            metadata: Optional additional metadata
        """
        now = datetime.now()
        m = self.metrics[language]
        
        m['predictions'] += 1
        m['total_latency_ms'] += latency_ms
        m['latencies'].append(latency_ms)
        m['last_seen'] = now.isoformat()
        
        if m['first_seen'] is None:
            m['first_seen'] = now.isoformat()
        
        if predicted == actual:
            m['correct'] += 1
        else:
            error_record = {
                'timestamp': now.isoformat(),
                'predicted': str(predicted),
                'actual': str(actual),
                'latency_ms': latency_ms,
            }
            if text_sample:
                error_record['text_sample'] = text_sample[:200]  # Truncate
            if metadata:
                error_record['metadata'] = metadata
            
            m['errors'].append(error_record)
            
            # Keep only last 1000 errors per language
            if len(m['errors']) > 1000:
                m['errors'] = m['errors'][-1000:]
    
    def get_accuracy(self, language: str) -> float:
        """Get accuracy for a specific language."""
        m = self.metrics[language]
        if m['predictions'] == 0:
            return 0.0
        return m['correct'] / m['predictions']
    
    def get_avg_latency(self, language: str) -> float:
        """Get average latency in ms for a specific language."""
        m = self.metrics[language]
        if m['predictions'] == 0:
            return 0.0
        return m['total_latency_ms'] / m['predictions']
    
    def get_latency_percentiles(self, language: str) -> Dict[str, float]:
        """Get latency percentiles (p50, p95, p99) for a language."""
        import numpy as np
        
        m = self.metrics[language]
        if not m['latencies']:
            return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0}
        
        latencies = m['latencies']
        return {
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
        }
    
    def get_error_rate(self, language: str) -> float:
        """Get error rate for a specific language."""
        return 1.0 - self.get_accuracy(language)
    
    def get_language_report(self, language: str) -> Dict[str, Any]:
        """Get detailed report for a specific language."""
        m = self.metrics[language]
        
        return {
            'language': language,
            'predictions': m['predictions'],
            'accuracy': self.get_accuracy(language),
            'error_rate': self.get_error_rate(language),
            'avg_latency_ms': self.get_avg_latency(language),
            'latency_percentiles': self.get_latency_percentiles(language),
            'error_count': len(m['errors']),
            'recent_errors': m['errors'][-5:],  # Last 5 errors
            'first_seen': m['first_seen'],
            'last_seen': m['last_seen'],
        }
    
    def get_summary_report(self) -> List[Dict[str, Any]]:
        """Get summary report for all languages."""
        report = []
        for lang in self.metrics.keys():
            m = self.metrics[lang]
            report.append({
                'language': lang,
                'predictions': m['predictions'],
                'accuracy': round(self.get_accuracy(lang), 4),
                'avg_latency_ms': round(self.get_avg_latency(lang), 2),
                'error_count': len(m['errors']),
            })
        
        # Sort by prediction count (descending)
        return sorted(report, key=lambda x: x['predictions'], reverse=True)
    
    def check_alerts(
        self,
        accuracy_threshold: float = 0.03,
        latency_threshold_pct: float = 0.20,
        baseline_metrics: Optional[Dict[str, Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Check for alert conditions.
        
        Args:
            accuracy_threshold: Alert if accuracy drops more than this
            latency_threshold_pct: Alert if latency increases more than this %
            baseline_metrics: Baseline metrics to compare against
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        if baseline_metrics is None:
            return alerts
        
        for lang in self.metrics.keys():
            if lang not in baseline_metrics:
                continue
            
            current_acc = self.get_accuracy(lang)
            baseline_acc = baseline_metrics[lang].get('accuracy', current_acc)
            
            if baseline_acc - current_acc > accuracy_threshold:
                alerts.append({
                    'type': 'accuracy_drop',
                    'language': lang,
                    'severity': 'high',
                    'message': f"{lang}: Accuracy dropped from {baseline_acc:.1%} to {current_acc:.1%}",
                    'baseline': baseline_acc,
                    'current': current_acc,
                })
            
            current_latency = self.get_avg_latency(lang)
            baseline_latency = baseline_metrics[lang].get('avg_latency_ms', current_latency)
            
            if baseline_latency > 0:
                latency_increase = (current_latency - baseline_latency) / baseline_latency
                if latency_increase > latency_threshold_pct:
                    alerts.append({
                        'type': 'latency_increase',
                        'language': lang,
                        'severity': 'medium',
                        'message': f"{lang}: Latency increased by {latency_increase:.1%}",
                        'baseline': baseline_latency,
                        'current': current_latency,
                    })
        
        return alerts
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        data = {
            'exported_at': datetime.now().isoformat(),
            'languages': {}
        }
        
        for lang in self.metrics.keys():
            data['languages'][lang] = self.get_language_report(lang)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics exported to {filepath}")


# Example usage
if __name__ == "__main__":
    print("Multilingual Metrics Tracker Example")
    print("=" * 40)
    
    tracker = MultilingualMetricsTracker()
    
    # Simulate some predictions
    import random
    
    languages = ['en', 'es', 'ru', 'de', 'fr']
    accuracy_rates = {'en': 0.90, 'es': 0.85, 'ru': 0.75, 'de': 0.88, 'fr': 0.82}
    latency_bases = {'en': 40, 'es': 45, 'ru': 65, 'de': 48, 'fr': 50}
    
    for _ in range(1000):
        lang = random.choice(languages)
        actual = random.choice(['positive', 'negative', 'neutral'])
        
        # Simulate accuracy based on language
        if random.random() < accuracy_rates[lang]:
            predicted = actual
        else:
            predicted = random.choice(['positive', 'negative', 'neutral'])
        
        # Simulate latency
        latency = latency_bases[lang] + random.gauss(0, 10)
        
        tracker.log_prediction(lang, predicted, actual, max(latency, 1))
    
    # Print summary
    print("\nSummary Report:")
    for row in tracker.get_summary_report():
        print(f"  {row['language']}: {row['predictions']} predictions, "
              f"{row['accuracy']:.1%} accuracy, {row['avg_latency_ms']:.1f}ms avg latency")
    
    # Check for alerts against baseline
    baseline = {
        'ru': {'accuracy': 0.85, 'avg_latency_ms': 50}  # Higher baseline = alert
    }
    
    alerts = tracker.check_alerts(baseline_metrics=baseline)
    if alerts:
        print("\n⚠️ Alerts:")
        for alert in alerts:
            print(f"  [{alert['severity'].upper()}] {alert['message']}")
