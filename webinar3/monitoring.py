"""
Production Monitoring Setup
===========================
Monitor multilingual model performance in production.

Part of Webinar 3: Optimizing Multilingual NLP Models
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class MultilingualMonitor:
    """
    Production monitoring for multilingual models.
    
    Tracks:
    - Per-language accuracy over time
    - Latency by language
    - Prediction confidence distribution
    - Drift detection
    """
    
    def __init__(self, languages: List[str], baseline_metrics: Dict[str, float]):
        self.languages = languages
        self.baseline = baseline_metrics
        self.metrics_history = {lang: [] for lang in languages}
        self.alerts = []
    
    def log_prediction(
        self,
        language: str,
        prediction: int,
        confidence: float,
        latency_ms: float,
        ground_truth: Optional[int] = None
    ):
        """Log a single prediction for monitoring."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'latency_ms': latency_ms,
            'correct': None if ground_truth is None else (prediction == ground_truth)
        }
        
        self.metrics_history[language].append(record)
        
        # Check for anomalies
        self._check_latency_anomaly(language, latency_ms)
        self._check_confidence_anomaly(language, confidence)
    
    def _check_latency_anomaly(self, language: str, latency_ms: float):
        """Alert on latency spikes."""
        baseline_latency = self.baseline.get(f'{language}_latency_p95', 100)
        if latency_ms > baseline_latency * 2:
            self.alerts.append({
                'type': 'latency_spike',
                'language': language,
                'value': latency_ms,
                'threshold': baseline_latency * 2,
                'timestamp': datetime.now().isoformat()
            })
    
    def _check_confidence_anomaly(self, language: str, confidence: float):
        """Alert on unusual confidence patterns (potential drift)."""
        if confidence < 0.5:
            recent = self.metrics_history[language][-100:]
            low_conf_rate = sum(1 for r in recent if r['confidence'] < 0.5) / len(recent) if recent else 0
            
            if low_conf_rate > 0.2:  # More than 20% low confidence
                self.alerts.append({
                    'type': 'confidence_drift',
                    'language': language,
                    'low_confidence_rate': low_conf_rate,
                    'timestamp': datetime.now().isoformat()
                })
    
    def calculate_rolling_accuracy(
        self,
        language: str,
        window_size: int = 1000
    ) -> Optional[float]:
        """Calculate rolling accuracy for a language."""
        recent = self.metrics_history[language][-window_size:]
        labeled = [r for r in recent if r['correct'] is not None]
        
        if not labeled:
            return None
        
        return sum(r['correct'] for r in labeled) / len(labeled)
    
    def check_accuracy_drift(self, language: str, threshold: float = 0.03) -> bool:
        """
        Check if accuracy has drifted from baseline.
        
        Returns True if drift detected.
        """
        current_accuracy = self.calculate_rolling_accuracy(language)
        
        if current_accuracy is None:
            return False
        
        baseline_accuracy = self.baseline.get(f'{language}_accuracy', 0.85)
        drift = baseline_accuracy - current_accuracy
        
        if drift > threshold:
            self.alerts.append({
                'type': 'accuracy_drift',
                'language': language,
                'baseline': baseline_accuracy,
                'current': current_accuracy,
                'drift': drift,
                'timestamp': datetime.now().isoformat()
            })
            return True
        
        return False
    
    def get_dashboard_metrics(self) -> Dict:
        """Get metrics formatted for a monitoring dashboard."""
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'languages': {}
        }
        
        for lang in self.languages:
            accuracy = self.calculate_rolling_accuracy(lang)
            recent = self.metrics_history[lang][-100:]
            
            dashboard['languages'][lang] = {
                'accuracy': accuracy,
                'avg_latency_ms': np.mean([r['latency_ms'] for r in recent]) if recent else None,
                'avg_confidence': np.mean([r['confidence'] for r in recent]) if recent else None,
                'prediction_count': len(self.metrics_history[lang]),
                'baseline_accuracy': self.baseline.get(f'{lang}_accuracy'),
                'status': 'healthy' if accuracy and accuracy > 0.80 else 'warning'
            }
        
        dashboard['active_alerts'] = self.alerts[-10:]
        
        return dashboard
    
    def clear_old_history(self, max_records: int = 10000):
        """Clear old records to prevent memory growth."""
        for lang in self.languages:
            if len(self.metrics_history[lang]) > max_records:
                self.metrics_history[lang] = self.metrics_history[lang][-max_records:]
