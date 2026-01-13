"""
Per-Language Evaluation Framework
=================================
Evaluate model performance broken down by language.

Part of Webinar 3: Optimizing Multilingual NLP Models
"""

import time
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LanguageMetrics:
    """Metrics for a single language."""
    language: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    latency_p50: float
    latency_p95: float
    sample_count: int
    error_rate: float
    
    def meets_threshold(self, min_accuracy: float = 0.80) -> bool:
        return self.accuracy >= min_accuracy


class PerLanguageEvaluator:
    """
    Comprehensive per-language evaluation with business metric integration.
    
    Key features:
    - Per-language accuracy, precision, recall, F1
    - Latency percentiles per language
    - Error cost calculation
    - Threshold violation detection
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        language_thresholds: Dict[str, float] = None,
        error_costs: Dict[str, float] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.language_thresholds = language_thresholds or {}
        self.error_costs = error_costs or {}
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[int],
        languages: List[str],
        predictions: Optional[List[int]] = None
    ) -> Dict[str, LanguageMetrics]:
        """
        Evaluate model performance per language.
        
        Returns dict mapping language code to LanguageMetrics.
        """
        from sklearn.metrics import precision_recall_fscore_support
        
        results = {}
        
        for lang in set(languages):
            # Filter data for this language
            mask = [l == lang for l in languages]
            lang_texts = [t for t, m in zip(texts, mask) if m]
            lang_labels = [l for l, m in zip(labels, mask) if m]
            
            if predictions:
                lang_preds = [p for p, m in zip(predictions, mask) if m]
            else:
                lang_preds = self._get_predictions(lang_texts)
            
            # Calculate metrics
            accuracy = sum(p == l for p, l in zip(lang_preds, lang_labels)) / len(lang_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(
                lang_labels, lang_preds, average='weighted', zero_division=0
            )
            
            # Measure latency
            latency = self._measure_latency(lang_texts[:50])
            
            results[lang] = LanguageMetrics(
                language=lang,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                latency_p50=latency['p50'],
                latency_p95=latency['p95'],
                sample_count=len(lang_labels),
                error_rate=1 - accuracy
            )
        
        return results
    
    def _get_predictions(self, texts: List[str]) -> List[int]:
        """Get model predictions for texts."""
        predictions = []
        self.model.eval()
        
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = outputs.logits.argmax(dim=-1).item()
            predictions.append(pred)
        
        return predictions
    
    def _measure_latency(self, texts: List[str]) -> Dict[str, float]:
        """Measure inference latency."""
        latencies = []
        
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(**inputs)
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            'p50': np.percentile(latencies, 50) if latencies else 0,
            'p95': np.percentile(latencies, 95) if latencies else 0
        }
    
    def calculate_error_weighted_cost(
        self,
        metrics: Dict[str, LanguageMetrics],
        traffic_distribution: Dict[str, float],
        error_costs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate error-weighted cost per language.
        
        Formula: traffic_share * error_rate * error_cost
        
        This helps identify which languages have the highest business impact.
        """
        costs = {}
        
        for lang, lang_metrics in metrics.items():
            traffic = traffic_distribution.get(lang, 0.0)
            error_cost = error_costs.get(lang, 1.0)
            costs[lang] = traffic * lang_metrics.error_rate * error_cost
        
        return costs
    
    def check_thresholds(
        self,
        metrics: Dict[str, LanguageMetrics]
    ) -> Dict[str, bool]:
        """Check if each language meets its accuracy threshold."""
        violations = {}
        
        for lang, lang_metrics in metrics.items():
            threshold = self.language_thresholds.get(lang, 0.80)
            violations[lang] = lang_metrics.accuracy < threshold
        
        return violations
    
    def generate_report(
        self,
        metrics: Dict[str, LanguageMetrics],
        traffic_distribution: Dict[str, float] = None
    ) -> str:
        """Generate a human-readable evaluation report."""
        report = ["=" * 60]
        report.append("MULTILINGUAL MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        # Overall metrics
        total_samples = sum(m.sample_count for m in metrics.values())
        weighted_accuracy = sum(
            m.accuracy * m.sample_count for m in metrics.values()
        ) / total_samples
        
        report.append(f"\nOverall Accuracy: {weighted_accuracy:.2%}")
        report.append(f"Total Samples: {total_samples:,}")
        
        # Per-language breakdown
        report.append("\n" + "-" * 60)
        report.append("PER-LANGUAGE BREAKDOWN")
        report.append("-" * 60)
        
        for lang, m in sorted(metrics.items(), key=lambda x: -x[1].accuracy):
            threshold = self.language_thresholds.get(lang, 0.80)
            status = "OK" if m.accuracy >= threshold else "BELOW THRESHOLD"
            
            report.append(f"\n{lang.upper()} [{status}]")
            report.append(f"  Accuracy:  {m.accuracy:.2%} (threshold: {threshold:.0%})")
            report.append(f"  F1 Score:  {m.f1:.3f}")
            report.append(f"  Latency:   p50={m.latency_p50:.1f}ms, p95={m.latency_p95:.1f}ms")
            report.append(f"  Samples:   {m.sample_count:,}")
        
        return "\n".join(report)
