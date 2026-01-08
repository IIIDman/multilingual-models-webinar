"""
Monitoring Metrics Utilities
============================
Compute and monitor per-language metrics.

Part of Webinar 2: Building Robust Data Pipelines for Multilingual Product Classification
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

from .encoding import QualityStatus, check_encoding
from .script_detection import detect_mixed_language_content
from .quality_checker import MultilingualQualityChecker


@dataclass
class LanguageMetrics:
    """Metrics for a single language."""
    language: str
    volume: int
    encoding_error_rate: float
    avg_text_length: float
    mixed_content_rate: float
    avg_quality_score: float
    timestamp: str


def compute_language_metrics(
    texts: List[Dict[str, Any]], 
    language: str
) -> LanguageMetrics:
    """
    Compute aggregate metrics for a batch of texts in one language.
    
    Args:
        texts: List of dicts with 'text' and optional metadata
        language: Language code
        
    Returns:
        LanguageMetrics object
        
    Example:
        >>> texts = [{'text': 'Hello'}, {'text': 'World'}]
        >>> metrics = compute_language_metrics(texts, 'en')
    """
    checker = MultilingualQualityChecker()
    
    encoding_errors = 0
    total_length = 0
    mixed_content_count = 0
    quality_scores = []
    
    for item in texts:
        text = item.get('text', '')
        
        # Encoding check
        status, _ = check_encoding(text)
        if status == QualityStatus.FAIL:
            encoding_errors += 1
        
        # Length
        total_length += len(text)
        
        # Mixed content
        mixed = detect_mixed_language_content(text)
        if mixed['is_mixed']:
            mixed_content_count += 1
        
        # Overall quality
        results = checker.run_all_checks(text, language)
        overall = checker.get_overall_status(results)
        quality_scores.append(1.0 if overall == QualityStatus.PASS else 0.5 if overall == QualityStatus.WARNING else 0.0)
    
    n = len(texts)
    
    return LanguageMetrics(
        language=language,
        volume=n,
        encoding_error_rate=encoding_errors / max(n, 1),
        avg_text_length=total_length / max(n, 1),
        mixed_content_rate=mixed_content_count / max(n, 1),
        avg_quality_score=sum(quality_scores) / max(len(quality_scores), 1),
        timestamp=datetime.utcnow().isoformat()
    )


def check_for_anomalies(
    current: LanguageMetrics, 
    baseline: LanguageMetrics,
    thresholds: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    Check if current metrics deviate significantly from baseline.
    
    Args:
        current: Current period metrics
        baseline: Historical baseline metrics
        thresholds: Custom thresholds for anomaly detection
        
    Returns:
        List of detected anomalies
    """
    if thresholds is None:
        thresholds = {
            'encoding_error_rate': 1.5,  # 50% increase
            'volume': 0.3,  # 30% deviation
            'avg_text_length': 0.2,  # 20% deviation
            'mixed_content_rate': 1.5,  # 50% increase
        }
    
    anomalies = []
    
    # Encoding error rate
    if current.encoding_error_rate > baseline.encoding_error_rate * thresholds['encoding_error_rate']:
        anomalies.append({
            'metric': 'encoding_error_rate',
            'severity': 'high',
            'current': current.encoding_error_rate,
            'baseline': baseline.encoding_error_rate,
            'message': f"Encoding errors increased from {baseline.encoding_error_rate:.2%} to {current.encoding_error_rate:.2%}"
        })
    
    # Volume deviation
    if baseline.volume > 0:
        volume_change = abs(current.volume - baseline.volume) / baseline.volume
        if volume_change > thresholds['volume']:
            anomalies.append({
                'metric': 'volume',
                'severity': 'medium',
                'current': current.volume,
                'baseline': baseline.volume,
                'message': f"Volume changed by {volume_change:.1%} (from {baseline.volume} to {current.volume})"
            })
    
    # Average length deviation
    if baseline.avg_text_length > 0:
        length_change = abs(current.avg_text_length - baseline.avg_text_length) / baseline.avg_text_length
        if length_change > thresholds['avg_text_length']:
            anomalies.append({
                'metric': 'avg_text_length',
                'severity': 'low',
                'current': current.avg_text_length,
                'baseline': baseline.avg_text_length,
                'message': f"Average text length changed significantly"
            })
    
    return anomalies
