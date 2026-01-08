"""
Quality Checks Framework
========================
Comprehensive quality validation for multilingual text.

Part of Webinar 2: Building Robust Data Pipelines for Multilingual Product Classification
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .encoding import QualityStatus, check_encoding
from .script_detection import detect_script, detect_mixed_language_content, check_script_consistency


@dataclass
class QualityCheckResult:
    """Result of a single quality check."""
    check_name: str
    status: QualityStatus
    reason: Optional[str]
    details: Optional[Dict[str, Any]] = None


class MultilingualQualityChecker:
    """
    Comprehensive quality checker for multilingual text data.
    
    Example:
        >>> checker = MultilingualQualityChecker()
        >>> results = checker.run_all_checks("Привет мир!", "ru")
        >>> for r in results:
        ...     print(f"{r.check_name}: {r.status.value}")
    """
    
    def __init__(self, 
                 min_length: int = 3,
                 max_length: int = 10000,
                 max_unknown_token_ratio: float = 0.3):
        self.min_length = min_length
        self.max_length = max_length
        self.max_unknown_token_ratio = max_unknown_token_ratio
    
    def check_encoding_quality(self, text: str) -> QualityCheckResult:
        """Check for encoding issues."""
        status, reason = check_encoding(text)
        return QualityCheckResult(
            check_name="encoding",
            status=status,
            reason=reason
        )
    
    def check_length_bounds(self, text: str) -> QualityCheckResult:
        """Check text length is within acceptable bounds."""
        length = len(text)
        
        if length < self.min_length:
            return QualityCheckResult(
                check_name="length",
                status=QualityStatus.FAIL,
                reason="too_short",
                details={"length": length, "min": self.min_length}
            )
        
        if length > self.max_length:
            return QualityCheckResult(
                check_name="length",
                status=QualityStatus.WARNING,
                reason="too_long",
                details={"length": length, "max": self.max_length}
            )
        
        return QualityCheckResult(
            check_name="length",
            status=QualityStatus.PASS,
            reason=None
        )
    
    def check_language_consistency(self, 
                                   text: str, 
                                   expected_language: str) -> QualityCheckResult:
        """Check if detected language matches expected."""
        status, reason = check_script_consistency(text, expected_language)
        
        return QualityCheckResult(
            check_name="language_consistency",
            status=status,
            reason=reason,
            details={"expected": expected_language, "distribution": detect_script(text)}
        )
    
    def check_mixed_content(self, text: str) -> QualityCheckResult:
        """Check for mixed-language content."""
        result = detect_mixed_language_content(text)
        
        if result['is_mixed']:
            return QualityCheckResult(
                check_name="mixed_content",
                status=QualityStatus.WARNING,
                reason="mixed_language_detected",
                details=result
            )
        
        return QualityCheckResult(
            check_name="mixed_content",
            status=QualityStatus.PASS,
            reason=None
        )
    
    def run_all_checks(self, 
                       text: str, 
                       expected_language: str = 'en') -> List[QualityCheckResult]:
        """
        Run all quality checks on text.
        
        Args:
            text: Input text
            expected_language: Expected language code
            
        Returns:
            List of QualityCheckResult objects
        """
        results = [
            self.check_encoding_quality(text),
            self.check_length_bounds(text),
            self.check_language_consistency(text, expected_language),
            self.check_mixed_content(text),
        ]
        
        return results
    
    def get_overall_status(self, results: List[QualityCheckResult]) -> QualityStatus:
        """
        Get overall status from list of check results.
        
        Returns FAIL if any check failed, WARNING if any warned, PASS otherwise.
        """
        statuses = [r.status for r in results]
        
        if QualityStatus.FAIL in statuses:
            return QualityStatus.FAIL
        if QualityStatus.WARNING in statuses:
            return QualityStatus.WARNING
        return QualityStatus.PASS
