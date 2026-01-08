"""
Multilingual Data Pipeline Utilities
====================================
Code examples from Webinar 2: Building Robust Data Pipelines 
for Multilingual Product Classification

Author: Dmitriy Tsarev

These utilities demonstrate best practices for building production-ready
multilingual NLP data pipelines.

Modules:
    encoding - Encoding validation and repair
    script_detection - Script/language detection
    normalization - Language-aware text normalization
    mixed_language - Mixed-language content handling
    transliteration - Transliteration detection
    quality_checker - Quality validation framework
    features - Feature extraction
    monitoring - Metrics and anomaly detection
"""

from .encoding import (
    QualityStatus,
    check_encoding,
    detect_encoding,
    fix_common_encoding_issues,
)

from .script_detection import (
    SCRIPT_RANGES,
    detect_script,
    detect_mixed_language_content,
    check_script_consistency,
)

from .normalization import (
    NormalizationConfig,
    NORMALIZATION_CONFIGS,
    turkish_lowercase,
    remove_diacritics,
    normalize_text,
)

from .mixed_language import (
    segment_by_script,
    normalize_mixed_language,
)

from .transliteration import (
    RUSSIAN_TRANSLIT_PATTERNS,
    COMMON_ENGLISH_IN_RUSSIAN,
    is_likely_transliteration,
)

from .quality_checker import (
    QualityCheckResult,
    MultilingualQualityChecker,
)

from .features import (
    extract_language_agnostic_features,
    extract_script_features,
)

from .monitoring import (
    LanguageMetrics,
    compute_language_metrics,
    check_for_anomalies,
)

__version__ = "1.0.0"
__author__ = "Dmitriy Tsarev"

__all__ = [
    # Encoding
    "QualityStatus",
    "check_encoding",
    "detect_encoding",
    "fix_common_encoding_issues",
    
    # Script detection
    "SCRIPT_RANGES",
    "detect_script",
    "detect_mixed_language_content",
    "check_script_consistency",
    
    # Normalization
    "NormalizationConfig",
    "NORMALIZATION_CONFIGS",
    "turkish_lowercase",
    "remove_diacritics",
    "normalize_text",
    
    # Mixed language
    "segment_by_script",
    "normalize_mixed_language",
    
    # Transliteration
    "RUSSIAN_TRANSLIT_PATTERNS",
    "COMMON_ENGLISH_IN_RUSSIAN",
    "is_likely_transliteration",
    
    # Quality checker
    "QualityCheckResult",
    "MultilingualQualityChecker",
    
    # Features
    "extract_language_agnostic_features",
    "extract_script_features",
    
    # Monitoring
    "LanguageMetrics",
    "compute_language_metrics",
    "check_for_anomalies",
]
