"""
Feature Extraction Utilities
============================
Extract language-agnostic and script-based features.

Part of Webinar 2: Building Robust Data Pipelines for Multilingual Product Classification
"""

import re
from typing import Dict, Any

from .script_detection import detect_script


def extract_language_agnostic_features(text: str) -> Dict[str, Any]:
    """
    Extract features that work across all languages.
    
    These features don't require language-specific processing
    and provide baseline signal for any language.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Length features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = (
        features['char_count'] / features['word_count'] 
        if features['word_count'] > 0 else 0
    )
    
    # Character type ratios
    features['digit_ratio'] = sum(c.isdigit() for c in text) / max(len(text), 1)
    features['upper_ratio'] = sum(c.isupper() for c in text) / max(len(text), 1)
    features['space_ratio'] = sum(c.isspace() for c in text) / max(len(text), 1)
    
    # Punctuation features
    punctuation = set('.,!?;:()[]{}"\'-')
    features['punctuation_count'] = sum(c in punctuation for c in text)
    features['punctuation_ratio'] = features['punctuation_count'] / max(len(text), 1)
    
    # Special patterns
    features['has_url'] = bool(re.search(r'https?://|www\.', text))
    features['has_email'] = bool(re.search(r'\S+@\S+\.\S+', text))
    features['has_number'] = bool(re.search(r'\d+', text))
    
    # Structural features
    features['line_count'] = text.count('\n') + 1
    features['has_list_markers'] = bool(re.search(r'^\s*[-•*]\s', text, re.MULTILINE))
    
    return features


def extract_script_features(text: str) -> Dict[str, float]:
    """
    Extract features based on writing scripts present.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with script proportions
    """
    distribution = detect_script(text)
    
    features = {}
    for script in ['latin', 'cyrillic', 'arabic', 'chinese', 'thai']:
        features[f'script_{script}_ratio'] = distribution.get(script, 0.0)
    
    features['script_count'] = len([v for v in distribution.values() if v > 0.1])
    features['is_mixed_script'] = features['script_count'] > 1
    
    return features
