"""
Script Detection Utilities
==========================
Detect writing scripts and mixed-language content.

Part of Webinar 2: Building Robust Data Pipelines for Multilingual Product Classification
"""

from typing import Dict, Tuple, Optional, Any
from collections import defaultdict
from .encoding import QualityStatus


# Character ranges for script detection
SCRIPT_RANGES = {
    'latin': [(0x0041, 0x007A), (0x00C0, 0x00FF)],
    'cyrillic': [(0x0400, 0x04FF)],
    'arabic': [(0x0600, 0x06FF), (0x0750, 0x077F)],
    'chinese': [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    'japanese_hiragana': [(0x3040, 0x309F)],
    'japanese_katakana': [(0x30A0, 0x30FF)],
    'thai': [(0x0E00, 0x0E7F)],
    'hebrew': [(0x0590, 0x05FF)],
    'devanagari': [(0x0900, 0x097F)],  # Hindi, Sanskrit
}


def detect_script(text: str) -> Dict[str, float]:
    """
    Detect the writing scripts present in text.
    
    Returns distribution of scripts by character count.
    Useful for detecting mixed-script content.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary mapping script names to proportions
        
    Example:
        >>> detect_script("Hello Привет")
        {'latin': 0.5, 'cyrillic': 0.5}
    """
    script_counts = defaultdict(int)
    total_chars = 0
    
    for char in text:
        if char.isspace() or char.isdigit():
            continue
            
        code_point = ord(char)
        total_chars += 1
        
        found = False
        for script_name, ranges in SCRIPT_RANGES.items():
            for start, end in ranges:
                if start <= code_point <= end:
                    script_counts[script_name] += 1
                    found = True
                    break
            if found:
                break
        
        if not found:
            script_counts['other'] += 1
    
    if total_chars == 0:
        return {}
    
    return {script: count / total_chars 
            for script, count in script_counts.items()}


def detect_mixed_language_content(text: str, threshold: float = 0.1) -> Dict[str, Any]:
    """
    Detect if text contains significant mixed-language content.
    
    Args:
        text: Input text
        threshold: Minimum proportion to consider a script "significant"
        
    Returns:
        Dict with:
        - is_mixed: bool indicating mixed content
        - scripts: list of significant scripts
        - distribution: full script distribution
        
    Example:
        >>> detect_mixed_language_content("Нужен iPhone 15 Pro")
        {'is_mixed': True, 'scripts': ['cyrillic', 'latin'], ...}
    """
    distribution = detect_script(text)
    significant_scripts = [
        script for script, proportion in distribution.items()
        if proportion >= threshold and script != 'other'
    ]
    
    return {
        'is_mixed': len(significant_scripts) > 1,
        'scripts': significant_scripts,
        'distribution': distribution,
        'primary_script': max(distribution, key=distribution.get) if distribution else None
    }


def check_script_consistency(text: str, expected_language: str) -> Tuple[QualityStatus, Optional[str]]:
    """
    Check if text uses the expected script for a given language.
    
    Args:
        text: Input text
        expected_language: ISO language code (e.g., 'ru', 'en', 'zh')
        
    Returns:
        Tuple of (status, reason)
    """
    expected_scripts = {
        'ru': 'cyrillic',
        'uk': 'cyrillic',  # Ukrainian
        'bg': 'cyrillic',  # Bulgarian
        'en': 'latin',
        'es': 'latin',
        'de': 'latin',
        'fr': 'latin',
        'zh': 'chinese',
        'ja': ['japanese_hiragana', 'japanese_katakana', 'chinese'],
        'ar': 'arabic',
        'he': 'hebrew',
        'th': 'thai',
        'hi': 'devanagari',
    }
    
    if expected_language not in expected_scripts:
        return QualityStatus.PASS, None
    
    distribution = detect_script(text)
    expected = expected_scripts[expected_language]
    
    if isinstance(expected, list):
        # For Japanese, accept any of the valid scripts
        if any(script in distribution for script in expected):
            return QualityStatus.PASS, None
    else:
        if expected in distribution and distribution[expected] > 0.5:
            return QualityStatus.PASS, None
    
    # Check if it might be transliteration
    if expected in ['cyrillic', 'arabic', 'hebrew', 'chinese', 'thai']:
        if 'latin' in distribution and distribution.get('latin', 0) > 0.7:
            return QualityStatus.WARNING, "possible_transliteration"
    
    return QualityStatus.WARNING, f"unexpected_script_for_{expected_language}"
