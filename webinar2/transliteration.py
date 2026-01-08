"""
Transliteration Detection Utilities
====================================
Detect transliterated text (e.g., Russian in Latin script).

Part of Webinar 2: Building Robust Data Pipelines for Multilingual Product Classification
"""

from typing import Dict, Any


# Common transliteration patterns for Russian
RUSSIAN_TRANSLIT_PATTERNS = {
    'sh': 'ш',
    'ch': 'ч',
    'zh': 'ж',
    'ya': 'я',
    'yu': 'ю',
    'yo': 'ё',
    'ts': 'ц',
    'sch': 'щ',
}

# Common Latin words that appear in Russian (not transliteration)
COMMON_ENGLISH_IN_RUSSIAN = {
    'iphone', 'ipad', 'macbook', 'samsung', 'android', 'windows',
    'facebook', 'google', 'youtube', 'instagram', 'twitter',
    'email', 'wifi', 'bluetooth', 'usb', 'hdmi', 'led', 'lcd',
    'pro', 'max', 'mini', 'plus', 'lite', 'ultra',
}


def is_likely_transliteration(text: str, expected_language: str = 'ru') -> Dict[str, Any]:
    """
    Detect if Latin text is likely transliterated from another language.
    
    Args:
        text: Input text (should be primarily Latin script)
        expected_language: The language it might be transliterated from
        
    Returns:
        Dict with detection results
        
    Example:
        >>> is_likely_transliteration("privet kak dela", "ru")
        {'is_transliteration': True, 'confidence': 0.8, ...}
    """
    text_lower = text.lower()
    words = text_lower.split()
    
    # Skip if too short
    if len(words) < 2:
        return {
            'is_transliteration': False,
            'confidence': 0.0,
            'reason': 'text_too_short'
        }
    
    # Check for common English words (not transliteration)
    english_word_count = sum(1 for word in words if word in COMMON_ENGLISH_IN_RUSSIAN)
    if english_word_count / len(words) > 0.5:
        return {
            'is_transliteration': False,
            'confidence': 0.9,
            'reason': 'mostly_english_words'
        }
    
    # Check for transliteration patterns
    if expected_language == 'ru':
        pattern_matches = 0
        for pattern in RUSSIAN_TRANSLIT_PATTERNS:
            if pattern in text_lower:
                pattern_matches += 1
        
        # Check for characteristic Russian word endings in translit
        russian_endings = ['ov', 'ev', 'iy', 'yy', 'aya', 'oye', 'iye']
        ending_matches = sum(1 for word in words 
                           if any(word.endswith(e) for e in russian_endings))
        
        confidence = min(1.0, (pattern_matches * 0.2) + (ending_matches / len(words)))
        
        return {
            'is_transliteration': confidence > 0.3,
            'confidence': confidence,
            'pattern_matches': pattern_matches,
            'ending_matches': ending_matches,
            'reason': 'pattern_analysis'
        }
    
    return {
        'is_transliteration': False,
        'confidence': 0.0,
        'reason': 'language_not_supported'
    }
