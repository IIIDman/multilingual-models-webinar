"""
Mixed-Language Handling Utilities
=================================
Process text containing multiple languages/scripts.

Part of Webinar 2: Building Robust Data Pipelines for Multilingual Product Classification
"""

from typing import Dict, List
from .script_detection import SCRIPT_RANGES
from .normalization import normalize_text


def segment_by_script(text: str) -> List[Dict[str, str]]:
    """
    Segment text into chunks by writing script.
    
    Useful for applying different preprocessing to different
    parts of mixed-language text.
    
    Args:
        text: Input text
        
    Returns:
        List of dicts with 'text' and 'script' keys
        
    Example:
        >>> segment_by_script("Нужен iPhone 15 Pro")
        [{'text': 'Нужен ', 'script': 'cyrillic'},
         {'text': 'iPhone 15 Pro', 'script': 'latin'}]
    """
    segments = []
    current_segment = ""
    current_script = None
    
    for char in text:
        if char.isspace() or char.isdigit():
            current_segment += char
            continue
        
        # Determine script of this character
        char_script = None
        code_point = ord(char)
        
        for script_name, ranges in SCRIPT_RANGES.items():
            for start, end in ranges:
                if start <= code_point <= end:
                    char_script = script_name
                    break
            if char_script:
                break
        
        if char_script is None:
            char_script = 'other'
        
        # Check if script changed
        if current_script is None:
            current_script = char_script
        elif char_script != current_script and char_script != 'other':
            if current_segment.strip():
                segments.append({
                    'text': current_segment,
                    'script': current_script
                })
            current_segment = char
            current_script = char_script
        else:
            current_segment += char
    
    # Add final segment
    if current_segment.strip():
        segments.append({
            'text': current_segment,
            'script': current_script or 'other'
        })
    
    return segments


def normalize_mixed_language(text: str, primary_language: str = 'en') -> str:
    """
    Normalize mixed-language text with script-aware processing.
    
    Segments text by script and applies appropriate normalization
    to each segment.
    
    Args:
        text: Input text
        primary_language: Primary language for non-Latin scripts
        
    Returns:
        Normalized text
    """
    segments = segment_by_script(text)
    
    # Map scripts to languages for normalization
    script_to_lang = {
        'latin': 'en',
        'cyrillic': primary_language if primary_language in ['ru', 'uk', 'bg'] else 'ru',
        'arabic': 'ar',
        'chinese': 'zh',
        'thai': 'th',
    }
    
    normalized_segments = []
    for segment in segments:
        lang = script_to_lang.get(segment['script'], 'en')
        normalized = normalize_text(segment['text'], lang)
        normalized_segments.append(normalized)
    
    return ' '.join(normalized_segments)
