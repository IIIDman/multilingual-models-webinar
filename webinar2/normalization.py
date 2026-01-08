"""
Text Normalization Utilities
============================
Language-aware text normalization.

Part of Webinar 2: Building Robust Data Pipelines for Multilingual Product Classification
"""

import unicodedata
from typing import Dict
from dataclasses import dataclass


@dataclass
class NormalizationConfig:
    """Configuration for text normalization per language."""
    lowercase: bool = True
    turkish_lowercase: bool = False
    unicode_nfc: bool = True
    remove_diacritics: bool = False
    normalize_whitespace: bool = True
    remove_control_chars: bool = True


# Per-language normalization configurations
NORMALIZATION_CONFIGS: Dict[str, NormalizationConfig] = {
    'en': NormalizationConfig(lowercase=True),
    'de': NormalizationConfig(lowercase=True),
    'fr': NormalizationConfig(lowercase=True),
    'es': NormalizationConfig(lowercase=True),
    'ru': NormalizationConfig(lowercase=True),
    'tr': NormalizationConfig(lowercase=False, turkish_lowercase=True),
    'vi': NormalizationConfig(lowercase=True, remove_diacritics=False),  # NEVER remove diacritics
    'zh': NormalizationConfig(lowercase=False),  # No case in Chinese
    'ja': NormalizationConfig(lowercase=False),  # No case in Japanese
    'ar': NormalizationConfig(lowercase=False),  # No case in Arabic
    'th': NormalizationConfig(lowercase=False),  # No case in Thai
}


def turkish_lowercase(text: str) -> str:
    """
    Perform Turkish-aware lowercasing.
    
    In Turkish:
    - 'I' (dotted capital) → 'ı' (dotless lowercase)
    - 'İ' (dotted capital) → 'i' (dotted lowercase)
    
    Standard Python lower() doesn't handle this correctly.
    
    Args:
        text: Input text
        
    Returns:
        Lowercased text with Turkish rules
    """
    # Turkish-specific replacements
    result = text.replace('I', 'ı')  # Dotted capital I → dotless lowercase
    result = result.replace('İ', 'i')  # Dotted capital İ → dotted lowercase
    # Handle remaining characters
    return result.lower()


def remove_diacritics(text: str) -> str:
    """
    Remove diacritical marks from text.
    
    WARNING: This destroys meaning in some languages (e.g., Vietnamese).
    Use only when explicitly required and safe.
    
    Args:
        text: Input text
        
    Returns:
        Text with diacritics removed
    """
    # Decompose characters (é → e + combining accent)
    nfd = unicodedata.normalize('NFD', text)
    # Remove combining diacritical marks
    result = ''.join(char for char in nfd 
                     if unicodedata.category(char) != 'Mn')
    return result


def normalize_text(text: str, language: str = 'en') -> str:
    """
    Normalize text according to language-specific rules.
    
    Args:
        text: Input text
        language: ISO language code
        
    Returns:
        Normalized text
        
    Example:
        >>> normalize_text("  HELLO   World  ", "en")
        'hello world'
        >>> normalize_text("İSTANBUL", "tr")
        'istanbul'
    """
    config = NORMALIZATION_CONFIGS.get(language, NormalizationConfig())
    
    # Unicode NFC normalization (should almost always be applied)
    if config.unicode_nfc:
        text = unicodedata.normalize('NFC', text)
    
    # Remove control characters
    if config.remove_control_chars:
        text = ''.join(char for char in text if unicodedata.category(char) != 'Cc' or char in '\n\t')
    
    # Normalize whitespace
    if config.normalize_whitespace:
        text = ' '.join(text.split())
    
    # Lowercasing (with Turkish special case)
    if config.turkish_lowercase:
        text = turkish_lowercase(text)
    elif config.lowercase:
        text = text.lower()
    
    # Diacritic removal (use cautiously!)
    if config.remove_diacritics:
        text = remove_diacritics(text)
    
    return text
