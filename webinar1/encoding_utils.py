"""
Encoding Utilities for Multilingual NLP
========================================
Functions for validating and normalizing text encoding.
"""

import unicodedata
from typing import Union

try:
    import chardet
except ImportError:
    chardet = None
    print("Warning: chardet not installed. Run: pip install chardet")


def validate_and_fix_encoding(byte_string: bytes) -> str:
    """
    Detect encoding and convert text to UTF-8.
    
    Args:
        byte_string: Raw bytes to decode
        
    Returns:
        UTF-8 decoded string
        
    Example:
        >>> raw_bytes = "Привет".encode('windows-1251')
        >>> text = validate_and_fix_encoding(raw_bytes)
        >>> print(text)  # Привет
    """
    if chardet is None:
        raise ImportError("chardet is required. Install with: pip install chardet")
    
    detected = chardet.detect(byte_string)
    original_encoding = detected['encoding']
    confidence = detected['confidence']
    
    if original_encoding and original_encoding.lower() != 'utf-8':
        print(f"Warning: Detected {original_encoding} (confidence: {confidence:.2f})")
        text = byte_string.decode(original_encoding, errors='replace')
        return text
    
    return byte_string.decode('utf-8', errors='replace')


def normalize_text(text: str) -> str:
    """
    Normalize Unicode characters and remove invisible characters.
    
    Applies NFKC normalization and removes:
    - Zero-width spaces
    - Format characters
    - Extra whitespace
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
        
    Example:
        >>> messy = "Hello\u200b World"  # Contains zero-width space
        >>> clean = normalize_text(messy)
        >>> print(repr(clean))  # 'Hello World'
    """
    # NFKC normalization (compatibility decomposition + canonical composition)
    text = unicodedata.normalize('NFKC', text)
    
    # Remove zero-width and formatting characters (category 'Cf')
    text = ''.join(
        char for char in text 
        if unicodedata.category(char) != 'Cf'
    )
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def is_valid_utf8(text: Union[str, bytes]) -> bool:
    """
    Check if text is valid UTF-8.
    
    Args:
        text: String or bytes to validate
        
    Returns:
        True if valid UTF-8, False otherwise
    """
    try:
        if isinstance(text, bytes):
            text.decode('utf-8')
        else:
            text.encode('utf-8')
        return True
    except (UnicodeDecodeError, UnicodeEncodeError):
        return False


def detect_mojibake(text: str) -> bool:
    """
    Detect potential mojibake (character corruption).
    
    Common patterns that indicate encoding issues:
    - "Ã©" instead of "é"
    - "Ð" sequences in non-Cyrillic context
    
    Args:
        text: Text to check
        
    Returns:
        True if mojibake detected, False otherwise
    """
    # Common mojibake patterns
    mojibake_patterns = [
        'Ã©', 'Ã¨', 'Ã ', 'Ã¼', 'Ã¶', 'Ã¤',  # UTF-8 as Latin-1
        'â€™', 'â€"', 'â€œ', 'â€�',  # Smart quotes corrupted
        'Ð', 'Ñ'  # Cyrillic as Latin-1 (check context)
    ]
    
    for pattern in mojibake_patterns:
        if pattern in text:
            return True
    
    return False


# Example usage
if __name__ == "__main__":
    # Test normalization
    test_cases = [
        "Hello\u200bWorld",  # Zero-width space
        "café",  # Normal text
        "  Multiple   spaces  ",  # Extra whitespace
    ]
    
    print("Normalization examples:")
    for text in test_cases:
        normalized = normalize_text(text)
        print(f"  '{text}' -> '{normalized}'")
    
    # Test mojibake detection
    print("\nMojibake detection:")
    print(f"  'café' -> {detect_mojibake('café')}")  # False
    print(f"  'cafÃ©' -> {detect_mojibake('cafÃ©')}")  # True
