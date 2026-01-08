"""
Encoding Validation Utilities
=============================
Detect and fix encoding issues in multilingual text.

Part of Webinar 2: Building Robust Data Pipelines for Multilingual Product Classification
"""

from typing import Dict, Tuple, Optional, Any
from enum import Enum


class QualityStatus(Enum):
    """Quality check outcomes"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


def check_encoding(text: str) -> Tuple[QualityStatus, Optional[str]]:
    """
    Validate text encoding and detect common mojibake patterns.
    
    Mojibake occurs when text encoded in one format is decoded using another,
    resulting in garbled characters. Common patterns include:
    - UTF-8 interpreted as Latin-1: "Привет" → "ÐŸÑ€Ð¸Ð²ÐµÑ‚"
    - Windows-1251 misread as UTF-8
    
    Args:
        text: Input text string
        
    Returns:
        Tuple of (status, reason) where status is QualityStatus enum
        
    Example:
        >>> check_encoding("Hello, world!")
        (QualityStatus.PASS, None)
        >>> check_encoding("ÐŸÑ€Ð¸Ð²ÐµÑ‚")
        (QualityStatus.WARNING, 'possible_encoding_issue')
    """
    try:
        # Basic UTF-8 roundtrip check
        text.encode('utf-8').decode('utf-8')
        
        # Check for common mojibake patterns
        # These patterns indicate UTF-8 bytes interpreted as Latin-1
        mojibake_patterns = [
            'Ð',      # Common in Russian mojibake
            'Ã',      # Common in Western European mojibake
            'â€',     # Common UTF-8 -> Latin-1 pattern
            '\ufffd', # Unicode replacement character
        ]
        
        for pattern in mojibake_patterns:
            if pattern in text:
                return QualityStatus.WARNING, "possible_encoding_issue"
        
        # Check for suspicious lone surrogates
        for char in text:
            if 0xD800 <= ord(char) <= 0xDFFF:
                return QualityStatus.FAIL, "invalid_surrogate_character"
        
        return QualityStatus.PASS, None
        
    except UnicodeError as e:
        return QualityStatus.FAIL, f"invalid_utf8: {str(e)}"


def detect_encoding(raw_bytes: bytes) -> Dict[str, Any]:
    """
    Detect the encoding of raw bytes using heuristics.
    
    This is a simplified version. For production use, consider
    the `chardet` or `charset_normalizer` libraries.
    
    Args:
        raw_bytes: Raw bytes to analyze
        
    Returns:
        Dict with 'encoding' and 'confidence' keys
    """
    # Try common encodings in order of preference
    encodings_to_try = [
        ('utf-8', 'strict'),
        ('utf-8-sig', 'strict'),  # UTF-8 with BOM
        ('utf-16', 'strict'),
        ('cp1251', 'strict'),     # Windows Cyrillic
        ('cp1252', 'strict'),     # Windows Western European
        ('iso-8859-1', 'strict'), # Latin-1
        ('gb2312', 'strict'),     # Simplified Chinese
        ('big5', 'strict'),       # Traditional Chinese
    ]
    
    for encoding, errors in encodings_to_try:
        try:
            decoded = raw_bytes.decode(encoding, errors=errors)
            # Basic validation - check for common issues
            if '\x00' not in decoded:  # No null bytes in middle
                return {
                    'encoding': encoding,
                    'confidence': 0.8 if encoding == 'utf-8' else 0.6,
                    'decoded': decoded
                }
        except (UnicodeDecodeError, LookupError):
            continue
    
    # Fallback to latin-1 which never fails
    return {
        'encoding': 'iso-8859-1',
        'confidence': 0.3,
        'decoded': raw_bytes.decode('iso-8859-1', errors='replace')
    }


def fix_common_encoding_issues(text: str) -> str:
    """
    Attempt to fix common encoding corruption issues.
    
    This handles cases where UTF-8 text was incorrectly decoded as Latin-1
    and then re-encoded as UTF-8.
    
    Args:
        text: Potentially corrupted text
        
    Returns:
        Fixed text if corruption was detected, original otherwise
    """
    # Try to detect and fix UTF-8 interpreted as Latin-1
    try:
        # Encode as Latin-1 (which preserves byte values)
        # then decode as UTF-8
        fixed = text.encode('latin-1').decode('utf-8')
        # If this produces readable text with fewer replacement chars,
        # it's likely the correct interpretation
        if fixed.count('\ufffd') < text.count('\ufffd'):
            return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    
    return text
