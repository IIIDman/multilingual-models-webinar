# Webinar 2: Building Robust Data Pipelines for Multilingual Product Classification

Utilities for preprocessing multilingual text data. These handle the annoying edge cases that break production systems: encoding corruption, mixed-language content, script detection, and per-language normalization.

## Files

- `encoding.py` - Encoding validation, mojibake detection, corruption repair
- `script_detection.py` - Detect writing scripts (Cyrillic, Latin, Arabic, Chinese, etc.)
- `normalization.py` - Language-aware normalization, including Turkish I/ı handling
- `mixed_language.py` - Segment and process text with multiple scripts
- `transliteration.py` - Detect transliterated text (e.g., "privet" instead of "привет")
- `quality_checker.py` - Quality validation framework with configurable checks
- `features.py` - Extract language-agnostic features for classification
- `monitoring.py` - Per-language metrics and anomaly detection

## Setup

```bash
# No external dependencies - uses only Python standard library
# For better encoding detection in production, add:
pip install chardet
```

## Usage

```python
from webinar2 import (
    check_encoding,
    detect_mixed_language_content,
    normalize_text,
    MultilingualQualityChecker,
)

# Check for encoding issues
status, reason = check_encoding("Привет мир!")

# Detect mixed-language content (very common in e-commerce)
result = detect_mixed_language_content("Нужен iPhone 15 Pro")
# {'is_mixed': True, 'scripts': ['cyrillic', 'latin'], ...}

# Normalize with language-specific rules
text = normalize_text("İSTANBUL", "tr")  # -> "istanbul" (not "ıstanbul")

# Run all quality checks
checker = MultilingualQualityChecker()
results = checker.run_all_checks("Sample text", "en")
```

Run `python demo.py` to see everything in action.

## What This Covers

The webinar focused on data pipeline issues that cause most production failures:

- Encoding problems silently corrupt data and tank accuracy (35 point drops are real)
- Mixed-language content is the norm in e-commerce (45% of "Russian" product data has English)
- Normalization that works for English breaks other languages (Turkish I, Vietnamese diacritics)
- Quality thresholds need to be per-language, not global
- Per-language monitoring catches issues that aggregate metrics hide

## Contact

Dmitriy Tsarev  
tsarevdmit@gmail.com  
[LinkedIn](https://www.linkedin.com/in/cxbrv/)
