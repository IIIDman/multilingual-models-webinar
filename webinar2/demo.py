#!/usr/bin/env python3
"""
Demo: Multilingual Pipeline Utilities
=====================================
Run this script to see the utilities in action.

Usage:
    python demo.py
"""

from webinar2_code import (
    check_encoding,
    detect_mixed_language_content,
    normalize_text,
    MultilingualQualityChecker,
    extract_language_agnostic_features,
    is_likely_transliteration,
)


def main():
    print("=" * 60)
    print("MULTILINGUAL PIPELINE UTILITIES - DEMO")
    print("=" * 60)
    
    # Test encoding detection
    print("\n1. ENCODING CHECKS:")
    test_texts = [
        ("Hello, world!", "Clean English"),
        ("Привет, мир!", "Clean Russian"),
        ("ÐŸÑ€Ð¸Ð²ÐµÑ‚", "Corrupted Russian (mojibake)"),
    ]
    
    for text, description in test_texts:
        status, reason = check_encoding(text)
        print(f"   {description}: {status.value} ({reason or 'OK'})")
    
    # Test script detection
    print("\n2. SCRIPT DETECTION:")
    mixed_text = "Нужен iPhone 15 Pro Max в blue color"
    result = detect_mixed_language_content(mixed_text)
    print(f"   Text: '{mixed_text}'")
    print(f"   Is mixed: {result['is_mixed']}")
    print(f"   Scripts: {result['scripts']}")
    
    # Test normalization
    print("\n3. NORMALIZATION:")
    test_norm = [
        ("  HELLO   World  ", "en"),
        ("İSTANBUL", "tr"),
        ("МОСКВА", "ru"),
    ]
    
    for text, lang in test_norm:
        normalized = normalize_text(text, lang)
        print(f"   '{text}' ({lang}) → '{normalized}'")
    
    # Test quality checker
    print("\n4. QUALITY CHECKS:")
    checker = MultilingualQualityChecker()
    test_quality = "Нужен новый телефон iPhone для работы"
    results = checker.run_all_checks(test_quality, "ru")
    print(f"   Text: '{test_quality}'")
    for r in results:
        print(f"   - {r.check_name}: {r.status.value}")
    overall = checker.get_overall_status(results)
    print(f"   Overall: {overall.value}")
    
    # Test feature extraction
    print("\n5. FEATURE EXTRACTION:")
    features = extract_language_agnostic_features(test_quality)
    print(f"   char_count: {features['char_count']}")
    print(f"   word_count: {features['word_count']}")
    print(f"   has_number: {features['has_number']}")
    
    # Test transliteration detection
    print("\n6. TRANSLITERATION DETECTION:")
    translit_tests = [
        "privet kak dela",
        "iPhone Pro Max",
        "khorosho spasibo",
    ]
    
    for text in translit_tests:
        result = is_likely_transliteration(text, "ru")
        print(f"   '{text}': translit={result['is_transliteration']} (conf={result['confidence']:.2f})")
    
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
