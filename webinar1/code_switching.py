"""
Code-Switching Data Generator
=============================
Generate synthetic code-switched training examples.
"""

import random
from typing import List, Dict, Optional, Tuple
import re


class CodeSwitchingGenerator:
    """
    Generate synthetic code-switched text for training data augmentation.
    
    Code-switching is when speakers alternate between languages within
    a single conversation or sentence. This is common in multilingual
    communities and challenging for NLP models.
    
    Example:
        >>> generator = CodeSwitchingGenerator()
        >>> generator.add_translations('ru', {
        ...     'phone': 'телефон',
        ...     'good': 'хороший',
        ...     'buy': 'купить'
        ... })
        >>> variants = generator.generate("I want to buy a good phone", 'ru')
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.translations: Dict[str, Dict[str, str]] = {}
        if seed is not None:
            random.seed(seed)
    
    def add_translations(self, target_lang: str, word_map: Dict[str, str]) -> None:
        """
        Add word translations for a target language.
        
        Args:
            target_lang: Language code (e.g., 'ru', 'es', 'de')
            word_map: Dictionary mapping source words to translations
        """
        if target_lang not in self.translations:
            self.translations[target_lang] = {}
        
        # Store lowercase for matching
        for source, target in word_map.items():
            self.translations[target_lang][source.lower()] = target
    
    def generate(
        self,
        text: str,
        target_lang: str,
        switch_probability: float = 0.3,
        num_variants: int = 5
    ) -> List[str]:
        """
        Generate code-switched variants of text.
        
        Args:
            text: Original text (typically English)
            target_lang: Language to switch to
            switch_probability: Probability of switching each word
            num_variants: Number of variants to generate
            
        Returns:
            List of code-switched variants
        """
        if target_lang not in self.translations:
            raise ValueError(f"No translations for language: {target_lang}")
        
        word_map = self.translations[target_lang]
        variants = []
        
        # Tokenize preserving punctuation
        tokens = self._tokenize(text)
        
        for _ in range(num_variants):
            new_tokens = []
            for token in tokens:
                word_lower = token.lower().strip('.,!?"\':;')
                
                if word_lower in word_map and random.random() < switch_probability:
                    # Replace with translation, preserve case and punctuation
                    translated = word_map[word_lower]
                    
                    # Preserve trailing punctuation
                    trailing = ''
                    for char in reversed(token):
                        if char in '.,!?"\':;':
                            trailing = char + trailing
                        else:
                            break
                    
                    new_tokens.append(translated + trailing)
                else:
                    new_tokens.append(token)
            
            variant = ' '.join(new_tokens)
            if variant != text and variant not in variants:
                variants.append(variant)
        
        return variants
    
    def generate_intersentential(
        self,
        sentences: List[str],
        target_lang: str,
        translations: Dict[str, str],
        switch_probability: float = 0.3
    ) -> str:
        """
        Generate inter-sentential code-switching (switching between sentences).
        
        Args:
            sentences: List of sentences in source language
            target_lang: Target language code
            translations: Full sentence translations
            switch_probability: Probability of switching each sentence
            
        Returns:
            Text with some sentences switched
        """
        result = []
        for sentence in sentences:
            if sentence in translations and random.random() < switch_probability:
                result.append(translations[sentence])
            else:
                result.append(sentence)
        
        return ' '.join(result)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization preserving punctuation."""
        return text.split()
    
    def get_available_languages(self) -> List[str]:
        """Get list of languages with translations."""
        return list(self.translations.keys())
    
    def get_coverage(self, text: str, target_lang: str) -> Dict[str, any]:
        """
        Check what percentage of words can be code-switched.
        
        Args:
            text: Text to analyze
            target_lang: Target language
            
        Returns:
            Coverage statistics
        """
        if target_lang not in self.translations:
            return {'coverage': 0, 'switchable': 0, 'total': 0}
        
        word_map = self.translations[target_lang]
        tokens = self._tokenize(text)
        
        switchable = sum(
            1 for token in tokens 
            if token.lower().strip('.,!?"\':;') in word_map
        )
        
        return {
            'coverage': switchable / len(tokens) if tokens else 0,
            'switchable': switchable,
            'total': len(tokens)
        }


# Pre-built translation dictionaries for common languages
COMMON_TRANSLATIONS = {
    'ru': {
        # Tech terms
        'phone': 'телефон',
        'smartphone': 'смартфон',
        'laptop': 'ноутбук',
        'computer': 'компьютер',
        'tablet': 'планшет',
        'camera': 'камера',
        'screen': 'экран',
        'battery': 'батарея',
        
        # Common adjectives
        'good': 'хороший',
        'bad': 'плохой',
        'new': 'новый',
        'old': 'старый',
        'cheap': 'дешёвый',
        'expensive': 'дорогой',
        'fast': 'быстрый',
        'slow': 'медленный',
        'best': 'лучший',
        
        # Common verbs
        'buy': 'купить',
        'want': 'хотеть',
        'need': 'нужен',
        'like': 'нравится',
        'love': 'люблю',
        
        # Other
        'price': 'цена',
        'quality': 'качество',
        'delivery': 'доставка',
        'color': 'цвет',
        'size': 'размер',
    },
    'es': {
        'phone': 'teléfono',
        'computer': 'computadora',
        'good': 'bueno',
        'bad': 'malo',
        'new': 'nuevo',
        'cheap': 'barato',
        'expensive': 'caro',
        'buy': 'comprar',
        'want': 'querer',
        'price': 'precio',
        'quality': 'calidad',
    },
    'de': {
        'phone': 'Telefon',
        'computer': 'Computer',
        'good': 'gut',
        'bad': 'schlecht',
        'new': 'neu',
        'cheap': 'billig',
        'expensive': 'teuer',
        'buy': 'kaufen',
        'want': 'wollen',
        'price': 'Preis',
        'quality': 'Qualität',
    }
}


def create_augmented_dataset(
    texts: List[str],
    target_langs: List[str],
    variants_per_text: int = 3,
    switch_probability: float = 0.3
) -> List[Tuple[str, str]]:
    """
    Create augmented dataset with code-switched examples.
    
    Args:
        texts: Original texts
        target_langs: Languages to generate code-switching for
        variants_per_text: Number of variants per original text
        switch_probability: Probability of switching words
        
    Returns:
        List of (text, language_tag) tuples
    """
    generator = CodeSwitchingGenerator()
    
    # Load common translations
    for lang, translations in COMMON_TRANSLATIONS.items():
        if lang in target_langs:
            generator.add_translations(lang, translations)
    
    augmented = []
    
    for text in texts:
        # Keep original
        augmented.append((text, 'original'))
        
        # Generate code-switched variants
        for lang in target_langs:
            if lang in generator.get_available_languages():
                try:
                    variants = generator.generate(
                        text, lang, switch_probability, variants_per_text
                    )
                    for variant in variants:
                        augmented.append((variant, f'code-switched-{lang}'))
                except ValueError:
                    continue
    
    return augmented


# Example usage
if __name__ == "__main__":
    print("Code-Switching Generator Example")
    print("=" * 40)
    
    # Create generator and add translations
    generator = CodeSwitchingGenerator(seed=42)
    generator.add_translations('ru', COMMON_TRANSLATIONS['ru'])
    
    # Original text
    original = "I want to buy a new phone with good camera and fast delivery"
    print(f"\nOriginal: {original}")
    
    # Check coverage
    coverage = generator.get_coverage(original, 'ru')
    print(f"Coverage: {coverage['switchable']}/{coverage['total']} words ({coverage['coverage']:.0%})")
    
    # Generate variants
    print("\nCode-switched variants:")
    variants = generator.generate(original, 'ru', switch_probability=0.4, num_variants=5)
    for i, variant in enumerate(variants, 1):
        print(f"  {i}. {variant}")
    
    # Dataset augmentation example
    print("\n" + "=" * 40)
    print("Dataset Augmentation Example")
    
    sample_texts = [
        "Great phone with excellent camera",
        "Bad quality, very expensive",
        "Fast delivery, good price"
    ]
    
    augmented = create_augmented_dataset(
        sample_texts, 
        target_langs=['ru', 'es'], 
        variants_per_text=2
    )
    
    print(f"\nOriginal texts: {len(sample_texts)}")
    print(f"Augmented dataset: {len(augmented)} examples")
    print("\nSample augmented examples:")
    for text, tag in augmented[:10]:
        print(f"  [{tag}] {text}")
