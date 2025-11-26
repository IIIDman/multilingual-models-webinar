"""
Language Detection Utilities
============================
Functions for detecting language in text using fastText.
"""

from typing import Tuple, Optional

try:
    import fasttext
    # Suppress fastText warning about loading model
    fasttext.FastText.eprint = lambda x: None
except ImportError:
    fasttext = None
    print("Warning: fasttext not installed. Run: pip install fasttext")


class LanguageDetector:
    """
    Language detection using fastText.
    
    Download the model from:
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    
    Example:
        >>> detector = LanguageDetector('lid.176.bin')
        >>> lang, conf = detector.detect("Hello, how are you?")
        >>> print(f"{lang}: {conf:.2f}")  # en: 0.98
    """
    
    def __init__(self, model_path: str = 'lid.176.bin'):
        """
        Initialize the language detector.
        
        Args:
            model_path: Path to fastText language identification model
        """
        if fasttext is None:
            raise ImportError("fasttext is required. Install with: pip install fasttext")
        
        self.model = fasttext.load_model(model_path)
    
    def detect(self, text: str, min_confidence: float = 0.0) -> Tuple[str, float]:
        """
        Detect the language of text.
        
        Args:
            text: Text to analyze
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            
        Returns:
            Tuple of (language_code, confidence)
        """
        # Clean text for detection
        text = text.replace('\n', ' ').strip()
        
        if not text:
            return 'unknown', 0.0
        
        predictions = self.model.predict(text, k=1)
        language = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]
        
        if confidence < min_confidence:
            return 'unknown', confidence
        
        return language, confidence
    
    def detect_with_fallback(
        self, 
        text: str, 
        default: str = 'en',
        min_length: int = 20,
        min_confidence: float = 0.5
    ) -> Tuple[str, float]:
        """
        Detect language with fallback for short/ambiguous text.
        
        Args:
            text: Text to analyze
            default: Default language if detection fails
            min_length: Minimum text length for reliable detection
            min_confidence: Minimum confidence to trust result
            
        Returns:
            Tuple of (language_code, confidence)
        """
        text = text.strip()
        
        # Short text is unreliable
        if len(text) < min_length:
            return default, 0.0
        
        lang, conf = self.detect(text)
        
        if lang == 'unknown' or conf < min_confidence:
            return default, conf
        
        return lang, conf
    
    def detect_multiple(self, text: str, k: int = 3) -> list:
        """
        Get top-k language predictions.
        
        Useful for detecting code-switched or mixed-language text.
        
        Args:
            text: Text to analyze
            k: Number of predictions to return
            
        Returns:
            List of (language_code, confidence) tuples
        """
        text = text.replace('\n', ' ').strip()
        
        if not text:
            return [('unknown', 0.0)]
        
        predictions = self.model.predict(text, k=k)
        
        results = []
        for lang, conf in zip(predictions[0], predictions[1]):
            lang = lang.replace('__label__', '')
            results.append((lang, conf))
        
        return results


def is_mixed_language(text: str, detector: LanguageDetector, threshold: float = 0.3) -> bool:
    """
    Check if text contains mixed languages.
    
    Args:
        text: Text to analyze
        detector: LanguageDetector instance
        threshold: If second language confidence > threshold, consider mixed
        
    Returns:
        True if text appears to be mixed-language
    """
    predictions = detector.detect_multiple(text, k=2)
    
    if len(predictions) < 2:
        return False
    
    # If second language has significant confidence, likely mixed
    return predictions[1][1] > threshold


# Example usage
if __name__ == "__main__":
    print("Language Detection Examples")
    print("=" * 40)
    print("\nNote: Requires lid.176.bin model file")
    print("Download from: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
    
    # Example (uncomment when you have the model):
    # detector = LanguageDetector('lid.176.bin')
    # 
    # test_texts = [
    #     "Hello, how are you today?",
    #     "Привет, как дела?",
    #     "Bonjour, comment allez-vous?",
    #     "Нужен iPhone 15 pro max",  # Mixed Russian + English
    # ]
    # 
    # for text in test_texts:
    #     lang, conf = detector.detect(text)
    #     print(f"  '{text[:30]}...' -> {lang} ({conf:.2f})")
