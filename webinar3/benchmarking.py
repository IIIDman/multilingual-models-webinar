"""
Model Benchmarking Utilities
============================
Benchmark multilingual models across languages.

Part of Webinar 3: Optimizing Multilingual NLP Models
"""

import time
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List


class MultilingualModelBenchmark:
    """
    Benchmark multilingual models across languages with comprehensive metrics.
    
    Usage:
        benchmark = MultilingualModelBenchmark(model_name="xlm-roberta-base")
        results = benchmark.run_benchmark(test_data, languages=['en', 'es', 'ru', 'th'])
    """
    
    def __init__(self, model_name: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
    
    def measure_latency(self, texts: List[str], num_runs: int = 100) -> Dict[str, float]:
        """Measure inference latency with percentiles."""
        latencies = []
        
        for text in texts[:num_runs]:
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   truncation=True, max_length=512).to(self.device)
            
            # Warmup
            with torch.no_grad():
                _ = self.model(**inputs)
            
            # Measure
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(**inputs)
            latencies.append((time.perf_counter() - start) * 1000)  # ms
        
        return {
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "mean": np.mean(latencies)
        }
    
    def measure_accuracy_per_language(
        self, 
        texts: List[str], 
        labels: List[int],
        languages: List[str]
    ) -> Dict[str, float]:
        """Calculate accuracy broken down by language."""
        results = {}
        
        for lang in set(languages):
            lang_mask = [l == lang for l in languages]
            lang_texts = [t for t, m in zip(texts, lang_mask) if m]
            lang_labels = [l for l, m in zip(labels, lang_mask) if m]
            
            correct = 0
            for text, label in zip(lang_texts, lang_labels):
                inputs = self.tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                pred = outputs.logits.argmax(dim=-1).item()
                correct += int(pred == label)
            
            results[lang] = correct / len(lang_labels) if lang_labels else 0.0
        
        return results
    
    def get_model_size_mb(self) -> float:
        """Calculate model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def measure_cold_start(self) -> float:
        """Measure model loading time (cold start)."""
        start = time.perf_counter()
        _ = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        return time.perf_counter() - start
