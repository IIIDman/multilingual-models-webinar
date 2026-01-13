"""
Quantization Workflows
======================
Export and quantize models for production deployment.

Part of Webinar 3: Optimizing Multilingual NLP Models
"""

import os
import time
import numpy as np
from typing import List, Dict


def export_to_onnx(model, tokenizer, output_path: str, max_length: int = 512):
    """
    Export PyTorch model to ONNX format for quantization and optimized inference.
    """
    import torch.onnx
    
    model.eval()
    
    # Create dummy input
    dummy_text = "This is a sample text for export."
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=max_length, truncation=True)
    
    # Export
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        output_path,
        export_params=True,
        opset_version=14,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size'}
        }
    )
    
    return output_path


def quantize_model_onnx(model_path: str, output_path: str, quantization_type: str = "dynamic"):
    """
    Quantize a model using ONNX Runtime for deployment.
    
    Args:
        model_path: Path to ONNX model
        output_path: Path for quantized output
        quantization_type: "dynamic" (PTQ) or "static" (requires calibration data)
    
    Dynamic quantization is simpler and works well for transformer models.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    if quantization_type == "dynamic":
        # Dynamic quantization - no calibration data needed
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
            optimize_model=True
        )
    else:
        # Static quantization requires calibration data
        raise NotImplementedError("Static quantization requires calibration data loader")
    
    return output_path


def benchmark_quantized_model(
    original_path: str, 
    quantized_path: str, 
    test_texts: List[str]
) -> Dict:
    """
    Compare original and quantized model performance.
    """
    import onnxruntime as ort
    
    results = {"original": {}, "quantized": {}}
    
    for name, path in [("original", original_path), ("quantized", quantized_path)]:
        session = ort.InferenceSession(path)
        
        latencies = []
        for _ in test_texts[:100]:
            start = time.perf_counter()
            # Note: actual inference would require tokenization
            latencies.append((time.perf_counter() - start) * 1000)
        
        results[name] = {
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "model_size_mb": os.path.getsize(path) / (1024 * 1024)
        }
    
    return results


def get_size_reduction(original_path: str, quantized_path: str) -> Dict:
    """Calculate size reduction from quantization."""
    original_size = os.path.getsize(original_path) / (1024 * 1024)
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
    
    return {
        "original_mb": original_size,
        "quantized_mb": quantized_size,
        "reduction_percent": (1 - quantized_size / original_size) * 100
    }
