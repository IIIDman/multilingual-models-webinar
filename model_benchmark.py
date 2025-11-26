"""
Model Benchmarking Utilities
============================
Benchmark inference speed and compare multilingual models.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import statistics

try:
    import numpy as np
except ImportError:
    np = None
    print("Warning: numpy not installed. Run: pip install numpy")


@dataclass
class BenchmarkResult:
    """Results from a model benchmark."""
    model_name: str
    num_samples: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_per_sec: float


class ModelBenchmarker:
    """
    Benchmark model inference speed.
    
    Example:
        >>> benchmarker = ModelBenchmarker()
        >>> # Add your model's predict function
        >>> result = benchmarker.benchmark(
        ...     model_name="xlm-roberta-base",
        ...     predict_fn=model.predict,
        ...     test_inputs=test_texts
        ... )
        >>> print(f"p95 latency: {result.p95_ms:.1f}ms")
    """
    
    def __init__(self, warmup_runs: int = 10):
        """
        Initialize benchmarker.
        
        Args:
            warmup_runs: Number of warmup runs before measurement
        """
        self.warmup_runs = warmup_runs
        self.results: List[BenchmarkResult] = []
    
    def benchmark(
        self,
        model_name: str,
        predict_fn: callable,
        test_inputs: List[Any],
        num_runs: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Benchmark a model's inference speed.
        
        Args:
            model_name: Name for this model
            predict_fn: Function that takes input and returns prediction
            test_inputs: List of test inputs
            num_runs: Number of runs (defaults to len(test_inputs))
            
        Returns:
            BenchmarkResult with timing statistics
        """
        if num_runs is None:
            num_runs = len(test_inputs)
        
        # Warmup
        print(f"Warming up {model_name}...")
        for i in range(min(self.warmup_runs, len(test_inputs))):
            _ = predict_fn(test_inputs[i])
        
        # Benchmark
        print(f"Benchmarking {model_name} ({num_runs} runs)...")
        latencies = []
        
        for i in range(num_runs):
            input_data = test_inputs[i % len(test_inputs)]
            
            start = time.perf_counter()
            _ = predict_fn(input_data)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # Convert to ms
        
        result = self._compute_stats(model_name, latencies)
        self.results.append(result)
        
        return result
    
    def _compute_stats(self, model_name: str, latencies: List[float]) -> BenchmarkResult:
        """Compute statistics from latency measurements."""
        if np is not None:
            return BenchmarkResult(
                model_name=model_name,
                num_samples=len(latencies),
                mean_ms=float(np.mean(latencies)),
                std_ms=float(np.std(latencies)),
                p50_ms=float(np.percentile(latencies, 50)),
                p95_ms=float(np.percentile(latencies, 95)),
                p99_ms=float(np.percentile(latencies, 99)),
                min_ms=float(np.min(latencies)),
                max_ms=float(np.max(latencies)),
                throughput_per_sec=1000 / float(np.mean(latencies))
            )
        else:
            # Fallback without numpy
            sorted_lat = sorted(latencies)
            mean = statistics.mean(latencies)
            
            def percentile(data, p):
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = f + 1 if f + 1 < len(data) else f
                return data[f] + (k - f) * (data[c] - data[f])
            
            return BenchmarkResult(
                model_name=model_name,
                num_samples=len(latencies),
                mean_ms=mean,
                std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
                p50_ms=percentile(sorted_lat, 50),
                p95_ms=percentile(sorted_lat, 95),
                p99_ms=percentile(sorted_lat, 99),
                min_ms=min(latencies),
                max_ms=max(latencies),
                throughput_per_sec=1000 / mean
            )
    
    def compare(self) -> str:
        """Generate comparison table of all benchmarked models."""
        if not self.results:
            return "No benchmark results available."
        
        # Header
        lines = [
            "Model Comparison",
            "=" * 80,
            f"{'Model':<30} {'Mean (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Throughput':<12}",
            "-" * 80
        ]
        
        # Sort by mean latency
        sorted_results = sorted(self.results, key=lambda r: r.mean_ms)
        
        for r in sorted_results:
            lines.append(
                f"{r.model_name:<30} {r.mean_ms:<12.2f} {r.p95_ms:<12.2f} "
                f"{r.p99_ms:<12.2f} {r.throughput_per_sec:<12.1f}/s"
            )
        
        return "\n".join(lines)


def benchmark_transformers_model(
    model_name: str,
    test_texts: List[str],
    num_runs: int = 100,
    device: str = "cpu"
) -> BenchmarkResult:
    """
    Benchmark a Hugging Face transformers model.
    
    Args:
        model_name: Hugging Face model name (e.g., 'bert-base-multilingual-cased')
        test_texts: List of test texts
        num_runs: Number of inference runs
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        BenchmarkResult with timing statistics
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError("transformers and torch required. Install with: pip install transformers torch")
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    def predict(text):
        inputs = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        return outputs.last_hidden_state
    
    benchmarker = ModelBenchmarker(warmup_runs=10)
    return benchmarker.benchmark(model_name, predict, test_texts, num_runs)


def estimate_cost(
    result: BenchmarkResult,
    requests_per_day: int,
    instance_cost_per_hour: float,
    requests_per_instance_per_second: Optional[float] = None
) -> Dict[str, float]:
    """
    Estimate infrastructure cost based on benchmark results.
    
    Args:
        result: Benchmark result
        requests_per_day: Expected daily request volume
        instance_cost_per_hour: Cost per instance per hour
        requests_per_instance_per_second: Override throughput (uses benchmark if None)
        
    Returns:
        Cost estimates
    """
    if requests_per_instance_per_second is None:
        requests_per_instance_per_second = result.throughput_per_sec * 0.7  # 70% efficiency
    
    requests_per_second = requests_per_day / 86400  # Seconds per day
    instances_needed = max(1, requests_per_second / requests_per_instance_per_second)
    instances_needed = int(instances_needed) + 1  # Round up + buffer
    
    hours_per_month = 730
    monthly_cost = instances_needed * instance_cost_per_hour * hours_per_month
    
    return {
        'requests_per_day': requests_per_day,
        'requests_per_second': requests_per_second,
        'throughput_per_instance': requests_per_instance_per_second,
        'instances_needed': instances_needed,
        'cost_per_hour': instances_needed * instance_cost_per_hour,
        'cost_per_month': monthly_cost,
        'cost_per_request': monthly_cost / (requests_per_day * 30),
    }


# Example usage
if __name__ == "__main__":
    print("Model Benchmarking Example")
    print("=" * 40)
    
    # Simulate model benchmarks (without actual model loading)
    benchmarker = ModelBenchmarker()
    
    # Simulate different models with mock predict functions
    import random
    
    def mock_mbert(text):
        time.sleep(random.gauss(0.035, 0.005))  # ~35ms
        return "prediction"
    
    def mock_xlmr_base(text):
        time.sleep(random.gauss(0.080, 0.010))  # ~80ms
        return "prediction"
    
    def mock_xlmr_large(text):
        time.sleep(random.gauss(0.150, 0.020))  # ~150ms
        return "prediction"
    
    test_texts = ["Sample text for testing " * 10] * 50
    
    print("\nRunning benchmarks (simulated)...\n")
    
    benchmarker.benchmark("mBERT-base", mock_mbert, test_texts, num_runs=50)
    benchmarker.benchmark("XLM-R-base", mock_xlmr_base, test_texts, num_runs=50)
    benchmarker.benchmark("XLM-R-large", mock_xlmr_large, test_texts, num_runs=50)
    
    print("\n" + benchmarker.compare())
    
    # Cost estimation
    print("\n" + "=" * 40)
    print("Cost Estimation (1M requests/day)")
    print("=" * 40)
    
    for result in benchmarker.results:
        cost = estimate_cost(
            result,
            requests_per_day=1_000_000,
            instance_cost_per_hour=0.10  # $0.10/hour (e.g., t3.medium)
        )
        print(f"\n{result.model_name}:")
        print(f"  Instances needed: {cost['instances_needed']}")
        print(f"  Monthly cost: ${cost['cost_per_month']:.2f}")
        print(f"  Cost per 1K requests: ${cost['cost_per_request'] * 1000:.4f}")
