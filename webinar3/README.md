# Webinar 3: Optimizing Multilingual NLP Models

Code for model optimization techniques: distillation, quantization, and the trade-offs you'll actually face in production.

## Files

- `benchmarking.py` - Benchmark models across languages (latency, accuracy, cold start)
- `distillation.py` - Knowledge distillation with language-balanced batches
- `quantization.py` - ONNX export and INT8 quantization
- `evaluation.py` - Per-language evaluation with threshold checking
- `cost_analysis.py` - Compare scenarios by total cost (infra + error costs)
- `monitoring.py` - Track accuracy drift and latency in production
- `visualizations.py` - Generate the charts from the webinar slides

## Setup

```bash
pip install torch transformers numpy scikit-learn

# For quantization
pip install onnx onnxruntime

# For visualizations
pip install matplotlib
```

## Usage

```python
from webinar3 import (
    MultilingualModelBenchmark,
    DistillationLoss,
    CostBenefitAnalyzer,
    OptimizationScenario,
)

# Benchmark a model
benchmark = MultilingualModelBenchmark("xlm-roberta-base")
latency = benchmark.measure_latency(test_texts)
print(f"p95 latency: {latency['p95']:.1f}ms")

# Compare optimization scenarios
analyzer = CostBenefitAnalyzer(
    traffic_distribution={'en': 0.7, 'es': 0.2, 'ru': 0.1},
    error_cost_per_language={'en': 10, 'es': 15, 'ru': 20},
    monthly_predictions=1_000_000
)

scenarios = [
    OptimizationScenario(
        name="Baseline",
        accuracy_by_language={'en': 0.92, 'es': 0.88, 'ru': 0.84},
        latency_p95_ms=180,
        monthly_infra_cost=5000,
        model_size_mb=2200
    ),
    # ... more scenarios
]

results = analyzer.compare_scenarios(scenarios)
```

Run `python cost_analysis.py` for a full example.

## What This Covers

The webinar focused on optimization decisions that aren't obvious from benchmarks alone:

- Distillation doesn't degrade all languages equally. Low-resource languages lose more.
- Quantization is nearly free for English, but can hurt Thai/Korean significantly.
- Per-language evaluation is mandatory. Aggregate metrics will mislead you.
- Infrastructure cost is often smaller than error cost. Cheaper model != cheaper system.
- The "best" model depends entirely on your traffic distribution and error costs.

## Generate Charts

```bash
python visualizations.py
```

Creates PNG files for the webinar slides.

## Contact

Dmitriy Tsarev  
tsarevdmit@gmail.com  
[LinkedIn](https://www.linkedin.com/in/cxbrv/)
