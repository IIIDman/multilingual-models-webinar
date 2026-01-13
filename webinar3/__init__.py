"""
Webinar 3: Optimizing Multilingual NLP Models
==============================================
Performance vs. Accuracy Trade-offs

Modules:
    benchmarking - Model benchmarking across languages
    distillation - Knowledge distillation implementation
    quantization - ONNX export and quantization
    evaluation - Per-language evaluation framework
    cost_analysis - Cost-benefit analysis tools
    monitoring - Production monitoring setup
    visualizations - Generate charts
"""

from .benchmarking import MultilingualModelBenchmark

from .distillation import (
    DistillationLoss,
    MultilingualDistillationTrainer,
    create_student_model,
)

from .quantization import (
    export_to_onnx,
    quantize_model_onnx,
    benchmark_quantized_model,
    get_size_reduction,
)

from .evaluation import (
    LanguageMetrics,
    PerLanguageEvaluator,
)

from .cost_analysis import (
    OptimizationScenario,
    CostBenefitAnalyzer,
    example_cost_benefit_analysis,
)

from .monitoring import MultilingualMonitor

__version__ = "1.0.0"
__author__ = "Dmitriy Tsarev"

__all__ = [
    # Benchmarking
    "MultilingualModelBenchmark",
    
    # Distillation
    "DistillationLoss",
    "MultilingualDistillationTrainer",
    "create_student_model",
    
    # Quantization
    "export_to_onnx",
    "quantize_model_onnx",
    "benchmark_quantized_model",
    "get_size_reduction",
    
    # Evaluation
    "LanguageMetrics",
    "PerLanguageEvaluator",
    
    # Cost analysis
    "OptimizationScenario",
    "CostBenefitAnalyzer",
    "example_cost_benefit_analysis",
    
    # Monitoring
    "MultilingualMonitor",
]
