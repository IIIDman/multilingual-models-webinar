# Multilingual AI in Production - Webinar Series

Code and utilities from my webinar series on deploying multilingual NLP systems. Three parts covering the full pipeline from model selection to production monitoring.

## Webinars

### [Webinar 1: Challenges of Deploying Multilingual Models](./webinar1/)
The reality of multilingual models in production vs. research benchmarks. Covers language detection, encoding issues, performance monitoring, and A/B testing strategies.

### [Webinar 2: Building Robust Data Pipelines](./webinar2/)
Data pipeline utilities for multilingual text. Encoding validation, script detection, normalization (including Turkish edge cases), mixed-language handling, and quality monitoring.

### [Webinar 3: Optimizing Multilingual NLP Models](./webinar3/)
Performance vs. accuracy trade-offs. Knowledge distillation, quantization, per-language evaluation, cost-benefit analysis, and production monitoring.

## Structure

```
├── webinar1/          # Deployment challenges, monitoring, A/B testing
├── webinar2/          # Data pipelines, preprocessing, quality checks
└── webinar3/          # Model optimization, distillation, quantization
```

## Getting Started

Each webinar folder has its own README with setup instructions. The code is meant to be adapted to your use case, not used as-is.

```bash
git clone https://github.com/IIIDman/multilingual-models-webinar.git
cd multilingual-models-webinar/webinar1
pip install -r requirements.txt
```

## Why This Exists

Most multilingual NLP content focuses on model architectures and benchmark scores. This series covers the unglamorous parts that actually matter in production: encoding bugs, per-language monitoring, data quality, and the trade-offs you'll inevitably make.

## Author

Dmitriy Tsarev  
AI/ML Engineer  
[LinkedIn](https://www.linkedin.com/in/cxbrv/) | tsarevdmit@gmail.com

## License

MIT - use however you want.
