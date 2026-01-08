# Webinar 1: Challenges of Deploying Multilingual Models in Production

Code examples from the first webinar in the series. These are practical utilities I've used or adapted from real production systems.

## Files

- `encoding_utils.py` - UTF-8 validation, mojibake detection, Unicode normalization
- `language_detection.py` - Language detection wrapper using fastText
- `drift_detection.py` - Monitor data drift across languages over time
- `metrics_tracker.py` - Track model performance per language (not just aggregate)
- `ab_testing.py` - Route traffic for A/B tests with language stratification
- `code_switching.py` - Generate synthetic mixed-language data for training
- `model_benchmark.py` - Benchmark inference latency across different model sizes

## Setup

```bash
pip install -r requirements.txt

# For language detection, you'll need the fastText model
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

## Main Points from the Webinar

The webinar covered the gap between multilingual model benchmarks and production reality:

1. Performance varies wildly by language. A model that's 95% accurate on English might be 70% on Thai.
2. Research benchmarks are clean. Production data has encoding issues, mixed languages, transliteration.
3. You can't have it all - accuracy, latency, and cost are always in tension.
4. Aggregate metrics lie. Your model might be failing for 10% of users and you won't see it in overall numbers.
5. Start with zero-shot, see where it breaks, then invest in the languages that matter for your use case.

## Links

- [fastText language identification](https://fasttext.cc/docs/en/language-identification.html)
- [Hugging Face multilingual models](https://huggingface.co/models?language=multilingual)
- [ONNX Runtime](https://onnxruntime.ai/) for production inference

## Contact

Dmitriy Tsarev  
tsarevdmit@gmail.com  
[LinkedIn](https://www.linkedin.com/in/cxbrv/)
