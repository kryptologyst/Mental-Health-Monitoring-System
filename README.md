# Mental Health Monitoring System

A comprehensive AI-powered system for mental health monitoring through natural language processing, designed for research and educational purposes.

## ⚠️ IMPORTANT DISCLAIMER

**This system is for RESEARCH AND EDUCATIONAL PURPOSES ONLY. It should NOT be used for:**
- Clinical diagnosis or medical decision-making
- Treatment recommendations
- Mental health assessments in clinical settings
- Any form of medical advice

Always consult with qualified healthcare professionals for mental health concerns. This tool is designed to demonstrate AI capabilities in mental health research, not to replace professional medical care.

## Overview

This project implements a modern, research-ready mental health monitoring system that analyzes text input to detect emotional states and mental health indicators. The system uses state-of-the-art NLP models including ClinicalBERT and emotion-aware architectures.

### Key Features

- **Advanced NLP Models**: ClinicalBERT, emotion-aware classifiers, uncertainty quantification
- **Privacy Protection**: Built-in de-identification and PII detection
- **Explainability**: Attention visualization, SHAP analysis, gradient-based explanations
- **Clinical Metrics**: Sensitivity, specificity, calibration, uncertainty quantification
- **Interactive Demo**: Streamlit-based web interface
- **Comprehensive Evaluation**: ROC curves, calibration plots, confusion matrices
- **Research-Ready**: Proper data splits, reproducible results, extensive logging

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA support (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Mental-Health-Monitoring-System.git
   cd Mental-Health-Monitoring-System
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package** (optional):
   ```bash
   pip install -e .
   ```

### Alternative Installation with Conda

```bash
conda create -n mental-health python=3.10
conda activate mental-health
pip install -r requirements.txt
```

## Quick Start

### 1. Training a Model

```bash
# Train with default configuration
python train.py

# Train with specific model type
python train.py --model-type clinical_bert

# Train with custom configuration
python train.py --config configs/custom.yaml --seed 123
```

### 2. Running the Demo

```bash
# Start the Streamlit demo
streamlit run demo/app.py

# Or with custom port
streamlit run demo/app.py --server.port 8502
```

### 3. Using the API (Optional)

```bash
# Start the FastAPI server
python scripts/serve.py --port 8000
```

## Project Structure

```
mental-health-monitoring/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   ├── data/                     # Data processing
│   ├── train/                    # Training utilities
│   ├── metrics/                  # Evaluation metrics
│   ├── utils/                    # Utility functions
│   │   ├── privacy.py           # Privacy protection
│   │   └── explainability.py    # Model explanations
│   └── eval/                     # Evaluation scripts
├── configs/                      # Configuration files
├── demo/                         # Demo applications
├── scripts/                      # Utility scripts
├── tests/                        # Test files
├── assets/                       # Generated assets
├── checkpoints/                  # Model checkpoints
├── data/                         # Data files
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Configuration

The system uses YAML configuration files. Key configuration options:

### Model Configuration
```yaml
model:
  name: "mental_health_classifier"
  base_model: "microsoft/DialoGPT-medium"
  num_labels: 6
  max_length: 512
  dropout: 0.1
```

### Training Configuration
```yaml
training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 10
  warmup_steps: 100
  weight_decay: 0.01
```

### Privacy Configuration
```yaml
privacy:
  enable_deid: true
  deid_entities: ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"]
  redaction_char: "*"
```

## Data

### Synthetic Data Generation

The system includes a comprehensive synthetic data generator that creates realistic mental health text samples:

```python
from src.data import MentalHealthDataProcessor

processor = MentalHealthDataProcessor(config)
synthetic_data = processor.generate_synthetic_data(size=1000)
```

### Data Format

The system expects data in the following format:

```json
{
  "text": "I'm feeling really anxious about my job interview tomorrow.",
  "label": 3,
  "emotion": "anxious",
  "confidence": 0.9
}
```

### Supported Emotions

- **Neutral** (0): Balanced, normal emotional state
- **Happy** (1): Positive emotions, joy, satisfaction
- **Sad** (2): Negative emotions, sadness, depression
- **Anxious** (3): Worry, anxiety, nervousness
- **Angry** (4): Anger, frustration, irritation
- **Fearful** (5): Fear, uncertainty, apprehension

## Models

### Available Model Types

1. **Base Model**: Standard transformer-based classifier
2. **ClinicalBERT**: Specialized for clinical text
3. **Emotion-Aware**: Enhanced with emotion-specific attention
4. **Uncertainty-Aware**: Includes uncertainty quantification

### Model Architecture

```python
from src.models import create_model

model, tokenizer = create_model(config, model_type="clinical_bert")
```

## Evaluation

### Metrics

The system provides comprehensive evaluation metrics:

- **Classification**: Accuracy, F1-score, Precision, Recall
- **Clinical**: Sensitivity, Specificity, PPV, NPV
- **Calibration**: Expected Calibration Error, Brier Score
- **Uncertainty**: Entropy, confidence intervals

### Evaluation Script

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

## Privacy and Security

### De-identification

The system includes built-in privacy protection:

```python
from src.utils.privacy import PrivacyProtector

protector = PrivacyProtector()
anonymized_text = protector.anonymize_text(text)
```

### Supported PII Types

- Person names
- Phone numbers
- Email addresses
- Locations
- Dates and times
- Credit card numbers
- Social Security Numbers

## Explainability

### Available Explanation Methods

1. **Attention Visualization**: Token-level attention weights
2. **Gradient-based**: Integrated Gradients, Saliency
3. **SHAP Analysis**: Feature importance using SHAP values
4. **Uncertainty Quantification**: Model confidence and uncertainty

### Using Explanations

```python
from src.utils.explainability import MentalHealthExplainer

explainer = MentalHealthExplainer(model, tokenizer, device)
explanation = explainer.create_explanation_summary(text)
```

## API Usage

### REST API Endpoints

```bash
# Start the API server
python scripts/serve.py

# Available endpoints:
POST /predict          # Get emotion prediction
POST /explain         # Get model explanation
POST /calibrate       # Calibrate model predictions
GET /health          # Health check
```

### Example API Usage

```python
import requests

# Predict emotion
response = requests.post("http://localhost:8000/predict", 
                        json={"text": "I'm feeling anxious today"})
result = response.json()
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for all classes and functions
- Write tests for new functionality
- Update documentation as needed

## Research Applications

### Academic Use

This system is designed for research in:
- Mental health NLP
- Emotion recognition
- Clinical text analysis
- Model explainability
- Uncertainty quantification

### Citation

If you use this system in your research, please cite:

```bibtex
@software{mental_health_monitoring,
  title={Mental Health Monitoring System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Mental-Health-Monitoring-System}
}
```

## Limitations

### Known Limitations

1. **Not for Clinical Use**: This is a research tool only
2. **Synthetic Data**: Default training uses synthetic data
3. **Limited Languages**: Currently supports English only
4. **Bias Considerations**: Models may reflect training data biases
5. **Privacy**: Always use de-identification for sensitive data

### Ethical Considerations

- Respect user privacy and confidentiality
- Use appropriate de-identification measures
- Consider potential biases in model predictions
- Ensure appropriate disclaimers are displayed
- Follow institutional review board guidelines for research

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Loading Errors**: Check checkpoint file paths
3. **Tokenization Issues**: Verify tokenizer compatibility
4. **Privacy Errors**: Install Presidio dependencies

### Getting Help

- Check the issues section for common problems
- Review the documentation
- Contact the development team for support

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- Presidio for privacy protection
- Streamlit for the demo interface
- The mental health research community

## Changelog

### Version 0.1.0
- Initial release
- Basic emotion classification
- Privacy protection
- Explainability features
- Interactive demo

---

**Remember: This tool is for research and educational purposes only. Always consult healthcare professionals for mental health concerns.**
# Mental-Health-Monitoring-System
