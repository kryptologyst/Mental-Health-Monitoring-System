# Quick Start Guide

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Run the setup script
python setup.py

# Or manually install dependencies
pip install -r requirements.txt
```

### 2. Train a Model

```bash
# Train with default settings
python train.py

# Train with ClinicalBERT
python train.py --model-type clinical_bert

# Train with custom config
python train.py --config configs/default.yaml --seed 42
```

### 3. Run the Demo

```bash
# Start interactive demo
streamlit run demo/app.py

# Access at http://localhost:8501
```

### 4. Evaluate Model

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

## 📊 Example Usage

### Python API

```python
from src.models import create_model
from src.utils import get_device
import torch

# Load model
config = {"base_model": "distilbert-base-uncased", "num_labels": 6}
model, tokenizer = create_model(config)
device = get_device()

# Make prediction
text = "I'm feeling anxious about my presentation tomorrow"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs["logits"], dim=-1)
    predicted_class = torch.argmax(outputs["logits"], dim=-1).item()

emotions = ["neutral", "happy", "sad", "anxious", "angry", "fearful"]
print(f"Predicted emotion: {emotions[predicted_class]}")
print(f"Confidence: {probabilities[0][predicted_class]:.3f}")
```

### Privacy Protection

```python
from src.utils.privacy import PrivacyProtector

# Initialize privacy protector
protector = PrivacyProtector()

# Check for PII
text = "My name is John Doe and my email is john@example.com"
pii_entities = protector.detect_pii(text)
print(f"PII detected: {len(pii_entities)} entities")

# Anonymize text
anonymized = protector.anonymize_text(text)
print(f"Anonymized: {anonymized}")
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src

# Run specific test file
python -m pytest tests/test_system.py -v
```

## 🔧 Development

```bash
# Format code
black src/ tests/

# Lint code
ruff src/ tests/

# Type checking
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

## 📁 Project Structure

```
mental-health-monitoring/
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Data processing
│   ├── train/             # Training utilities
│   ├── metrics/           # Evaluation metrics
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── demo/                  # Demo applications
├── scripts/               # Utility scripts
├── tests/                 # Test files
├── assets/                # Generated assets
├── checkpoints/           # Model checkpoints
└── data/                  # Data files
```

## ⚠️ Important Notes

- **Research Only**: This tool is for research and educational purposes
- **Not Clinical**: Do not use for clinical diagnosis or medical decisions
- **Privacy**: Always use de-identification for sensitive data
- **Professional Help**: Consult healthcare professionals for mental health concerns

## 🆘 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in config
2. **Model Loading Error**: Check checkpoint file paths
3. **Import Errors**: Ensure all dependencies are installed
4. **Demo Not Loading**: Check if Streamlit is installed

### Getting Help

- Check the README.md for detailed documentation
- Review the DISCLAIMER.md for usage guidelines
- Check the issues section for common problems
- Contact the development team for support

---

**Remember: This is a research tool only. Always consult healthcare professionals for mental health concerns.**
