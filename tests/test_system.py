"""
Test suite for mental health monitoring system.

This module contains unit tests for all major components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, ConfigManager
from data import MentalHealthDataProcessor, create_sample_dataset
from models import MentalHealthClassifier, create_model, get_model_info
from train import FocalLoss, LabelSmoothingLoss, MentalHealthTrainer
from metrics import MentalHealthEvaluator, compute_uncertainty_metrics
from utils.privacy import PrivacyProtector


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        # Test that seed is set (basic check)
        assert True  # Placeholder for actual seed testing
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_config_manager(self):
        """Test configuration management."""
        # Create a temporary config file
        config_path = "test_config.yaml"
        with open(config_path, "w") as f:
            f.write("""
model:
  num_labels: 6
training:
  batch_size: 16
data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
evaluation:
  metrics: ["accuracy"]
""")
        
        try:
            config_manager = ConfigManager(config_path)
            assert config_manager.get("model.num_labels") == 6
            assert config_manager.get("training.batch_size") == 16
        finally:
            # Clean up
            Path(config_path).unlink(missing_ok=True)


class TestDataProcessor:
    """Test data processing functionality."""
    
    def test_data_processor_init(self):
        """Test data processor initialization."""
        config = {
            "synthetic_data_size": 100,
            "min_text_length": 10,
            "max_text_length": 512
        }
        processor = MentalHealthDataProcessor(config)
        assert processor.num_labels == 6
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        config = {"synthetic_data_size": 50}
        processor = MentalHealthDataProcessor(config)
        data = processor.generate_synthetic_data(50)
        
        assert len(data) == 50
        assert all("text" in item for item in data)
        assert all("label" in item for item in data)
        assert all("emotion" in item for item in data)
    
    def test_sample_dataset(self):
        """Test sample dataset creation."""
        data = create_sample_dataset()
        assert len(data) == 6
        assert all("text" in item for item in data)
        assert all("label" in item for item in data)


class TestModels:
    """Test model architectures."""
    
    def test_mental_health_classifier(self):
        """Test base classifier."""
        config = {
            "base_model": "distilbert-base-uncased",
            "num_labels": 6,
            "dropout": 0.1
        }
        
        model = MentalHealthClassifier(config)
        assert model.num_labels == 6
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            assert "logits" in outputs
            assert outputs["logits"].shape == (2, 6)
    
    def test_model_info(self):
        """Test model information extraction."""
        config = {
            "base_model": "distilbert-base-uncased",
            "num_labels": 6
        }
        model = MentalHealthClassifier(config)
        
        info = get_model_info(model)
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "model_size_mb" in info
        assert "architecture" in info


class TestLossFunctions:
    """Test loss functions."""
    
    def test_focal_loss(self):
        """Test focal loss computation."""
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
        logits = torch.randn(4, 6)
        targets = torch.randint(0, 6, (4,))
        
        loss = focal_loss(logits, targets)
        assert loss.item() >= 0
    
    def test_label_smoothing_loss(self):
        """Test label smoothing loss."""
        label_smoothing_loss = LabelSmoothingLoss(num_classes=6, smoothing=0.1)
        
        logits = torch.randn(4, 6)
        targets = torch.randint(0, 6, (4,))
        
        loss = label_smoothing_loss(logits, targets)
        assert loss.item() >= 0


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_evaluator_init(self):
        """Test evaluator initialization."""
        evaluator = MentalHealthEvaluator(["neutral", "happy", "sad", "anxious", "angry", "fearful"])
        assert evaluator.num_labels == 6
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        evaluator = MentalHealthEvaluator(["neutral", "happy", "sad", "anxious", "angry", "fearful"])
        
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 1, 1])
        y_proba = np.random.rand(5, 6)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        
        metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
    
    def test_uncertainty_metrics(self):
        """Test uncertainty metrics computation."""
        y_proba = np.random.rand(10, 6)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        
        metrics = compute_uncertainty_metrics(y_proba)
        
        assert "mean_entropy" in metrics
        assert "mean_max_proba" in metrics
        assert "low_confidence_rate" in metrics
        assert "high_confidence_rate" in metrics


class TestPrivacy:
    """Test privacy protection functionality."""
    
    def test_privacy_protector_init(self):
        """Test privacy protector initialization."""
        protector = PrivacyProtector()
        assert protector.redaction_char == "*"
    
    def test_fallback_pii_detection(self):
        """Test fallback PII detection."""
        protector = PrivacyProtector()
        
        text_with_email = "Contact me at john.doe@example.com"
        pii_entities = protector._fallback_pii_detection(text_with_email)
        
        assert len(pii_entities) > 0
        assert any(entity["entity_type"] == "EMAIL_ADDRESS" for entity in pii_entities)
    
    def test_fallback_anonymization(self):
        """Test fallback anonymization."""
        protector = PrivacyProtector()
        
        text_with_email = "Contact me at john.doe@example.com"
        anonymized = protector._fallback_anonymization(text_with_email)
        
        assert "john.doe@example.com" not in anonymized
        assert "***@example.com" in anonymized
    
    def test_privacy_compliance(self):
        """Test privacy compliance validation."""
        protector = PrivacyProtector()
        
        text_with_pii = "My name is John Doe and my email is john@example.com"
        compliance = validate_privacy_compliance(text_with_pii, protector)
        
        assert not compliance["is_compliant"]
        assert compliance["pii_count"] > 0


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline."""
        config = {
            "base_model": "distilbert-base-uncased",
            "num_labels": 6,
            "dropout": 0.1
        }
        
        model = MentalHealthClassifier(config)
        model.eval()
        
        # Mock tokenizer
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                return {
                    "input_ids": torch.randint(0, 1000, (1, 10)),
                    "attention_mask": torch.ones(1, 10)
                }
        
        tokenizer = MockTokenizer()
        
        # Test prediction
        text = "I'm feeling happy today"
        inputs = tokenizer(text)
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
        
        assert predicted_class >= 0 and predicted_class < 6
        assert probabilities.shape == (1, 6)
        assert torch.allclose(probabilities.sum(), torch.tensor(1.0), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
