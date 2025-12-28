"""
Model architectures for mental health monitoring.

This module provides various neural network architectures optimized
for mental health text classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertForSequenceClassification, BertTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer
)
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MentalHealthClassifier(nn.Module):
    """Base mental health text classifier."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the classifier.
        
        Args:
            config: Model configuration dictionary.
        """
        super().__init__()
        self.config = config
        self.num_labels = config["num_labels"]
        self.dropout_rate = config.get("dropout", 0.1)
        
        # Load pre-trained model
        self.backbone = AutoModel.from_pretrained(config["base_model"])
        self.hidden_size = self.backbone.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            
        Returns:
            Dictionary containing logits and hidden states.
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        dropped_output = self.dropout(pooled_output)
        logits = self.classifier(dropped_output)
        
        return {
            "logits": logits,
            "hidden_states": outputs.last_hidden_state,
            "pooler_output": pooled_output
        }


class ClinicalBERTClassifier(MentalHealthClassifier):
    """Mental health classifier using ClinicalBERT."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ClinicalBERT classifier.
        
        Args:
            config: Model configuration dictionary.
        """
        # Override base model to use ClinicalBERT
        config = config.copy()
        config["base_model"] = "emilyalsentzer/Bio_ClinicalBERT"
        
        super().__init__(config)
        
        # Load ClinicalBERT-specific tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        
        logger.info("Initialized ClinicalBERT classifier for mental health monitoring")


class EmotionAwareClassifier(MentalHealthClassifier):
    """Enhanced classifier with emotion-specific attention."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize emotion-aware classifier.
        
        Args:
            config: Model configuration dictionary.
        """
        super().__init__(config)
        
        # Emotion-specific attention layers
        self.emotion_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Emotion-specific projection layers
        self.emotion_projections = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size)
            for _ in range(self.num_labels)
        ])
        
        # Final classification layer
        self.final_classifier = nn.Linear(self.hidden_size, self.num_labels)
        
        logger.info("Initialized emotion-aware classifier")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with emotion-specific attention.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            
        Returns:
            Dictionary containing logits and attention weights.
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Apply emotion-specific attention
        attended_output, attention_weights = self.emotion_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Global average pooling
        pooled_output = attended_output.mean(dim=1)
        
        # Apply dropout
        dropped_output = self.dropout(pooled_output)
        
        # Final classification
        logits = self.final_classifier(dropped_output)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "hidden_states": hidden_states,
            "pooler_output": pooled_output
        }


class UncertaintyAwareClassifier(MentalHealthClassifier):
    """Classifier with uncertainty quantification capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize uncertainty-aware classifier.
        
        Args:
            config: Model configuration dictionary.
        """
        super().__init__(config)
        
        # Monte Carlo dropout layers
        self.mc_dropout_layers = nn.ModuleList([
            nn.Dropout(self.dropout_rate) for _ in range(3)
        ])
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info("Initialized uncertainty-aware classifier")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                mc_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty estimation.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            mc_samples: Number of Monte Carlo samples for uncertainty.
            
        Returns:
            Dictionary containing logits, uncertainty, and MC samples.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Monte Carlo dropout for uncertainty estimation
        mc_logits = []
        for _ in range(mc_samples):
            # Apply MC dropout
            dropped_output = pooled_output
            for dropout_layer in self.mc_dropout_layers:
                dropped_output = dropout_layer(dropped_output)
            
            # Get logits
            logits = self.classifier(dropped_output)
            mc_logits.append(logits)
        
        # Stack MC samples
        mc_logits = torch.stack(mc_logits, dim=0)  # [mc_samples, batch_size, num_labels]
        
        # Compute mean and variance
        mean_logits = mc_logits.mean(dim=0)
        logits_variance = mc_logits.var(dim=0)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_head(pooled_output)
        
        return {
            "logits": mean_logits,
            "logits_variance": logits_variance,
            "uncertainty": uncertainty,
            "mc_samples": mc_logits,
            "pooler_output": pooled_output
        }


def create_model(config: Dict[str, Any], model_type: str = "base") -> Tuple[nn.Module, Any]:
    """Create model instance based on configuration.
    
    Args:
        config: Model configuration dictionary.
        model_type: Type of model to create.
        
    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Creating {model_type} model")
    
    if model_type == "clinical_bert":
        model = ClinicalBERTClassifier(config)
        tokenizer = model.tokenizer
    elif model_type == "emotion_aware":
        model = EmotionAwareClassifier(config)
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    elif model_type == "uncertainty_aware":
        model = UncertaintyAwareClassifier(config)
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    else:
        model = MentalHealthClassifier(config)
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    
    return model, tokenizer


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get model information and statistics.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Dictionary containing model information.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "architecture": model.__class__.__name__
    }
