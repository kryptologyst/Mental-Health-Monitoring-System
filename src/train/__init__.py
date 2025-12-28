"""
Training module for mental health monitoring models.

This module provides training loops, loss functions, and optimization
utilities for mental health text classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, f1_score
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class.
            gamma: Focusing parameter.
            reduction: Reduction method.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model predictions (logits).
            targets: Ground truth labels.
            
        Returns:
            Focal loss value.
        """
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """Initialize label smoothing loss.
        
        Args:
            num_classes: Number of classes.
            smoothing: Smoothing factor.
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss.
        
        Args:
            inputs: Model predictions (logits).
            targets: Ground truth labels.
            
        Returns:
            Label smoothing loss value.
        """
        log_preds = nn.LogSoftmax(dim=-1)(inputs)
        true_dist = torch.zeros_like(log_preds)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))


class MentalHealthTrainer:
    """Trainer for mental health classification models."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            config: Training configuration.
            device: Device to train on.
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        # Initialize loss function
        self.loss_fn = self._get_loss_function()
        
        # Initialize scheduler
        self.scheduler = None
        
        # Training history
        self.train_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": []
        }
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_model_state = None
    
    def _get_loss_function(self) -> nn.Module:
        """Get loss function based on configuration.
        
        Returns:
            Loss function.
        """
        loss_type = self.config.get("loss_type", "cross_entropy")
        
        if loss_type == "focal":
            return FocalLoss(
                alpha=self.config.get("focal_alpha", 1.0),
                gamma=self.config.get("focal_gamma", 2.0)
            )
        elif loss_type == "label_smoothing":
            return LabelSmoothingLoss(
                num_classes=self.config.get("num_labels", 6),
                smoothing=self.config.get("smoothing", 0.1)
            )
        else:
            return nn.CrossEntropyLoss()
    
    def setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler.
        
        Args:
            num_training_steps: Total number of training steps.
        """
        warmup_steps = self.config.get("warmup_steps", 100)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            logits = outputs["logits"]
            
            # Compute loss
            loss = self.loss_fn(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get("max_grad_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["max_grad_norm"]
                )
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                logits = outputs["logits"]
                
                # Compute loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Compute epoch metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
            "predictions": np.array(all_predictions),
            "probabilities": np.array(all_probabilities)
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              save_dir: str = "checkpoints") -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            save_dir: Directory to save checkpoints.
            
        Returns:
            Training history.
        """
        logger.info("Starting training...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup scheduler
        num_training_steps = len(train_loader) * self.config["num_epochs"]
        self.setup_scheduler(num_training_steps)
        
        # Training loop
        for epoch in range(self.config["num_epochs"]):
            logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update history
            self.train_history["train_loss"].append(train_metrics["loss"])
            self.train_history["train_acc"].append(train_metrics["accuracy"])
            self.train_history["val_loss"].append(val_metrics["loss"])
            self.train_history["val_acc"].append(val_metrics["accuracy"])
            self.train_history["val_f1"].append(val_metrics["f1"])
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.4f}, "
                       f"Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint_path = os.path.join(save_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_f1": val_metrics["f1"],
                    "config": self.config
                }, checkpoint_path)
                
                logger.info(f"New best model saved with F1: {val_metrics['f1']:.4f}")
            
            # Save last checkpoint
            last_checkpoint_path = os.path.join(save_dir, "last_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_f1": val_metrics["f1"],
                "config": self.config
            }, last_checkpoint_path)
        
        logger.info("Training completed!")
        return self.train_history
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Checkpoint information.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        logger.info(f"Validation F1: {checkpoint['val_f1']:.4f}")
        
        return checkpoint
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history.
        
        Args:
            save_path: Path to save the plot.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        epochs = range(1, len(self.train_history["train_loss"]) + 1)
        
        # Loss plot
        axes[0].plot(epochs, self.train_history["train_loss"], label="Train Loss")
        axes[0].plot(epochs, self.train_history["val_loss"], label="Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(epochs, self.train_history["train_acc"], label="Train Accuracy")
        axes[1].plot(epochs, self.train_history["val_acc"], label="Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True)
        
        # F1 plot
        axes[2].plot(epochs, self.train_history["val_f1"], label="Validation F1")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("F1 Score")
        axes[2].set_title("Validation F1 Score")
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_data_collator(tokenizer: Any, max_length: int = 512):
    """Create data collator for training.
    
    Args:
        tokenizer: Tokenizer for the model.
        max_length: Maximum sequence length.
        
    Returns:
        Data collator function.
    """
    def collate_fn(batch):
        """Collate function for data loader."""
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]
        
        # Tokenize texts
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "label": torch.tensor(labels, dtype=torch.long)
        }
    
    return collate_fn
