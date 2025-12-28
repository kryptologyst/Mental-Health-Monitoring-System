#!/usr/bin/env python3
"""
Main training script for mental health monitoring system.

This script provides a complete training pipeline for mental health
text classification models with proper evaluation and logging.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict
from omegaconf import DictConfig

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import ConfigManager, set_seed, get_device
from data import MentalHealthDataProcessor, create_sample_dataset
from models import create_model, get_model_info
from train import MentalHealthTrainer, create_data_collator
from metrics import MentalHealthEvaluator, compute_uncertainty_metrics, create_evaluation_report
from utils.privacy import PrivacyProtector


def setup_logging(config: DictConfig) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )


def prepare_data(config: DictConfig) -> DatasetDict:
    """Prepare training data."""
    logger = logging.getLogger(__name__)
    
    # Initialize data processor
    data_processor = MentalHealthDataProcessor(config.data)
    
    # Load or generate data
    if config.data.get("data_path") and Path(config.data.data_path).exists():
        logger.info(f"Loading data from {config.data.data_path}")
        data = data_processor.load_data(config.data.data_path)
    else:
        logger.info("Generating synthetic data")
        data = data_processor.generate_synthetic_data(config.data.synthetic_data_size)
    
    # Create dataset splits
    dataset = data_processor.create_dataset_splits(data)
    
    # Log data statistics
    logger.info(f"Dataset created with {len(dataset['train'])} train, "
               f"{len(dataset['validation'])} val, {len(dataset['test'])} test samples")
    
    return dataset


def create_model_and_tokenizer(config: DictConfig, device: torch.device):
    """Create model and tokenizer."""
    logger = logging.getLogger(__name__)
    
    # Create model
    model, tokenizer = create_model(config.model, config.model.get("type", "base"))
    
    # Move to device
    model = model.to(device)
    
    # Log model info
    model_info = get_model_info(model)
    logger.info(f"Model created: {model_info['architecture']}")
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    return model, tokenizer


def train_model(model: torch.nn.Module, tokenizer: Any, dataset: DatasetDict, 
                config: DictConfig, device: torch.device) -> MentalHealthTrainer:
    """Train the model."""
    logger = logging.getLogger(__name__)
    
    # Create data collator
    collate_fn = create_data_collator(tokenizer, config.model.max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset["train"],
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        dataset["validation"],
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create trainer
    trainer = MentalHealthTrainer(model, config.training, device)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader, "checkpoints")
    
    # Plot training history
    trainer.plot_training_history("assets/training_history.png")
    
    logger.info("Training completed!")
    return trainer


def evaluate_model(model: torch.nn.Module, tokenizer: Any, dataset: DatasetDict,
                   config: DictConfig, device: torch.device) -> Dict[str, Any]:
    """Evaluate the trained model."""
    logger = logging.getLogger(__name__)
    
    # Create evaluator
    evaluator = MentalHealthEvaluator(["neutral", "happy", "sad", "anxious", "angry", "fearful"])
    
    # Create data loader
    collate_fn = create_data_collator(tokenizer, config.model.max_length)
    test_loader = DataLoader(
        dataset["test"],
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Evaluate model
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probabilities)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
    clinical_metrics = evaluator.compute_clinical_metrics(y_true, y_proba)
    calibration_metrics = evaluator.compute_calibration_metrics(y_true, y_proba)
    uncertainty_metrics = compute_uncertainty_metrics(y_proba)
    
    # Create visualizations
    evaluator.create_confusion_matrix(y_true, y_pred, "assets/confusion_matrix.png")
    evaluator.create_calibration_plot(y_true, y_proba, "assets/calibration_plot.png")
    evaluator.create_roc_curves(y_true, y_proba, "assets/roc_curves.png")
    
    # Create evaluation report
    report = create_evaluation_report(metrics, clinical_metrics, calibration_metrics, uncertainty_metrics)
    
    # Save report
    with open("assets/evaluation_report.txt", "w") as f:
        f.write(report)
    
    logger.info("Evaluation completed!")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test F1 Macro: {metrics['f1_macro']:.4f}")
    
    return {
        "metrics": metrics,
        "clinical_metrics": clinical_metrics,
        "calibration_metrics": calibration_metrics,
        "uncertainty_metrics": uncertainty_metrics,
        "predictions": y_pred,
        "probabilities": y_proba,
        "true_labels": y_true
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train mental health monitoring model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--model-type", type=str, default="base",
                       choices=["base", "clinical_bert", "emotion_aware", "uncertainty_aware"],
                       help="Type of model to train")
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # Override model type if specified
    if args.model_type != "base":
        config.model.type = args.model_type
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Setup device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create assets directory
    os.makedirs("assets", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    try:
        # Prepare data
        logger.info("Preparing data...")
        dataset = prepare_data(config)
        
        # Create model and tokenizer
        logger.info("Creating model...")
        model, tokenizer = create_model_and_tokenizer(config, device)
        
        # Train model
        trainer = train_model(model, tokenizer, dataset, config, device)
        
        # Load best model
        best_checkpoint_path = "checkpoints/best_model.pt"
        if os.path.exists(best_checkpoint_path):
            trainer.load_checkpoint(best_checkpoint_path)
            logger.info("Loaded best model for evaluation")
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = evaluate_model(model, tokenizer, dataset, config, device)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
