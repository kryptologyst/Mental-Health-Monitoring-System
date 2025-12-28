#!/usr/bin/env python3
"""
Evaluation script for mental health monitoring system.

This script evaluates a trained model on test data and generates
comprehensive evaluation reports and visualizations.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import DatasetDict
from omegaconf import DictConfig

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import ConfigManager, get_device
from data import MentalHealthDataProcessor
from models import create_model
from train import create_data_collator
from metrics import MentalHealthEvaluator, compute_uncertainty_metrics, create_evaluation_report


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("evaluation.log"),
            logging.StreamHandler()
        ]
    )


def load_model_and_data(checkpoint_path: str, config_path: str) -> tuple:
    """Load model and test data."""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.config
    
    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create model
    model, tokenizer = create_model(config.model, config.model.get("type", "base"))
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Checkpoint validation F1: {checkpoint.get('val_f1', 'N/A')}")
    
    # Prepare test data
    data_processor = MentalHealthDataProcessor(config.data)
    
    if config.data.get("data_path") and Path(config.data.data_path).exists():
        logger.info(f"Loading data from {config.data.data_path}")
        data = data_processor.load_data(config.data.data_path)
    else:
        logger.info("Generating synthetic test data")
        data = data_processor.generate_synthetic_data(config.data.synthetic_data_size)
    
    # Create dataset splits
    dataset = data_processor.create_dataset_splits(data)
    
    logger.info(f"Test dataset size: {len(dataset['test'])}")
    
    return model, tokenizer, dataset["test"], config, device


def evaluate_model(model: torch.nn.Module, tokenizer: Any, test_dataset: DatasetDict,
                   config: DictConfig, device: torch.device) -> Dict[str, Any]:
    """Evaluate the model on test data."""
    logger = logging.getLogger(__name__)
    
    # Create evaluator
    evaluator = MentalHealthEvaluator(["neutral", "happy", "sad", "anxious", "angry", "fearful"])
    
    # Create data loader
    collate_fn = create_data_collator(tokenizer, config.model.max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Evaluate model
    logger.info("Starting evaluation...")
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
    
    logger.info(f"Evaluated {len(y_true)} samples")
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
    clinical_metrics = evaluator.compute_clinical_metrics(y_true, y_proba)
    calibration_metrics = evaluator.compute_calibration_metrics(y_true, y_proba)
    uncertainty_metrics = compute_uncertainty_metrics(y_proba)
    
    # Create visualizations
    logger.info("Creating evaluation visualizations...")
    os.makedirs("assets", exist_ok=True)
    
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
    logger.info(f"Test AUC OVR: {metrics.get('auc_ovr', 'N/A')}")
    
    return {
        "metrics": metrics,
        "clinical_metrics": clinical_metrics,
        "calibration_metrics": calibration_metrics,
        "uncertainty_metrics": uncertainty_metrics,
        "predictions": y_pred,
        "probabilities": y_proba,
        "true_labels": y_true,
        "report": report
    }


def print_summary(results: Dict[str, Any]) -> None:
    """Print evaluation summary."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    metrics = results["metrics"]
    clinical_metrics = results["clinical_metrics"]
    calibration_metrics = results["calibration_metrics"]
    uncertainty_metrics = results["uncertainty_metrics"]
    
    print(f"\nCLASSIFICATION METRICS:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  Precision Macro: {metrics['precision_macro']:.4f}")
    print(f"  Recall Macro: {metrics['recall_macro']:.4f}")
    
    if 'auc_ovr' in metrics:
        print(f"  AUC OVR: {metrics['auc_ovr']:.4f}")
    if 'avg_precision' in metrics:
        print(f"  Average Precision: {metrics['avg_precision']:.4f}")
    
    print(f"\nCALIBRATION METRICS:")
    print(f"  ECE Macro: {calibration_metrics['ece_macro']:.4f}")
    print(f"  Brier Score Macro: {calibration_metrics['brier_macro']:.4f}")
    
    print(f"\nUNCERTAINTY METRICS:")
    print(f"  Mean Entropy: {uncertainty_metrics['mean_entropy']:.4f}")
    print(f"  Mean Max Probability: {uncertainty_metrics['mean_max_proba']:.4f}")
    print(f"  Low Confidence Rate: {uncertainty_metrics['low_confidence_rate']:.4f}")
    print(f"  High Confidence Rate: {uncertainty_metrics['high_confidence_rate']:.4f}")
    
    print(f"\nPER-CLASS CLINICAL METRICS:")
    emotions = ["neutral", "happy", "sad", "anxious", "angry", "fearful"]
    for emotion in emotions:
        if f"sensitivity_{emotion}" in clinical_metrics:
            print(f"  {emotion.upper()}:")
            print(f"    Sensitivity: {clinical_metrics[f'sensitivity_{emotion}']:.4f}")
            print(f"    Specificity: {clinical_metrics[f'specificity_{emotion}']:.4f}")
            print(f"    PPV: {clinical_metrics[f'ppv_{emotion}']:.4f}")
            print(f"    NPV: {clinical_metrics[f'npv_{emotion}']:.4f}")
    
    print("\n" + "="*80)
    print("DISCLAIMER: This evaluation is for research purposes only.")
    print("Results should not be used for clinical diagnosis or medical decisions.")
    print("="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate mental health monitoring model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="assets",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load model and data
        logger.info("Loading model and data...")
        model, tokenizer, test_dataset, config, device = load_model_and_data(
            args.checkpoint, args.config
        )
        
        # Evaluate model
        results = evaluate_model(model, tokenizer, test_dataset, config, device)
        
        # Print summary
        print_summary(results)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
