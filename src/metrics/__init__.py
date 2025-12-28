"""
Evaluation metrics and calibration for mental health monitoring.

This module provides clinically meaningful evaluation metrics,
calibration tools, and uncertainty quantification for mental health
text classification models.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MentalHealthEvaluator:
    """Evaluator for mental health classification models."""
    
    def __init__(self, label_names: List[str]):
        """Initialize evaluator.
        
        Args:
            label_names: List of emotion label names.
        """
        self.label_names = label_names
        self.num_labels = len(label_names)
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional).
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro averages
        metrics["precision_macro"] = np.mean(precision)
        metrics["recall_macro"] = np.mean(recall)
        metrics["f1_macro"] = np.mean(f1)
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["precision_weighted"] = precision_weighted
        metrics["recall_weighted"] = recall_weighted
        metrics["f1_weighted"] = f1_weighted
        
        # Per-class metrics
        for i, label in enumerate(self.label_names):
            metrics[f"precision_{label}"] = precision[i]
            metrics[f"recall_{label}"] = recall[i]
            metrics[f"f1_{label}"] = f1[i]
            metrics[f"support_{label}"] = support[i]
        
        # AUC metrics (if probabilities provided)
        if y_proba is not None:
            try:
                # Multi-class AUC
                if self.num_labels > 2:
                    metrics["auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                    metrics["auc_ovo"] = roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro")
                else:
                    metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
                
                # Average Precision
                metrics["avg_precision"] = average_precision_score(y_true, y_proba, average="macro")
                
            except Exception as e:
                logger.warning(f"Could not compute AUC metrics: {e}")
        
        return metrics
    
    def compute_clinical_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Compute clinically relevant metrics.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            
        Returns:
            Dictionary of clinical metrics.
        """
        clinical_metrics = {}
        
        # Sensitivity and Specificity for each class
        for i, label in enumerate(self.label_names):
            # Binary classification for this class
            y_true_binary = (y_true == i).astype(int)
            y_proba_binary = y_proba[:, i]
            
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_proba_binary)
            
            # Find optimal threshold (Youden's J statistic)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Compute metrics at optimal threshold
            y_pred_binary = (y_proba_binary >= optimal_threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            clinical_metrics[f"sensitivity_{label}"] = sensitivity
            clinical_metrics[f"specificity_{label}"] = specificity
            clinical_metrics[f"ppv_{label}"] = ppv
            clinical_metrics[f"npv_{label}"] = npv
            clinical_metrics[f"optimal_threshold_{label}"] = optimal_threshold
        
        return clinical_metrics
    
    def compute_calibration_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Compute calibration metrics.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            
        Returns:
            Dictionary of calibration metrics.
        """
        calibration_metrics = {}
        
        # Expected Calibration Error (ECE)
        ece_scores = []
        for i in range(self.num_labels):
            y_true_binary = (y_true == i).astype(int)
            y_proba_binary = y_proba[:, i]
            
            ece = self._compute_ece(y_true_binary, y_proba_binary)
            ece_scores.append(ece)
            calibration_metrics[f"ece_{self.label_names[i]}"] = ece
        
        calibration_metrics["ece_macro"] = np.mean(ece_scores)
        
        # Brier Score
        brier_scores = []
        for i in range(self.num_labels):
            y_true_binary = (y_true == i).astype(int)
            y_proba_binary = y_proba[:, i]
            
            brier_score = np.mean((y_proba_binary - y_true_binary) ** 2)
            brier_scores.append(brier_score)
            calibration_metrics[f"brier_{self.label_names[i]}"] = brier_score
        
        calibration_metrics["brier_macro"] = np.mean(brier_scores)
        
        return calibration_metrics
    
    def _compute_ece(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error.
        
        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities.
            n_bins: Number of bins for calibration.
            
        Returns:
            Expected Calibration Error.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create confusion matrix visualization.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_names, yticklabels=self.label_names,
                   ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_calibration_plot(self, y_true: np.ndarray, y_proba: np.ndarray,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create calibration plot.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, label in enumerate(self.label_names):
            y_true_binary = (y_true == i).astype(int)
            y_proba_binary = y_proba[:, i]
            
            # Compute calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binary, y_proba_binary, n_bins=10
            )
            
            # Plot calibration curve
            axes[i].plot(mean_predicted_value, fraction_of_positives, "s-", 
                        label=f"{label}")
            axes[i].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            axes[i].set_xlabel('Mean Predicted Probability')
            axes[i].set_ylabel('Fraction of Positives')
            axes[i].set_title(f'Calibration Plot - {label}')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                         save_path: Optional[str] = None) -> plt.Figure:
        """Create ROC curves for each class.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, label in enumerate(self.label_names):
            y_true_binary = (y_true == i).astype(int)
            y_proba_binary = y_proba[:, i]
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba_binary)
            auc = roc_auc_score(y_true_binary, y_proba_binary)
            
            # Plot ROC curve
            axes[i].plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})')
            axes[i].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'ROC Curve - {label}')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def compute_uncertainty_metrics(y_proba: np.ndarray, y_proba_variance: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute uncertainty quantification metrics.
    
    Args:
        y_proba: Predicted probabilities.
        y_proba_variance: Variance of predicted probabilities (optional).
        
    Returns:
        Dictionary of uncertainty metrics.
    """
    uncertainty_metrics = {}
    
    # Entropy-based uncertainty
    entropy = -np.sum(y_proba * np.log(y_proba + 1e-8), axis=1)
    uncertainty_metrics["mean_entropy"] = np.mean(entropy)
    uncertainty_metrics["std_entropy"] = np.std(entropy)
    
    # Max probability uncertainty
    max_proba = np.max(y_proba, axis=1)
    uncertainty_metrics["mean_max_proba"] = np.mean(max_proba)
    uncertainty_metrics["std_max_proba"] = np.std(max_proba)
    
    # Confidence intervals
    uncertainty_metrics["low_confidence_rate"] = np.mean(max_proba < 0.5)
    uncertainty_metrics["high_confidence_rate"] = np.mean(max_proba > 0.8)
    
    # Variance-based uncertainty (if available)
    if y_proba_variance is not None:
        mean_variance = np.mean(y_proba_variance, axis=1)
        uncertainty_metrics["mean_variance"] = np.mean(mean_variance)
        uncertainty_metrics["std_variance"] = np.std(mean_variance)
    
    return uncertainty_metrics


def create_evaluation_report(metrics: Dict[str, float], 
                           clinical_metrics: Dict[str, float],
                           calibration_metrics: Dict[str, float],
                           uncertainty_metrics: Dict[str, float]) -> str:
    """Create comprehensive evaluation report.
    
    Args:
        metrics: Basic classification metrics.
        clinical_metrics: Clinical metrics.
        calibration_metrics: Calibration metrics.
        uncertainty_metrics: Uncertainty metrics.
        
    Returns:
        Formatted evaluation report.
    """
    report = "=" * 80 + "\n"
    report += "MENTAL HEALTH MONITORING - EVALUATION REPORT\n"
    report += "=" * 80 + "\n\n"
    
    # Basic Metrics
    report += "CLASSIFICATION METRICS:\n"
    report += "-" * 40 + "\n"
    report += f"Accuracy: {metrics.get('accuracy', 0):.4f}\n"
    report += f"F1 Macro: {metrics.get('f1_macro', 0):.4f}\n"
    report += f"F1 Weighted: {metrics.get('f1_weighted', 0):.4f}\n"
    report += f"AUC OVR: {metrics.get('auc_ovr', 0):.4f}\n"
    report += f"Average Precision: {metrics.get('avg_precision', 0):.4f}\n\n"
    
    # Clinical Metrics
    report += "CLINICAL METRICS:\n"
    report += "-" * 40 + "\n"
    emotions = ["neutral", "happy", "sad", "anxious", "angry", "fearful"]
    for emotion in emotions:
        if f"sensitivity_{emotion}" in clinical_metrics:
            report += f"{emotion.upper()}:\n"
            report += f"  Sensitivity: {clinical_metrics[f'sensitivity_{emotion}']:.4f}\n"
            report += f"  Specificity: {clinical_metrics[f'specificity_{emotion}']:.4f}\n"
            report += f"  PPV: {clinical_metrics[f'ppv_{emotion}']:.4f}\n"
            report += f"  NPV: {clinical_metrics[f'npv_{emotion}']:.4f}\n\n"
    
    # Calibration Metrics
    report += "CALIBRATION METRICS:\n"
    report += "-" * 40 + "\n"
    report += f"ECE Macro: {calibration_metrics.get('ece_macro', 0):.4f}\n"
    report += f"Brier Score Macro: {calibration_metrics.get('brier_macro', 0):.4f}\n\n"
    
    # Uncertainty Metrics
    report += "UNCERTAINTY METRICS:\n"
    report += "-" * 40 + "\n"
    report += f"Mean Entropy: {uncertainty_metrics.get('mean_entropy', 0):.4f}\n"
    report += f"Mean Max Probability: {uncertainty_metrics.get('mean_max_proba', 0):.4f}\n"
    report += f"Low Confidence Rate: {uncertainty_metrics.get('low_confidence_rate', 0):.4f}\n"
    report += f"High Confidence Rate: {uncertainty_metrics.get('high_confidence_rate', 0):.4f}\n\n"
    
    report += "=" * 80 + "\n"
    report += "DISCLAIMER: This is a research demonstration tool and should not\n"
    report += "be used for clinical diagnosis or medical decision-making.\n"
    report += "=" * 80 + "\n"
    
    return report
