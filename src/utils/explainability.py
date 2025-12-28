"""
Explainability and interpretability tools for mental health monitoring.

This module provides tools for understanding model predictions through
attention visualization, SHAP analysis, and uncertainty quantification.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from captum.attr import IntegratedGradients, Saliency, GradientShap
from captum.attr import visualization as viz
import shap
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class MentalHealthExplainer:
    """Explainability tools for mental health classification models."""
    
    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: torch.device):
        """Initialize explainer.
        
        Args:
            model: Trained mental health classifier.
            tokenizer: Tokenizer for the model.
            device: Device to run computations on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Initialize SHAP explainer
        self.shap_explainer = None
        
        # Label mapping
        self.label_names = ["neutral", "happy", "sad", "anxious", "angry", "fearful"]
    
    def explain_with_attention(self, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Explain prediction using attention weights.
        
        Args:
            text: Input text to explain.
            target_class: Target class for explanation (optional).
            
        Returns:
            Dictionary containing attention weights and tokens.
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get predicted class
            predicted_class = torch.argmax(logits, dim=-1).item()
            
            # Get attention weights if available
            attention_weights = None
            if "attention_weights" in outputs:
                attention_weights = outputs["attention_weights"]
            elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'encoder'):
                # Extract attention from transformer layers
                attention_weights = self._extract_attention_weights(input_ids, attention_mask)
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Compute attention scores
        attention_scores = None
        if attention_weights is not None:
            attention_scores = self._compute_attention_scores(attention_weights, target_class or predicted_class)
        
        return {
            "text": text,
            "tokens": tokens,
            "predicted_class": predicted_class,
            "predicted_emotion": self.label_names[predicted_class],
            "probabilities": probabilities[0].cpu().numpy(),
            "attention_scores": attention_scores,
            "attention_weights": attention_weights
        }
    
    def explain_with_gradients(self, text: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """Explain prediction using gradient-based methods.
        
        Args:
            text: Input text to explain.
            target_class: Target class for explanation (optional).
            
        Returns:
            Dictionary containing gradient attributions.
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Get predicted class
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs["logits"]
            predicted_class = torch.argmax(logits, dim=-1).item()
        
        target_class = target_class or predicted_class
        
        # Initialize gradient-based explainers
        integrated_gradients = IntegratedGradients(self.model)
        saliency = Saliency(self.model)
        
        # Compute attributions
        def model_forward(input_ids, attention_mask):
            return self.model(input_ids, attention_mask)["logits"]
        
        # Integrated Gradients
        ig_attributions = integrated_gradients.attribute(
            input_ids,
            target=target_class,
            additional_forward_args=(attention_mask,),
            n_steps=50
        )
        
        # Saliency
        saliency_attributions = saliency.attribute(
            input_ids,
            target=target_class,
            additional_forward_args=(attention_mask,)
        )
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return {
            "text": text,
            "tokens": tokens,
            "target_class": target_class,
            "target_emotion": self.label_names[target_class],
            "integrated_gradients": ig_attributions[0].cpu().numpy(),
            "saliency": saliency_attributions[0].cpu().numpy(),
            "predicted_class": predicted_class
        }
    
    def explain_with_shap(self, texts: List[str], max_samples: int = 100) -> Dict[str, Any]:
        """Explain predictions using SHAP values.
        
        Args:
            texts: List of input texts.
            max_samples: Maximum number of samples for SHAP computation.
            
        Returns:
            Dictionary containing SHAP values and explanations.
        """
        # Limit samples for computational efficiency
        if len(texts) > max_samples:
            texts = texts[:max_samples]
        
        # Tokenize all texts
        tokenized_texts = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            tokenized_texts.append({
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            })
        
        # Create SHAP explainer
        def model_predict(input_ids_batch):
            """Wrapper function for SHAP explainer."""
            predictions = []
            for input_ids in input_ids_batch:
                # Convert back to tensor
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
                attention_mask = torch.ones_like(input_ids).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs["logits"]
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions.append(probabilities[0].cpu().numpy())
            
            return np.array(predictions)
        
        # Initialize SHAP explainer
        explainer = shap.Explainer(model_predict, tokenized_texts[0]["input_ids"][0].numpy())
        
        # Compute SHAP values
        shap_values = explainer(tokenized_texts[0]["input_ids"][0].numpy())
        
        return {
            "shap_values": shap_values.values,
            "base_values": shap_values.base_values,
            "data": shap_values.data,
            "texts": texts
        }
    
    def visualize_attention(self, explanation: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """Visualize attention weights.
        
        Args:
            explanation: Explanation dictionary from explain_with_attention.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        tokens = explanation["tokens"]
        attention_scores = explanation["attention_scores"]
        
        if attention_scores is None:
            logger.warning("No attention scores available for visualization")
            return None
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Normalize attention scores
        attention_scores = np.abs(attention_scores)
        attention_scores = attention_scores / np.max(attention_scores)
        
        # Create color map
        colors = plt.cm.Reds(attention_scores)
        
        # Plot tokens with attention colors
        y_pos = np.arange(len(tokens))
        bars = ax.barh(y_pos, attention_scores, color=colors)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens)
        ax.set_xlabel('Attention Score')
        ax.set_title(f'Attention Visualization - {explanation["predicted_emotion"].upper()}')
        ax.invert_yaxis()
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attention Score')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_gradients(self, explanation: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """Visualize gradient attributions.
        
        Args:
            explanation: Explanation dictionary from explain_with_gradients.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        tokens = explanation["tokens"]
        ig_scores = explanation["integrated_gradients"]
        saliency_scores = explanation["saliency"]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Integrated Gradients
        ig_scores = np.abs(ig_scores)
        ig_scores = ig_scores / np.max(ig_scores) if np.max(ig_scores) > 0 else ig_scores
        
        colors1 = plt.cm.Blues(ig_scores)
        y_pos = np.arange(len(tokens))
        
        bars1 = ax1.barh(y_pos, ig_scores, color=colors1)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(tokens)
        ax1.set_xlabel('Integrated Gradients Score')
        ax1.set_title(f'Integrated Gradients - {explanation["target_emotion"].upper()}')
        ax1.invert_yaxis()
        
        # Saliency
        saliency_scores = np.abs(saliency_scores)
        saliency_scores = saliency_scores / np.max(saliency_scores) if np.max(saliency_scores) > 0 else saliency_scores
        
        colors2 = plt.cm.Greens(saliency_scores)
        
        bars2 = ax2.barh(y_pos, saliency_scores, color=colors2)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(tokens)
        ax2.set_xlabel('Saliency Score')
        ax2.set_title(f'Saliency - {explanation["target_emotion"].upper()}')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_explanation_summary(self, text: str) -> Dict[str, Any]:
        """Create comprehensive explanation summary.
        
        Args:
            text: Input text to explain.
            
        Returns:
            Dictionary containing all explanation methods.
        """
        # Get attention explanation
        attention_explanation = self.explain_with_attention(text)
        
        # Get gradient explanation
        gradient_explanation = self.explain_with_gradients(text)
        
        # Combine explanations
        summary = {
            "text": text,
            "predicted_class": attention_explanation["predicted_class"],
            "predicted_emotion": attention_explanation["predicted_emotion"],
            "probabilities": attention_explanation["probabilities"],
            "tokens": attention_explanation["tokens"],
            "attention_scores": attention_explanation["attention_scores"],
            "integrated_gradients": gradient_explanation["integrated_gradients"],
            "saliency": gradient_explanation["saliency"]
        }
        
        # Compute importance rankings
        if attention_explanation["attention_scores"] is not None:
            attention_importance = np.argsort(attention_explanation["attention_scores"])[::-1]
            summary["attention_importance"] = attention_importance.tolist()
        
        ig_importance = np.argsort(np.abs(gradient_explanation["integrated_gradients"]))[::-1]
        summary["ig_importance"] = ig_importance.tolist()
        
        saliency_importance = np.argsort(np.abs(gradient_explanation["saliency"]))[::-1]
        summary["saliency_importance"] = saliency_importance.tolist()
        
        return summary
    
    def _extract_attention_weights(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract attention weights from transformer layers.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            
        Returns:
            Attention weights tensor.
        """
        # This is a simplified version - in practice, you'd need to modify
        # the model to return attention weights
        with torch.no_grad():
            outputs = self.model.backbone(input_ids, attention_mask, output_attentions=True)
            # Average attention across all layers and heads
            attention_weights = torch.stack(outputs.attentions).mean(dim=(0, 2))  # [batch, seq, seq]
            return attention_weights
    
    def _compute_attention_scores(self, attention_weights: torch.Tensor, target_class: int) -> np.ndarray:
        """Compute attention scores for tokens.
        
        Args:
            attention_weights: Attention weights tensor.
            target_class: Target class for attention computation.
            
        Returns:
            Attention scores for each token.
        """
        # Average attention weights to get token importance
        # This is a simplified approach - you might want to use class-specific attention
        attention_scores = attention_weights[0].mean(dim=0).cpu().numpy()  # [seq_len]
        return attention_scores


def create_explanation_report(explanation: Dict[str, Any]) -> str:
    """Create human-readable explanation report.
    
    Args:
        explanation: Explanation dictionary.
        
    Returns:
        Formatted explanation report.
    """
    report = "=" * 80 + "\n"
    report += "MENTAL HEALTH MONITORING - EXPLANATION REPORT\n"
    report += "=" * 80 + "\n\n"
    
    report += f"Input Text: {explanation['text']}\n\n"
    report += f"Predicted Emotion: {explanation['predicted_emotion'].upper()}\n"
    report += f"Confidence: {explanation['probabilities'][explanation['predicted_class']]:.4f}\n\n"
    
    # Probability distribution
    report += "EMOTION PROBABILITIES:\n"
    report += "-" * 40 + "\n"
    for i, emotion in enumerate(["neutral", "happy", "sad", "anxious", "angry", "fearful"]):
        report += f"{emotion.capitalize()}: {explanation['probabilities'][i]:.4f}\n"
    
    report += "\n"
    
    # Top important tokens
    if "attention_importance" in explanation:
        report += "MOST IMPORTANT TOKENS (Attention):\n"
        report += "-" * 40 + "\n"
        tokens = explanation["tokens"]
        importance = explanation["attention_importance"][:10]  # Top 10
        for i, idx in enumerate(importance):
            if idx < len(tokens):
                report += f"{i+1}. {tokens[idx]}\n"
    
    report += "\n"
    report += "=" * 80 + "\n"
    report += "DISCLAIMER: This explanation is for research purposes only.\n"
    report += "It should not be used for clinical diagnosis or treatment decisions.\n"
    report += "=" * 80 + "\n"
    
    return report
