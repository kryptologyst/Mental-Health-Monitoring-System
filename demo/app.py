"""
Streamlit demo application for mental health monitoring.

This application provides an interactive interface for users to input
text and receive mental health sentiment analysis with explanations.
"""

import streamlit as st
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import ConfigManager, get_device
from models import create_model
from utils.explainability import MentalHealthExplainer, create_explanation_report
from utils.privacy import PrivacyProtector, validate_privacy_compliance
from data import MentalHealthDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Mental Health Monitoring Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .emotion-neutral { color: #6c757d; }
    .emotion-happy { color: #28a745; }
    .emotion-sad { color: #007bff; }
    .emotion-anxious { color: #ffc107; }
    .emotion-angry { color: #dc3545; }
    .emotion-fearful { color: #6f42c1; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "privacy_protector" not in st.session_state:
    st.session_state.privacy_protector = None


@st.cache_resource
def load_model():
    """Load the trained model."""
    try:
        # Load configuration
        config_manager = ConfigManager("configs/default.yaml")
        config = config_manager.config
        
        # Setup device
        device = get_device()
        
        # Create model
        model, tokenizer = create_model(config.model, config.model.get("type", "base"))
        
        # Load checkpoint if available
        checkpoint_path = "checkpoints/best_model.pt"
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded trained model from checkpoint")
        else:
            logger.warning("No checkpoint found, using untrained model")
        
        model.eval()
        return model, tokenizer, config
        
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None


def create_emotion_visualization(probabilities: np.ndarray) -> go.Figure:
    """Create emotion probability visualization."""
    emotions = ["Neutral", "Happy", "Sad", "Anxious", "Angry", "Fearful"]
    colors = ["#6c757d", "#28a745", "#007bff", "#ffc107", "#dc3545", "#6f42c1"]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probabilities,
            marker_color=colors,
            text=[f"{p:.3f}" for p in probabilities],
            textposition="auto",
        )
    ])
    
    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis_title="Emotions",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig


def create_attention_visualization(explanation: Dict[str, Any]) -> go.Figure:
    """Create attention visualization."""
    tokens = explanation["tokens"]
    attention_scores = explanation["attention_scores"]
    
    if attention_scores is None:
        return None
    
    # Normalize attention scores
    attention_scores = np.abs(attention_scores)
    attention_scores = attention_scores / np.max(attention_scores) if np.max(attention_scores) > 0 else attention_scores
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(len(tokens))),
            y=attention_scores,
            text=tokens,
            textposition="auto",
            marker_color=attention_scores,
            marker_colorscale="Reds"
        )
    ])
    
    fig.update_layout(
        title="Token Attention Scores",
        xaxis_title="Token Position",
        yaxis_title="Attention Score",
        height=300
    )
    
    return fig


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Mental Health Monitoring Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
        This is a research demonstration tool and should NOT be used for clinical diagnosis, 
        medical decision-making, or treatment purposes. Always consult with qualified healthcare 
        professionals for mental health concerns. This tool is for educational and research 
        purposes only.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Privacy settings
        st.subheader("Privacy Settings")
        enable_deid = st.checkbox("Enable De-identification", value=True)
        
        # Model settings
        st.subheader("Model Settings")
        show_explanations = st.checkbox("Show Explanations", value=True)
        show_uncertainty = st.checkbox("Show Uncertainty", value=True)
        
        # Load model
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                model, tokenizer, config = load_model()
                if model is not None:
                    st.session_state.model_loaded = True
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.config = config
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model")
    
    # Main content
    if not st.session_state.model_loaded:
        st.info("Please load the model using the sidebar to begin.")
        return
    
    # Initialize privacy protector
    if st.session_state.privacy_protector is None:
        st.session_state.privacy_protector = PrivacyProtector()
    
    # Text input
    st.header("Text Analysis")
    
    # Example texts
    st.subheader("Example Texts")
    example_texts = [
        "I'm feeling really anxious about my job interview tomorrow.",
        "Had an amazing day today! Got promoted and celebrated with friends.",
        "I've been feeling really down lately. Nothing seems to bring me joy anymore.",
        "I'm so frustrated with this situation. Why does everything have to be so difficult?",
        "Today was just a regular day. Nothing special happened.",
        "I'm scared about what might happen next. I feel uncertain about the future."
    ]
    
    selected_example = st.selectbox("Select an example text:", ["Custom input"] + example_texts)
    
    if selected_example == "Custom input":
        text_input = st.text_area(
            "Enter your text:",
            placeholder="Type your text here for mental health sentiment analysis...",
            height=100
        )
    else:
        text_input = st.text_area(
            "Enter your text:",
            value=selected_example,
            height=100
        )
    
    # Analyze button
    if st.button("Analyze Text", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        with st.spinner("Analyzing text..."):
            try:
                # Privacy check
                if enable_deid:
                    privacy_check = validate_privacy_compliance(
                        text_input, st.session_state.privacy_protector
                    )
                    
                    if not privacy_check["is_compliant"]:
                        st.warning(f"⚠️ PII detected: {privacy_check['pii_count']} entities found. "
                                 f"Risk level: {privacy_check['risk_level']}")
                        
                        # Show anonymized version
                        anonymized_text = st.session_state.privacy_protector.anonymize_text(text_input)
                        st.info(f"Anonymized text: {anonymized_text}")
                        text_input = anonymized_text
                
                # Tokenize input
                inputs = st.session_state.tokenizer(
                    text_input, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                
                # Move to device
                device = get_device()
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                # Get model prediction
                with torch.no_grad():
                    outputs = st.session_state.model(input_ids, attention_mask)
                    logits = outputs["logits"]
                    probabilities = torch.softmax(logits, dim=-1)
                    predicted_class = torch.argmax(logits, dim=-1).item()
                
                # Convert to numpy
                probabilities = probabilities[0].cpu().numpy()
                
                # Emotion labels
                emotions = ["neutral", "happy", "sad", "anxious", "angry", "fearful"]
                predicted_emotion = emotions[predicted_class]
                confidence = probabilities[predicted_class]
                
                # Display results
                st.header("Analysis Results")
                
                # Main prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Predicted Emotion",
                        predicted_emotion.upper(),
                        delta=f"{confidence:.3f}"
                    )
                
                with col2:
                    st.metric(
                        "Confidence",
                        f"{confidence:.3f}",
                        delta=f"{confidence*100:.1f}%"
                    )
                
                with col3:
                    # Confidence level
                    if confidence > 0.8:
                        conf_level = "High"
                        conf_color = "green"
                    elif confidence > 0.6:
                        conf_level = "Medium"
                        conf_color = "orange"
                    else:
                        conf_level = "Low"
                        conf_color = "red"
                    
                    st.metric(
                        "Confidence Level",
                        conf_level,
                        delta=None
                    )
                
                # Emotion probability chart
                st.subheader("Emotion Probability Distribution")
                fig = create_emotion_visualization(probabilities)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed probabilities
                st.subheader("Detailed Probabilities")
                prob_df = pd.DataFrame({
                    "Emotion": [e.capitalize() for e in emotions],
                    "Probability": probabilities,
                    "Percentage": probabilities * 100
                })
                st.dataframe(prob_df, use_container_width=True)
                
                # Explanations
                if show_explanations:
                    st.header("Explanation")
                    
                    # Create explainer
                    explainer = MentalHealthExplainer(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        device
                    )
                    
                    # Get explanation
                    explanation = explainer.create_explanation_summary(text_input)
                    
                    # Attention visualization
                    attention_fig = create_attention_visualization(explanation)
                    if attention_fig:
                        st.subheader("Token Attention")
                        st.plotly_chart(attention_fig, use_container_width=True)
                    
                    # Important tokens
                    if "attention_importance" in explanation:
                        st.subheader("Most Important Tokens")
                        tokens = explanation["tokens"]
                        importance = explanation["attention_importance"][:10]
                        
                        important_tokens = []
                        for i, idx in enumerate(importance):
                            if idx < len(tokens):
                                important_tokens.append({
                                    "Rank": i + 1,
                                    "Token": tokens[idx],
                                    "Position": idx
                                })
                        
                        if important_tokens:
                            st.dataframe(pd.DataFrame(important_tokens), use_container_width=True)
                    
                    # Explanation report
                    report = create_explanation_report(explanation)
                    st.subheader("Detailed Explanation")
                    st.text(report)
                
                # Uncertainty analysis
                if show_uncertainty and hasattr(st.session_state.model, 'forward'):
                    st.header("Uncertainty Analysis")
                    
                    # Compute uncertainty metrics
                    uncertainty_metrics = compute_uncertainty_metrics(
                        probabilities.reshape(1, -1)
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Entropy",
                            f"{uncertainty_metrics['mean_entropy']:.4f}",
                            help="Higher entropy indicates more uncertainty"
                        )
                    
                    with col2:
                        st.metric(
                            "Max Probability",
                            f"{uncertainty_metrics['mean_max_proba']:.4f}",
                            help="Lower max probability indicates more uncertainty"
                        )
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logger.error(f"Analysis error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Mental Health Monitoring Demo - Research Tool Only<br>
        Not for clinical use. Consult healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
