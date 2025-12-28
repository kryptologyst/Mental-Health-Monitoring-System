"""
Privacy and de-identification utilities for mental health monitoring.

This module provides tools for protecting patient privacy by identifying
and anonymizing personally identifiable information (PII) in text data.
"""

import re
from typing import List, Dict, Any, Optional
import logging
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


class PrivacyProtector:
    """Privacy protection utility for text de-identification."""
    
    def __init__(self, entities: Optional[List[str]] = None, redaction_char: str = "*"):
        """Initialize privacy protector.
        
        Args:
            entities: List of entity types to anonymize.
            redaction_char: Character used for redaction.
        """
        self.entities = entities or [
            "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", 
            "LOCATION", "DATE_TIME", "CREDIT_CARD", "SSN"
        ]
        self.redaction_char = redaction_char
        
        # Initialize Presidio engines
        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        except Exception as e:
            logger.warning(f"Failed to initialize Presidio engines: {e}")
            self.analyzer = None
            self.anonymizer = None
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII in text using Presidio.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            List of detected PII entities.
        """
        if not self.analyzer:
            return self._fallback_pii_detection(text)
        
        try:
            results = self.analyzer.analyze(text=text, entities=self.entities)
            return [
                {
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "text": text[result.start:result.end]
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"PII detection failed: {e}")
            return self._fallback_pii_detection(text)
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize PII in text.
        
        Args:
            text: Input text to anonymize.
            
        Returns:
            Anonymized text.
        """
        if not self.anonymizer:
            return self._fallback_anonymization(text)
        
        try:
            results = self.analyzer.analyze(text=text, entities=self.entities)
            anonymized_result = self.anonymizer.anonymize(
                text=text, 
                analyzer_results=results
            )
            return anonymized_result.text
        except Exception as e:
            logger.error(f"Text anonymization failed: {e}")
            return self._fallback_anonymization(text)
    
    def _fallback_pii_detection(self, text: str) -> List[Dict[str, Any]]:
        """Fallback PII detection using regex patterns.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            List of detected PII entities.
        """
        patterns = {
            "EMAIL_ADDRESS": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE_NUMBER": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
            "DATE_TIME": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        }
        
        detected = []
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                detected.append({
                    "entity_type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.8,  # Default confidence
                    "text": match.group()
                })
        
        return detected
    
    def _fallback_anonymization(self, text: str) -> str:
        """Fallback anonymization using regex replacement.
        
        Args:
            text: Input text to anonymize.
            
        Returns:
            Anonymized text.
        """
        anonymized_text = text
        
        # Email addresses
        anonymized_text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            f'{self.redaction_char * 10}@example.com',
            anonymized_text
        )
        
        # Phone numbers
        anonymized_text = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            f'{self.redaction_char * 3}-{self.redaction_char * 3}-{self.redaction_char * 4}',
            anonymized_text
        )
        
        # SSN
        anonymized_text = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            f'{self.redaction_char * 3}-{self.redaction_char * 2}-{self.redaction_char * 4}',
            anonymized_text
        )
        
        return anonymized_text
    
    def is_safe_for_logging(self, text: str) -> bool:
        """Check if text is safe for logging (no PII detected).
        
        Args:
            text: Text to check.
            
        Returns:
            True if text is safe for logging.
        """
        pii_entities = self.detect_pii(text)
        return len(pii_entities) == 0


def create_safe_log_entry(entry: Dict[str, Any], privacy_protector: PrivacyProtector) -> Dict[str, Any]:
    """Create a safe log entry by anonymizing PII.
    
    Args:
        entry: Original log entry.
        privacy_protector: Privacy protector instance.
        
    Returns:
        Safe log entry with anonymized text.
    """
    safe_entry = entry.copy()
    
    # Anonymize text fields
    text_fields = ["entry", "text", "content", "message"]
    for field in text_fields:
        if field in safe_entry and isinstance(safe_entry[field], str):
            safe_entry[field] = privacy_protector.anonymize_text(safe_entry[field])
    
    return safe_entry


def validate_privacy_compliance(text: str, privacy_protector: PrivacyProtector) -> Dict[str, Any]:
    """Validate privacy compliance of text.
    
    Args:
        text: Text to validate.
        privacy_protector: Privacy protector instance.
        
    Returns:
        Validation results including compliance status and detected PII.
    """
    pii_entities = privacy_protector.detect_pii(text)
    
    return {
        "is_compliant": len(pii_entities) == 0,
        "pii_count": len(pii_entities),
        "pii_entities": pii_entities,
        "risk_level": "high" if len(pii_entities) > 3 else "medium" if len(pii_entities) > 0 else "low"
    }
