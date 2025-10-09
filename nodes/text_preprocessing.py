# nodes/text_preprocessing.py
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter

import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from utils.logger import get_logger
from utils.json_reader import load_json

logger = get_logger(__name__)


class TextPreprocessor:
    """Preprocess transcripts: clean text, remove fillers, redact PII"""
    
    def __init__(self, results_dir: str = "preprocessed_results", redact_pii: bool = True):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.redact_pii = redact_pii
        
        # Load models
        try:
            self.nlp = spacy.load("en_core_web_sm")
            if redact_pii:
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                logger.info("Models loaded: spaCy + Presidio")
        except OSError:
            raise OSError("Run: python -m spacy download en_core_web_sm")
    
    def clean_text(self, text: str) -> str:
        """Remove elongated sounds, normalize whitespace"""
        text = text.lower()
        text = re.sub(r'\b(u+h+|m+h+|h+m+|a+h+|o+h+|e+r+)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'([.!?,;:]){2,}', r'\1', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def remove_fillers(self, text: str) -> str:
        """Remove interjections and repetitive short words"""
        doc = self.nlp(text)
        words = [t.text for t in doc if not t.is_punct]
        
        # Find frequent short words (dynamic fillers)
        counter = Counter(words)
        total = len(words)
        dynamic_fillers = {w for w, c in counter.items() if len(w) <= 3 and c / max(total, 1) > 0.1}
        
        # Common fillers
        all_fillers = dynamic_fillers | {"like", "you know", "i mean", "sort of", "kind of", "actually", "basically"}
        
        # Filter tokens
        tokens = [t.text for t in doc if t.pos_ != "INTJ" and t.text.lower() not in all_fillers and not t.is_punct and not t.is_space]
        return " ".join(tokens)
    
    def redact_pii_text(self, text: str) -> Dict[str, Any]:
        """Redact PII and return redacted text + entities found"""
        if not self.redact_pii or not text.strip():
            return {"text": text, "entities": []}
        
        try:
            results = self.analyzer.analyze(text=text, language="en")
            anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
            entities = [{"type": r.entity_type, "score": r.score} for r in results]
            return {"text": anonymized.text, "entities": entities}
        except Exception as e:
            logger.error(f"PII redaction failed: {e}")
            return {"text": text, "entities": []}
    
    def preprocess_transcript(self, transcript_json: str, request_id: str = "test") -> Dict[str, Any]:
        """Load transcript, clean, remove fillers, redact PII per speaker"""
        logger.info(f"Processing transcript: {request_id}")
        
        transcript = load_json(transcript_json)
        speaker_texts = {}
        pii_summary = {}
        
        for entry in transcript:
            speaker = entry.get("speaker", "unknown")
            raw = entry.get("text", "")
            
            if len(raw.strip()) < 10:
                continue
            
            # Clean -> Remove fillers -> Redact PII
            cleaned = self.clean_text(raw)
            cleaned = self.remove_fillers(cleaned)
            redacted = self.redact_pii_text(cleaned)
            
            # Track PII per speaker
            if redacted["entities"]:
                pii_summary.setdefault(speaker, []).extend(redacted["entities"])
            
            speaker_texts.setdefault(speaker, []).append(redacted["text"])
        
        # Merge per speaker
        speaker_texts = {spk: " ".join(texts) for spk, texts in speaker_texts.items()}
        
        # PII stats
        pii_stats = {spk: dict(Counter(e["type"] for e in ents)) for spk, ents in pii_summary.items()}
        
        # Save output
        output = {
            "request_id": request_id,
            "speaker_texts": speaker_texts,
            "pii_stats": pii_stats
        }
        
        save_path = self.results_dir / f"preprocessed_transcript_{request_id}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved: {save_path}")
        
        # Preview
        for speaker, text in speaker_texts.items():
            preview = text[:200] + "..." if len(text) > 200 else text
            logger.info(f"[{speaker}] {preview}")
        
        return {"speaker_texts": speaker_texts, "pii_summary": pii_stats, "saved_json_path": str(save_path)}