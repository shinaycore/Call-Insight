# nodes/text_preprocessing.py
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from utils.json_reader import load_json
from utils.logger import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """Preprocess transcripts: clean text, remove fillers, redact PII"""

    def __init__(
        self, results_dir: str = "preprocessed_results", redact_pii: bool = True
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.redact_pii = redact_pii

        try:
            self.nlp = spacy.load("en_core_web_lg")
            if redact_pii:
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                logger.info("Models loaded: spaCy (lg) + Presidio")
        except OSError:
            raise OSError("Run: python -m spacy download en_core_web_lg")

    # ------------------------
    # Cleaning Stage
    # ------------------------
    def clean_text(self, text: str) -> str:
        text = text.lower()

        text = re.sub(r"\b(uh+|umm+|mm+|ah+|oh+)\b", "", text)

        phrase_fillers = ["you know", "i mean", "sort of", "kind of"]
        for phrase in phrase_fillers:
            text = re.sub(rf"\b{phrase}\b", "", text)

        text = re.sub(r"([.!?,;:]){2,}", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    # ------------------------
    # Filler Word Removal
    # ------------------------
    def remove_fillers(self, text: str) -> str:
        doc = self.nlp(text)
        words = [t.text for t in doc if not t.is_punct]

        counter = Counter(words)
        total = len(words)

        dynamic_fillers = {
            w
            for w, c in counter.items()
            if len(w) <= 3
            and c / max(total, 1) > 0.2
            and w not in {"yes", "no", "hey", "sir"}
        }

        tokens = [
            t.text
            for t in doc
            if t.pos_ != "INTJ"
            and t.text.lower() not in dynamic_fillers
            and not t.is_punct
            and not t.is_space
        ]

        return " ".join(tokens)

    # ------------------------
    # PII Redaction
    # ------------------------
    def redact_pii_text(self, text: str) -> Dict[str, Any]:
        if not self.redact_pii or not text.strip():
            return {"text": text, "entities": []}

        try:
            results = self.analyzer.analyze(text=text, language="en")
            anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
            entities = [{"type": r.entity_type, "score": r.score} for r in results]

            return {"text": anonymized.text, "entities": entities}

        except Exception:
            logger.exception("PII redaction failed")
            return {"text": text, "entities": []}

    # ------------------------
    # Full Pipeline
    # ------------------------
    def preprocess_transcript(
        self, transcript_json: str, request_id: str = "request"
    ) -> Dict[str, Any]:
        logger.info(f"Processing transcript: {request_id}")

        transcript = load_json(transcript_json)

        if isinstance(transcript, dict):
            transcript = transcript.get("results", [])

        speaker_texts = {}
        pii_summary = {}

        for entry in transcript:
            speaker = entry.get("speaker", "unknown")
            raw = entry.get("text", "")

            if len(raw.strip()) < 2:
                continue

            cleaned = self.clean_text(raw)
            cleaned = self.remove_fillers(cleaned)
            redacted = self.redact_pii_text(cleaned)

            if redacted["entities"]:
                pii_summary.setdefault(speaker, []).extend(redacted["entities"])

            speaker_texts.setdefault(speaker, []).append(
                redacted["text"].strip() + "\n"
            )

        # merge per speaker
        speaker_texts = {
            spk: "".join(texts).strip() for spk, texts in speaker_texts.items()
        }

        # PII stats
        pii_stats = {
            spk: dict(Counter(ent["type"] for ent in ents))
            for spk, ents in pii_summary.items()
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ------------------------
        # NEW: Save TXT + JSON
        # ------------------------

        # 1. Save JSON
        json_path = self.results_dir / f"{request_id}_{ts}.json"
        output = {
            "request_id": request_id,
            "speaker_texts": speaker_texts,
            "pii_stats": pii_stats,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # 2. Save TXT (speaker-wise + combined)
        txt_path = self.results_dir / f"{request_id}_{ts}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("PREPROCESSED TRANSCRIPT\n")
            f.write("=======================\n\n")

            for spk, text in speaker_texts.items():
                f.write(f"{spk.upper()}:\n{text}\n\n")

            f.write("\nPII SUMMARY:\n")
            f.write(json.dumps(pii_stats, indent=2))

        logger.info(f"Saved JSON to: {json_path}")
        logger.info(f"Saved TXT to:  {txt_path}")

        # preview
        for speaker, text in speaker_texts.items():
            preview = text[:200] + "..." if len(text) > 200 else text

        return {
            "speaker_texts": speaker_texts,
            "pii_summary": pii_stats,
            "saved_json_path": str(json_path),
            "saved_txt_path": str(txt_path),  # NEW
        }
