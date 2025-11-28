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
    def clean_text(self, text: str, preserve_case: bool = True) -> str:
        """
        Clean text while preserving readability

        Args:
            text: Input text
            preserve_case: If True, keeps original casing for better readability
        """
        # Remove excessive filler words (uh, umm, etc.) but keep natural speech
        text = re.sub(r"\b(uh+|umm+|mm+|ah+|oh+)\b", "", text, flags=re.IGNORECASE)

        # Remove excessive repetitions of punctuation
        text = re.sub(r"([.!?,;:]){2,}", r"\1", text)

        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ------------------------
    # Smart Filler Removal
    # ------------------------
    def remove_fillers(self, text: str, aggressive: bool = False) -> str:
        """
        Remove filler words intelligently without destroying readability

        Args:
            text: Input text
            aggressive: If True, removes more aggressively (may harm readability)
        """
        if not aggressive:
            # Light cleaning - only remove obvious fillers
            filler_patterns = [
                r"\b(like|you know|i mean|sort of|kind of|basically)\b",
                r"\b(actually|literally)\b(?=.*\b(actually|literally)\b)",  # Only if repeated
            ]

            for pattern in filler_patterns:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        else:
            # Aggressive cleaning using spaCy
            doc = self.nlp(text)
            words = [t.text for t in doc if not t.is_punct]

            counter = Counter(words)
            total = len(words)

            # Find words that appear too frequently
            dynamic_fillers = {
                w
                for w, c in counter.items()
                if len(w) <= 3
                and c / max(total, 1) > 0.2
                and w.lower() not in {"yes", "no", "hey", "sir"}
            }

            tokens = [
                t.text
                for t in doc
                if t.pos_ != "INTJ" and t.text.lower() not in dynamic_fillers
            ]

            text = " ".join(tokens)

        # Clean up spacing
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------
    # PII Redaction (Improved)
    # ------------------------
    def redact_pii_text(
        self, text: str, use_generic_labels: bool = True
    ) -> Dict[str, Any]:
        """
        Redact PII with readable replacements

        Args:
            text: Input text
            use_generic_labels: If True, uses readable labels instead of brackets
        """
        if not self.redact_pii or not text.strip():
            return {"text": text, "entities": []}

        try:
            results = self.analyzer.analyze(text=text, language="en")

            if not use_generic_labels:
                # Use Presidio's default anonymization
                anonymized = self.anonymizer.anonymize(
                    text=text, analyzer_results=results
                )
                entities = [{"type": r.entity_type, "score": r.score} for r in results]
                return {"text": anonymized.text, "entities": entities}

            # Custom anonymization with readable replacements
            entity_map = {
                "PERSON": "[NAME]",
                "EMAIL_ADDRESS": "[EMAIL]",
                "PHONE_NUMBER": "[PHONE]",
                "LOCATION": "[LOCATION]",
                "DATE_TIME": "[DATE]",
                "CREDIT_CARD": "[CARD]",
                "CRYPTO": "[CRYPTO]",
                "IBAN_CODE": "[IBAN]",
                "IP_ADDRESS": "[IP]",
                "NRP": "[ID]",
                "MEDICAL_LICENSE": "[LICENSE]",
                "US_SSN": "[SSN]",
            }

            # Sort by start position (reverse) to replace from end to start
            sorted_results = sorted(results, key=lambda x: x.start, reverse=True)

            entities = []
            redacted_text = text

            for result in sorted_results:
                replacement = entity_map.get(
                    result.entity_type, f"[{result.entity_type}]"
                )
                redacted_text = (
                    redacted_text[: result.start]
                    + replacement
                    + redacted_text[result.end :]
                )
                entities.append(
                    {
                        "type": result.entity_type,
                        "score": result.score,
                        "original": text[result.start : result.end],
                    }
                )

            return {"text": redacted_text, "entities": entities}

        except Exception as e:
            logger.exception(f"PII redaction failed: {e}")
            return {"text": text, "entities": []}

    # ------------------------
    # Full Pipeline
    # ------------------------
    def preprocess_transcript(
        self,
        transcript_json: str,
        request_id: str = "request",
        cleaning_mode: str = "light",  # "light", "moderate", "aggressive"
    ) -> Dict[str, Any]:
        """
        Preprocess transcript with configurable cleaning levels

        Args:
            transcript_json: Path to transcript JSON
            request_id: Request ID
            cleaning_mode:
                - "light": Minimal cleaning, preserve most text
                - "moderate": Remove obvious fillers, clean formatting
                - "aggressive": Heavy cleaning (may reduce readability)
        """
        logger.info(f"Processing transcript: {request_id} (mode: {cleaning_mode})")

        transcript = load_json(transcript_json)

        if isinstance(transcript, dict):
            transcript = transcript.get("results", [])

        speaker_texts = {}
        speaker_texts_cleaned = {}
        pii_summary = {}

        for entry in transcript:
            speaker = entry.get("speaker", "unknown")
            raw = entry.get("text", "")

            if len(raw.strip()) < 2:
                continue

            # Extract timestamp if present in format [MM:SS - MM:SS] Speaker: text
            timestamp = ""
            text_content = raw

            timestamp_match = re.match(
                r"\[(\d{2}:\d{2} - \d{2}:\d{2})\]\s*Speaker \d+:\s*(.*)", raw
            )
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                text_content = timestamp_match.group(2)

            # Apply cleaning based on mode
            if cleaning_mode == "light":
                cleaned = self.clean_text(text_content, preserve_case=True)
            elif cleaning_mode == "moderate":
                cleaned = self.clean_text(text_content, preserve_case=True)
                cleaned = self.remove_fillers(cleaned, aggressive=False)
            else:  # aggressive
                cleaned = self.clean_text(text_content, preserve_case=False)
                cleaned = self.remove_fillers(cleaned, aggressive=True)

            # Redact PII
            redacted = self.redact_pii_text(cleaned, use_generic_labels=True)

            if redacted["entities"]:
                pii_summary.setdefault(speaker, []).extend(redacted["entities"])

            # Store with timestamp for reference
            formatted_text = (
                f"[{timestamp}] {redacted['text']}" if timestamp else redacted["text"]
            )

            speaker_texts.setdefault(speaker, []).append(formatted_text.strip())
            speaker_texts_cleaned.setdefault(speaker, []).append(
                redacted["text"].strip()
            )

        # Merge per speaker - use cleaned version without timestamps for better summarization
        speaker_texts_for_summary = {
            spk: " ".join(texts) for spk, texts in speaker_texts_cleaned.items()
        }

        speaker_texts_with_timestamps = {
            spk: "\n".join(texts) for spk, texts in speaker_texts.items()
        }

        # PII stats
        pii_stats = {
            spk: dict(Counter(ent["type"] for ent in ents))
            for spk, ents in pii_summary.items()
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ------------------------
        # Save JSON + TXT
        # ------------------------

        # 1. Save JSON (for downstream processing)
        json_path = self.results_dir / f"{request_id}_{ts}.json"
        output = {
            "request_id": request_id,
            "speaker_texts": speaker_texts_for_summary,  # Clean version
            "pii_stats": pii_stats,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # 2. Save TXT for LLM (clean, readable format)
        txt_path = self.results_dir / f"{request_id}_{ts}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("TRANSCRIPT FOR SUMMARIZATION\n")
            f.write("=" * 50 + "\n\n")

            for spk, text in speaker_texts_for_summary.items():
                f.write(f"{spk.upper()}:\n{text}\n\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("Note: PII has been redacted for privacy\n")

        # 3. Save detailed version with timestamps (for reference)
        detailed_path = self.results_dir / f"{request_id}_{ts}_detailed.txt"
        with open(detailed_path, "w", encoding="utf-8") as f:
            f.write("DETAILED TRANSCRIPT WITH TIMESTAMPS\n")
            f.write("=" * 50 + "\n\n")

            for spk, text in speaker_texts_with_timestamps.items():
                f.write(f"{spk.upper()}:\n{text}\n\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("PII REDACTION SUMMARY:\n")
            f.write(json.dumps(pii_stats, indent=2))

        logger.info(f"Saved JSON to: {json_path}")
        logger.info(f"Saved TXT (for LLM) to: {txt_path}")
        logger.info(f"Saved detailed TXT to: {detailed_path}")

        return {
            "speaker_texts": speaker_texts_for_summary,
            "pii_summary": pii_stats,
            "saved_json_path": str(json_path),
            "saved_txt_path": str(txt_path),
            "saved_detailed_path": str(detailed_path),
        }
