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
        text = re.sub(r"\b(uh+|umm+|mm+|ah+|oh+)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"([.!?,;:]){2,}", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------
    # Smart Filler Removal
    # ------------------------
    def remove_fillers(self, text: str, aggressive: bool = False) -> str:
        if not aggressive:
            filler_patterns = [
                r"\b(like|you know|i mean|sort of|kind of|basically)\b",
                r"\b(actually|literally)\b(?=.*\b(actually|literally)\b)",
            ]
            for pattern in filler_patterns:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        else:
            doc = self.nlp(text)
            words = [t.text for t in doc if not t.is_punct]
            counter = Counter(words)
            total = len(words)

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

        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------
    # PII Redaction
    # ------------------------
    def redact_pii_text(
        self, text: str, use_generic_labels: bool = True
    ) -> Dict[str, Any]:
        if not self.redact_pii or not text.strip():
            return {"text": text, "entities": []}

        try:
            results = self.analyzer.analyze(text=text, language="en")

            if not use_generic_labels:
                anonymized = self.anonymizer.anonymize(
                    text=text, analyzer_results=results
                )
                entities = [{"type": r.entity_type, "score": r.score} for r in results]
                return {"text": anonymized.text, "entities": entities}

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
    # NEW: Smart Text Chunking
    # ------------------------
    def chunk_into_sentences(self, text: str, max_chars: int = 150) -> List[str]:
        """Break text into natural sentence chunks for better readability."""
        # Split on sentence boundaries
        sentences = re.split(r"([.!?]+\s+)", text)

        chunks = []
        current_chunk = ""

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            delimiter = sentences[i + 1] if i + 1 < len(sentences) else ""

            # Combine short fragments
            if len(current_chunk) + len(sentence) + len(delimiter) <= max_chars:
                current_chunk += sentence + delimiter
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + delimiter

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    # ------------------------
    # Full Pipeline
    # ------------------------
    def preprocess_transcript(
        self,
        transcript_json: str,
        request_id: str = "request",
        cleaning_mode: str = "light",
    ) -> Dict[str, Any]:
        logger.info(f"Processing transcript: {request_id} (mode: {cleaning_mode})")

        transcript = load_json(transcript_json)
        if isinstance(transcript, dict):
            transcript = transcript.get("results", [])

        speaker_texts = {}
        speaker_texts_cleaned = {}
        speaker_sentences = {}  # NEW: Store sentences separately
        pii_summary = {}

        for entry in transcript:
            speaker = entry.get("speaker", "unknown")
            raw = entry.get("text", "")

            if len(raw.strip()) < 2:
                continue

            timestamp = ""
            text_content = raw

            ts_match = re.match(
                r"\[(\d{2}:\d{2} - \d{2}:\d{2})\]\s*Speaker \d+:\s*(.*)", raw
            )
            if ts_match:
                timestamp = ts_match.group(1)
                text_content = ts_match.group(2)

            if cleaning_mode == "light":
                cleaned = self.clean_text(text_content, preserve_case=True)
            elif cleaning_mode == "moderate":
                cleaned = self.clean_text(text_content)
                cleaned = self.remove_fillers(cleaned, aggressive=False)
            else:
                cleaned = self.clean_text(text_content, preserve_case=False)
                cleaned = self.remove_fillers(cleaned, aggressive=True)

            redacted = self.redact_pii_text(cleaned)

            if redacted["entities"]:
                pii_summary.setdefault(speaker, []).extend(redacted["entities"])

            formatted = (
                f"[{timestamp}] {redacted['text']}" if timestamp else redacted["text"]
            )

            speaker_texts.setdefault(speaker, []).append(formatted.strip())
            speaker_texts_cleaned.setdefault(speaker, []).append(
                redacted["text"].strip()
            )

        # ------------------------
        # NEW: Create sentence-based structure
        # ------------------------
        speaker_texts_for_summary = {}
        for spk, texts in speaker_texts_cleaned.items():
            # Join all fragments first
            full_text = " ".join(texts)

            # Break into readable sentences
            sentences = self.chunk_into_sentences(full_text, max_chars=150)
            speaker_sentences[spk] = sentences

            # Store as joined text for backward compatibility
            speaker_texts_for_summary[spk] = " ".join(sentences)

        speaker_texts_with_timestamps = {
            spk: "\n".join(texts) for spk, texts in speaker_texts.items()
        }

        pii_stats = {
            spk: dict(Counter(ent["type"] for ent in ents))
            for spk, ents in pii_summary.items()
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ------------------------
        # Save JSON
        # ------------------------
        json_path = self.results_dir / f"{request_id}_{ts}.json"
        output = {
            "request_id": request_id,
            "speaker_texts": speaker_texts_for_summary,
            "pii_stats": pii_stats,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # ------------------------
        # NEW: Improved Clean TXT with line breaks
        # ------------------------
        txt_path = self.results_dir / f"{request_id}_{ts}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("TRANSCRIPT FOR SUMMARIZATION\n")
            f.write("=" * 50 + "\n\n")

            for spk in sorted(
                speaker_sentences.keys(),
                key=lambda s: int(s.split()[-1]) if s.split()[-1].isdigit() else 9999,
            ):
                f.write(f"{spk}:\n")

                # Write sentences with proper line breaks
                for sentence in speaker_sentences[spk]:
                    f.write(f"{sentence}\n")

                f.write("\n")  # Extra line between speakers

            f.write("=" * 50 + "\nNote: PII redacted\n")

        # ------------------------
        # Detailed TXT
        # ------------------------
        detailed_path = self.results_dir / f"{request_id}_{ts}_detailed.txt"
        with open(detailed_path, "w", encoding="utf-8") as f:
            f.write("DETAILED TRANSCRIPT (WITH TIMESTAMPS)\n")
            f.write("=" * 50 + "\n\n")

            for spk, text in speaker_texts_with_timestamps.items():
                f.write(f"{spk.upper()}:\n{text}\n\n")

            f.write("\nPII SUMMARY:\n")
            f.write(json.dumps(pii_stats, indent=2))

        # ------------------------
        # Merged TXT with paragraphs
        # ------------------------
        merged_path = self.results_dir / f"{request_id}_{ts}_merged.txt"
        with open(merged_path, "w", encoding="utf-8") as f:
            f.write("MERGED TRANSCRIPT (ALL SPEAKERS)\n")
            f.write("=" * 60 + "\n\n")

            def speaker_sort_key(s):
                try:
                    return int(s.split()[-1])
                except:
                    return 9999

            for spk in sorted(speaker_sentences.keys(), key=speaker_sort_key):
                f.write(f"=== {spk.upper()} ===\n\n")

                # Write in paragraph form with line breaks every ~3 sentences
                sentences = speaker_sentences[spk]
                for i, sentence in enumerate(sentences):
                    f.write(sentence + " ")
                    # Add paragraph break every 3 sentences
                    if (i + 1) % 3 == 0:
                        f.write("\n\n")

                f.write("\n\n")

            f.write("=" * 60 + "\nEnd of merged transcript\n")

        logger.info(f"Saved JSON to: {json_path}")
        logger.info(f"Saved TXT to: {txt_path}")
        logger.info(f"Saved detailed TXT to: {detailed_path}")
        logger.info(f"Saved merged TXT to: {merged_path}")

        return {
            "speaker_texts": speaker_texts_for_summary,
            "pii_summary": pii_stats,
            "saved_json_path": str(json_path),
            "saved_txt_path": str(txt_path),
            "saved_detailed_path": str(detailed_path),
            "saved_merged_path": str(merged_path),
        }
