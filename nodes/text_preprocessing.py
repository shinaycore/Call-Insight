# nodes/text_preprocessing.py
# Reads your Whisper transcript JSON (diarized_transcript.json).
# Cleans filler words, elongated sounds, and interjections.
# Normalizes whitespace and merges speaker text.
# Saves preprocessed speaker-wise JSON in preprocessed_results/.
# Prints a preview (first 200 chars) per speaker.

import re
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

import spacy
from utils.logger import get_logger
from utils.json_reader import load_json

logger = get_logger(__name__)

# Load spaCy English model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found. Run:\n"
        "python -m spacy download en_core_web_sm"
    )


class TextPreprocessor:
    """
    Preprocess Whisper transcripts:
    - Remove common and dynamic fillers
    - Remove elongated sounds
    - Normalize whitespace
    """

    def __init__(self, results_dir: str = "preprocessed_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning:
        - Lowercase
        - Remove elongated sounds (uhhh, ummm, hmm)
        - Normalize spaces
        """
        text = text.lower()
        # Remove elongated sounds like uhh, ummm, ahh, ohhh
        text = re.sub(r'\b(u+h+|m+h+|h+m+|a+h+|o+h+)\b', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_fillers(self, text: str, dynamic_fillers: List[str] = None) -> str:
        """
        Remove interjections and dynamic fillers using spaCy + frequency analysis
        dynamic_fillers: list of additional filler words to remove
        """
        doc = nlp(text)
        tokens = []
        word_list = [t.text for t in doc if not t.is_punct]

        # Frequency-based dynamic filler detection
        if dynamic_fillers is None:
            counter = Counter(word_list)
            total_words = len(word_list)
            dynamic_fillers = {word for word, count in counter.items()
                               if len(word) <= 3 and count / total_words > 0.1}  # heuristic

        for token in doc:
            # Remove POS-based interjections, punctuation, and dynamic fillers
            if token.pos_ == "INTJ":
                continue
            if token.text in dynamic_fillers:
                continue
            if token.is_punct or token.is_space:
                continue
            tokens.append(token.text)

        return " ".join(tokens)

    def preprocess_transcript(self, transcript_json: str, request_id: str = "test") -> Dict[str, Any]:
        """
        Load Whisper transcript JSON and preprocess text speaker-wise
        Returns dict: {speaker: cleaned_text}
        """
        transcript = load_json(transcript_json)
        speaker_texts = {}
        for entry in transcript:
            speaker = entry.get("speaker", "unknown")
            raw_text = entry.get("text", "")
            cleaned = self.clean_text(raw_text)
            cleaned = self.remove_fillers(cleaned)
            speaker_texts.setdefault(speaker, []).append(cleaned)

        # Merge per speaker
        for spk in speaker_texts:
            speaker_texts[spk] = " ".join(speaker_texts[spk])

        # Save preprocessed JSON
        save_path = self.results_dir / f"preprocessed_transcript_{request_id}.json"
        import json
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(speaker_texts, f, indent=2, ensure_ascii=False)

        logger.info(f"Preprocessed transcript saved: {save_path}")
        return {"speaker_texts": speaker_texts, "saved_json_path": str(save_path)}