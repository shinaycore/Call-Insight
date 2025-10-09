# keyword_extraction.py
import os
import json
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter
import re
from pathlib import Path

from utils.logger import get_logger
from utils.json_reader import load_json
from dotenv import load_dotenv

logger = get_logger(__name__)

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    raise ImportError(
        "Please install spacy and scikit-learn:\n"
        "pip install spacy scikit-learn && python -m spacy download en_core_web_sm"
    )

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --- spaCy stopwords + small custom noise set ---
CUSTOM_NOISE = {"thanks", "good", "well", "yeah", "hello", "episode", "www", "bye", "one"}
STOPWORDS = STOP_WORDS.union(CUSTOM_NOISE)


class KeywordExtractionNode:
    def __init__(self, results_dir: str = "keyword_results", top_n: int = 20):
        """
        Keyword extraction node using spaCy NLP + TF-IDF
        top_n: maximum number of keywords/action points to extract
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.top_n = top_n

    def clean_text(self, text: str) -> str:
        """Basic cleanup for transcript text"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_keywords_from_text(self, text: str, use_tfidf: bool = True) -> List[str]:
        """Extract keywords using spaCy + optional TF-IDF ranking"""
        text = self.clean_text(text)
        doc = nlp(text)

        candidate_keywords = []

        # --- Collect noun phrases and named entities ---
        for chunk in doc.noun_chunks:
            token_text = chunk.text.lower().strip()
            if token_text not in STOPWORDS and len(token_text) > 2:
                candidate_keywords.append(token_text)

        for ent in doc.ents:
            candidate_keywords.append(ent.text.strip().lower())

        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in STOPWORDS:
                candidate_keywords.append(token.text.lower())

        candidate_keywords = list(dict.fromkeys(candidate_keywords))  # remove duplicates

        # --- Optional TF-IDF scoring ---
        if use_tfidf and candidate_keywords:
            vectorizer = TfidfVectorizer(vocabulary=candidate_keywords, use_idf=True, smooth_idf=True)
            tfidf_matrix = vectorizer.fit_transform([text])
            scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
            # Pick top N
            sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            candidate_keywords = [kw for kw, score in sorted_keywords[:self.top_n]]

        return candidate_keywords

    def extract_keywords_from_transcript(self, transcript_json: str) -> Dict[str, List[str]]:
        """Extract keywords speaker-wise"""
        transcript = load_json(transcript_json)
        speaker_texts = {}
        for entry in transcript:
            speaker = entry.get("speaker", "unknown")
            speaker_texts.setdefault(speaker, []).append(entry.get("text", ""))

        speaker_keywords = {}
        for spk, texts in speaker_texts.items():
            merged_text = " ".join(texts)
            speaker_keywords[spk] = self.extract_keywords_from_text(merged_text)

        return speaker_keywords

    def compute_speaker_metrics(self, transcript_json: str, speaker_keywords: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Count keyword mentions per speaker and identify dominant topics
        """
        transcript = load_json(transcript_json)
        speaker_texts = {}
        for entry in transcript:
            speaker = entry.get("speaker", "unknown")
            speaker_texts.setdefault(speaker, []).append(entry.get("text", ""))

        speaker_metrics = {}
        for spk, texts in speaker_texts.items():
            merged_text = " ".join(texts).lower()
            counts = {}
            for kw in speaker_keywords.get(spk, []):
                counts[kw] = merged_text.count(kw.lower())
            sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
            speaker_metrics[spk] = {
                "keyword_counts": sorted_counts,
                "dominant_topics": [kw for kw, c in sorted_counts.items() if c > 0]
            }

        return speaker_metrics

    def extract_node(
        self,
        txt_file: str = None,
        transcript_json: str = None,
        request_id: str = "test"
    ) -> Dict[str, Any]:
        """Run keyword extraction node with speaker metrics"""
        try:
            results = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Global keywords from raw text
            if txt_file:
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read()
                global_keywords = self.extract_keywords_from_text(text)
                results["global_keywords"] = global_keywords

                txt_path = os.path.join(self.results_dir, f"keywords_{request_id}_{timestamp}.txt")
                with open(txt_path, "w", encoding="utf-8") as tf:
                    tf.write("Global Keywords / Action Points:\n")
                    for k in global_keywords:
                        tf.write(f"- {k}\n")
                logger.info(f"Global keywords saved: {txt_path}")
                results["saved_txt_path"] = txt_path

            # Speaker-wise keywords
            if transcript_json:
                speaker_keywords = self.extract_keywords_from_transcript(transcript_json)
                results["speaker_keywords"] = speaker_keywords

                # --- Compute speaker interaction metrics ---
                speaker_metrics = self.compute_speaker_metrics(transcript_json, speaker_keywords)
                results["speaker_metrics"] = speaker_metrics

            # Save JSON
            json_path = os.path.join(self.results_dir, f"keywords_{request_id}_{timestamp}.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(results, jf, indent=2, ensure_ascii=False)
            results["saved_json_path"] = json_path

            logger.info(f"Keyword extraction results saved: {json_path}")
            results["request_id"] = request_id
            return results

        except Exception as e:
            logger.error(f"Keyword extraction node failed: {e}")
            return {"error": str(e), "request_id": request_id}
