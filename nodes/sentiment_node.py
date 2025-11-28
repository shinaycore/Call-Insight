import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from transformers import pipeline
from utils.json_reader import load_json
from utils.logger import get_logger

logger = get_logger(__name__)


MAX_CHARS = 400  # safe for RoBERTa-based emotion models


def chunk_text(text: str, chunk_size: int = MAX_CHARS):
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    return [text[i : i + chunk_size].strip() for i in range(0, len(text), chunk_size)]


class SentimentNode:
    def __init__(self, config_path: str = "config/sentiment_config.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config = load_json(config_path)

        self.task_type = self.config.get("task_type", "emotion")
        self.model_path = self.config.get(
            "model_path",
            "j-hartmann/emotion-english-distilroberta-base",
        )

        logger.info(f"Loading model '{self.model_path}' for task '{self.task_type}'...")

        self.analyzer = pipeline(
            "text-classification",
            model=self.model_path,
            tokenizer=self.model_path,
            top_k=None,  # full emotion distribution
            truncation=True,
        )

        logger.info("Model loaded successfully.")

        self.results_dir = self.config.get("results_dir", "sentiment_results")
        os.makedirs(self.results_dir, exist_ok=True)

    # --------------------------------------------------------------
    # SAFE NORMALIZER
    # --------------------------------------------------------------
    def _normalize(self, raw):
        # raw is a list of dicts: [{label, score}, {...}]
        if not isinstance(raw, list) or len(raw) == 0:
            return {"label": "NEUTRAL", "sentiment": "NEUTRAL", "score": 0.0}

        best = max(raw, key=lambda x: x.get("score", 0.0))

        label = best.get("label", "neutral").upper()
        score = float(best.get("score", 0.0))

        pos = {"JOY", "LOVE", "HAPPINESS", "POSITIVE"}
        neg = {"ANGER", "FEAR", "DISGUST", "SADNESS", "NEGATIVE"}

        if label in pos:
            sentiment = "POSITIVE"
        elif label in neg:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

        return {"label": label, "sentiment": sentiment, "score": score}

    # --------------------------------------------------------------
    # ANALYSIS WITH CHUNKING + UNWRAPPING
    # --------------------------------------------------------------
    def _unwrap(self, out):
        """
        HF pipelines sometimes return:
        [[[{...}]]], [[[{...}]]], [{...}], etc.
        This safely unwraps nested lists until only the final list remains.
        """
        while isinstance(out, list) and len(out) == 1:
            out = out[0]
        return out

    def analyze_speakers_chunked(self, transcript: List[Dict[str, Any]]) -> List[Dict]:
        analyzed = []

        for entry in transcript:
            speaker = entry.get("speaker", "unknown")
            text = entry.get("text", "").strip()

            if not text:
                analyzed.append({"speaker": speaker, "text": "", "analysis": []})
                continue

            chunks = chunk_text(text)
            results = []

            for ch in chunks:
                try:
                    raw = self.analyzer(ch)
                    raw = self._unwrap(raw)
                    normalized = self._normalize(raw)
                    results.append(normalized)

                except Exception as e:
                    logger.error(f"Chunk sentiment failed: {e}")
                    results.append(
                        {"label": "NEUTRAL", "sentiment": "NEUTRAL", "score": 0.0}
                    )

            analyzed.append({"speaker": speaker, "text": text, "analysis": results})

        return analyzed

    # --------------------------------------------------------------
    # AGGREGATION
    # --------------------------------------------------------------
    def aggregate_per_speaker(self, analyzed: List[Dict]) -> Dict:
        final = {}

        for entry in analyzed:
            spk = entry["speaker"]
            entries = entry["analysis"]

            if not entries:
                final[spk] = {
                    "POSITIVE": 0,
                    "NEGATIVE": 0,
                    "NEUTRAL": 0,
                    "average_score": 0.0,
                    "total_chunks": 0,
                    "overall_sentiment": "NEUTRAL",
                }
                continue

            pos = sum(1 for x in entries if x["sentiment"] == "POSITIVE")
            neg = sum(1 for x in entries if x["sentiment"] == "NEGATIVE")
            neu = sum(1 for x in entries if x["sentiment"] == "NEUTRAL")

            avg_score = sum(x["score"] for x in entries) / len(entries)

            counts = {"POSITIVE": pos, "NEGATIVE": neg, "NEUTRAL": neu}
            top = max(counts, key=counts.get)

            final[spk] = {
                "POSITIVE": pos,
                "NEGATIVE": neg,
                "NEUTRAL": neu,
                "average_score": avg_score,
                "total_chunks": len(entries),
                "overall_sentiment": top,
            }

        return final

    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------
    def save_sentiment_results(self, results, request_id=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"sentiment_{request_id}_{ts}" if request_id else f"sentiment_{ts}"
        path = os.path.join(self.results_dir, f"{name}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"model": self.model_path, "task": self.task_type, "results": results},
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"Sentiment results saved: {path}")
        return path

    # --------------------------------------------------------------
    # LANGGRAPH NODE
    # --------------------------------------------------------------
    def sentiment_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            transcript = state.get("diarized_transcript", [])
            rid = state.get("request_id")

            if not transcript:
                return {"error": "No transcript", "saved_path": None}

            analyzed = self.analyze_speakers_chunked(transcript)
            aggregated = self.aggregate_per_speaker(analyzed)

            summary = {
                spk: {
                    "overall_sentiment": aggregated[spk]["overall_sentiment"],
                    "avg_score": aggregated[spk]["average_score"],
                    "chunks": aggregated[spk]["total_chunks"],
                }
                for spk in aggregated
            }

            saved_path = self.save_sentiment_results(analyzed, rid)

            return {
                "request_id": rid,
                "results": analyzed,
                "aggregated": aggregated,
                "summary": summary,
                "saved_path": saved_path,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Sentiment failed: {e}")
            return {"error": str(e), "saved_path": None}
