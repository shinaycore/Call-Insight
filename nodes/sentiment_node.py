import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from transformers import pipeline
from utils.json_reader import load_json
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentNode:
    def __init__(self, config_path: str = "config/sentiment_config.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config = load_json(config_path)

        self.task_type = self.config.get("task_type", "sentiment")
        self.model_path = self.config.get(
            "model_path",
            self.config.get("model", "distilbert-base-uncased-finetuned-sst-2-english"),
        )

        logger.info(f"Loading model '{self.model_path}' for task '{self.task_type}'...")

        self.analyzer = pipeline(
            "sentiment-analysis"
            if self.task_type == "sentiment"
            else "text-classification",
            model=self.model_path,
            tokenizer=self.model_path,
        )

        logger.info("Model loaded successfully.")

        self.results_dir = self.config.get("results_dir", "sentiment_results")
        os.makedirs(self.results_dir, exist_ok=True)

    def _normalize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize HF model output to a consistent structure."""
        label = result.get("label", "").upper()

        if self.task_type == "emotion":
            if label in ["JOY"]:
                sentiment = "POSITIVE"
            elif label in ["SURPRISE", "NEUTRAL"]:
                sentiment = "NEUTRAL"
            else:
                sentiment = "NEGATIVE"
        else:
            sentiment = label

        return {
            "label": label,
            "sentiment": sentiment,
            "score": result.get("score", 0.0),
        }

    def analyze_speakers_batched(
        self, transcript: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Batch sentiment analysis while skipping empty text chunks safely."""

        valid_entries = []
        texts = []

        # Collect only non-empty texts
        for entry in transcript:
            txt = entry.get("text", "")
            if txt and txt.strip():
                valid_entries.append(entry)
                texts.append(txt)
            else:
                entry["analysis"] = []

        # Run model only on valid entries
        if texts:
            raw_results = self.analyzer(texts)
        else:
            raw_results = []

        # Map results back
        analyzed = []
        idx = 0

        for entry in transcript:
            txt = entry.get("text", "")

            if txt and txt.strip():
                result = raw_results[idx]
                idx += 1

                if isinstance(result, list):
                    normalized = [self._normalize(r) for r in result]
                else:
                    normalized = [self._normalize(result)]

                analyzed.append(
                    {
                        "speaker": entry.get("speaker", "unknown"),
                        "text": txt,
                        "analysis": normalized,
                    }
                )
            else:
                analyzed.append(
                    {
                        "speaker": entry.get("speaker", "unknown"),
                        "text": txt,
                        "analysis": [],
                    }
                )

        return analyzed

    def save_sentiment_results(
        self, results: List[Dict[str, Any]], request_id: Optional[str] = None
    ) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_name = (
            f"sentiment_{request_id}_{timestamp}"
            if request_id
            else f"sentiment_{timestamp}"
        )
        json_path = os.path.join(self.results_dir, f"{base_name}.json")

        out = {
            "model_used": self.model_path,
            "task_type": self.task_type,
            "results": results,
        }

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(out, jf, indent=2, ensure_ascii=False)

        logger.info(f"Sentiment results saved: {json_path}")
        return {"json_path": json_path}

    def aggregate_per_speaker(
        self, analyzed_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        stats = defaultdict(
            lambda: {
                "POSITIVE": 0,
                "NEGATIVE": 0,
                "NEUTRAL": 0,
                "score_sum": 0.0,
                "count": 0,
            }
        )

        for item in analyzed_results:
            speaker = item.get("speaker", "unknown")

            for a in item.get("analysis", []):
                sent = a.get("sentiment", "NEUTRAL")
                score = a.get("score", 0.0)

                stats[speaker][sent] += 1
                stats[speaker]["score_sum"] += score
                stats[speaker]["count"] += 1

        final = {}
        for speaker, s in stats.items():
            avg = s["score_sum"] / s["count"] if s["count"] else 0.0

            final[speaker] = {
                "counts": {
                    "POSITIVE": s["POSITIVE"],
                    "NEGATIVE": s["NEGATIVE"],
                    "NEUTRAL": s["NEUTRAL"],
                },
                "average_score": avg,
                "total_chunks": s["count"],
            }

        return final

    def sentiment_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            transcript = state.get("diarized_transcript", [])
            if not transcript:
                return {
                    "error": "No diarized transcript provided",
                    "request_id": state.get("request_id"),
                }

            analyzed = self.analyze_speakers_batched(transcript)
            saved = self.save_sentiment_results(
                analyzed, request_id=state.get("request_id")
            )
            aggregated = self.aggregate_per_speaker(analyzed)

            return {
                "request_id": state.get("request_id"),
                "results": analyzed,
                "aggregated": aggregated,
                "saved_path": saved.get("json_path"),
            }

        except Exception as e:
            logger.error(f"Sentiment node failed: {e}")
            return {"error": str(e), "request_id": state.get("request_id")}
