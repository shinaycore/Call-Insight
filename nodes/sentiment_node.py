import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from transformers import pipeline
from utils.logger import get_logger
from utils.json_reader import load_json
from collections import defaultdict

logger = get_logger(__name__)

class SentimentNode:
    def __init__(self, config_path: str = "config/sentiment_config.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        self.config = load_json(config_path)

        self.task_type = self.config.get("task_type", "sentiment")
        self.model_path = self.config.get(
            "model_path", 
            self.config.get("model", "distilbert-base-uncased-finetuned-sst-2-english")
        )

        logger.info(f"Loading model '{self.model_path}' for task '{self.task_type}'...")
        self.analyzer = pipeline(
            "sentiment-analysis" if self.task_type == "sentiment" else "text-classification",
            model=self.model_path,
            tokenizer=self.model_path
        )
        logger.info("Model loaded successfully.")

        self.results_dir = self.config.get("results_dir", "sentiment_results")
        os.makedirs(self.results_dir, exist_ok=True)

    def analyze_speaker(self, speaker_text: Dict[str, str]) -> Dict[str, Any]:
        text = speaker_text.get("text", "")
        speaker = speaker_text.get("speaker", "unknown")
        if not text.strip():
            return {"speaker": speaker, "text": text, "analysis": []}

        results = self.analyzer(text)
        
        if self.task_type == "emotion":
            for r in results:
                label = r.get("label", "NEUTRAL").upper()
                if label in ["JOY"]:
                    r["sentiment"] = "POSITIVE"
                elif label in ["NEUTRAL", "SURPRISE"]:
                    r["sentiment"] = "NEUTRAL"
                else:
                    r["sentiment"] = "NEGATIVE"
        elif self.task_type == "sentiment":
            for r in results:
                r["label"] = r.get("label", "").upper()

        return {"speaker": speaker, "text": text, "analysis": results}

    def save_sentiment_results(self, results: List[Dict[str, Any]], request_id: Optional[str] = None) -> Dict[str, str]:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"sentiment_{request_id}_{timestamp}" if request_id else f"sentiment_{timestamp}"
            json_path = os.path.join(self.results_dir, f"{base_name}.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(results, jf, indent=2, ensure_ascii=False)
            logger.info(f"Sentiment results saved: {json_path}")
            return {"json_path": json_path}
        except Exception as e:
            logger.error(f"Error saving sentiment results: {e}")
            raise

    def aggregate_per_speaker(self, analyzed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics per speaker: counts per sentiment and average score.
        """
        speaker_stats = defaultdict(lambda: {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0, "score_sum": 0.0, "count": 0})

        for entry in analyzed_results:
            speaker = entry.get("speaker", "unknown")
            for a in entry.get("analysis", []):
                sentiment_label = a.get("sentiment") or a.get("label", "NEUTRAL")
                score = a.get("score", 0.0)

                speaker_stats[speaker][sentiment_label] += 1
                speaker_stats[speaker]["score_sum"] += score
                speaker_stats[speaker]["count"] += 1

        aggregated = {}
        for speaker, stats in speaker_stats.items():
            avg_score = stats["score_sum"] / stats["count"] if stats["count"] else 0.0
            aggregated[speaker] = {
                "counts": {k: stats[k] for k in ["POSITIVE", "NEGATIVE", "NEUTRAL"]},
                "average_score": avg_score,
                "total_chunks": stats["count"]
            }

        return aggregated

    def sentiment_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            transcript = state.get("diarized_transcript", [])
            if not transcript:
                return {"error": "No diarized transcript provided", "request_id": state.get("request_id")}

            # Analyze each speaker chunk
            analyzed = [self.analyze_speaker(s) for s in transcript]

            # Save per-chunk results
            saved = self.save_sentiment_results(analyzed, request_id=state.get("request_id"))

            # Aggregate metrics per speaker
            aggregated = self.aggregate_per_speaker(analyzed)

            # Log aggregated metrics
            logger.info("\n--- Aggregated Metrics per Speaker ---")
            for speaker, stats in aggregated.items():
                counts = stats["counts"]
                avg_score = stats["average_score"]
                total_chunks = stats["total_chunks"]
                logger.info(f"\nSpeaker: {speaker}")
                logger.info(f"  Total chunks: {total_chunks}")
                logger.info(f"  Counts: {counts}")
                logger.info(f"  Average sentiment score: {avg_score:.3f}")

            return {
                "request_id": state.get("request_id"),
                "results": analyzed,
                "aggregated": aggregated,
                "saved_path": saved.get("json_path")
            }

        except Exception as e:
            logger.error(f"Sentiment node failed: {e}")
            return {"error": str(e), "request_id": state.get("request_id")}