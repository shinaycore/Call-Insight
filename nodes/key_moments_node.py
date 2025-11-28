"""
Key Moments Node - Identifies critical timestamps in conversations
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


class KeyMomentsNode:
    def __init__(self, results_dir="key_moments_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        if seconds is None or seconds == 0:
            return "00:00"
        try:
            seconds = float(seconds)
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        except (ValueError, TypeError):
            return "00:00"

    def get_segment_timestamp(self, segment: Dict) -> float:
        """
        Extract timestamp from segment in various formats

        Tries multiple field names commonly used:
        - start, start_time, timestamp, time
        - Extracts from text if formatted like '[00:15 - 00:20]'
        """
        # Try direct timestamp fields
        timestamp = (
            segment.get("start")
            or segment.get("start_time")
            or segment.get("timestamp")
            or segment.get("time")
            or 0
        )

        if timestamp and timestamp > 0:
            return float(timestamp)

        # Try extracting from formatted text
        text = segment.get("text", "")
        extracted = self.extract_timestamp_from_text(text)

        # Convert MM:SS to seconds
        if extracted != "00:00":
            try:
                parts = extracted.split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except:
                pass

        return 0.0

    def extract_timestamp_from_text(self, text: str) -> str:
        """Extract timestamp from formatted text like '[00:15 - 00:20] Speaker: text'"""
        import re

        match = re.match(r"\[(\d{2}:\d{2})\s*-\s*(\d{2}:\d{2})\]", text)
        if match:
            return match.group(1)  # Return start time
        return "00:00"

    def detect_sentiment_shifts(
        self, sentiment_data: Dict, segments: List[Dict], threshold: float = 0.3
    ) -> List[Dict]:
        """
        Detect significant sentiment changes between consecutive segments

        Args:
            sentiment_data: Sentiment analysis results
            segments: Original speaker segments with timestamps
            threshold: Minimum change to consider significant (0-1 scale)
        """
        shifts = []
        results = sentiment_data.get("results", [])

        sentiment_map = {"POSITIVE": 1.0, "NEUTRAL": 0.0, "NEGATIVE": -1.0}

        for i in range(len(results) - 1):
            current = results[i]
            next_segment = results[i + 1]

            current_analysis = current.get("analysis", [])
            next_analysis = next_segment.get("analysis", [])

            if not current_analysis or not next_analysis:
                continue

            # Get dominant sentiment for each segment
            current_sent = current_analysis[0].get("sentiment", "NEUTRAL")
            next_sent = next_analysis[0].get("sentiment", "NEUTRAL")

            current_val = sentiment_map.get(current_sent, 0.0)
            next_val = sentiment_map.get(next_sent, 0.0)

            change = abs(next_val - current_val)

            if change >= threshold:
                # Get timestamp from original segment if available
                segment_idx = min(i + 1, len(segments) - 1)
                timestamp = self.get_segment_timestamp(segments[segment_idx])

                text = next_segment.get("text", "")

                shifts.append(
                    {
                        "type": "sentiment_shift",
                        "timestamp": timestamp,
                        "formatted_time": self.format_timestamp(timestamp),
                        "from_sentiment": current_sent,
                        "to_sentiment": next_sent,
                        "magnitude": change,
                        "speaker": next_segment.get("speaker", "Unknown"),
                        "description": f"Sentiment shifted from {current_sent} to {next_sent}",
                        "text": text,
                        "importance": "high" if change >= 0.6 else "medium",
                    }
                )

        return shifts

    def detect_decisions(self, segments: List[Dict]) -> List[Dict]:
        """
        Detect decision-making language in segments

        Keywords: decided, agreed, will do, let's go with, finalized, etc.
        """
        decision_keywords = [
            "decided",
            "agree",
            "agreed",
            "decision",
            "let's go with",
            "we'll do",
            "finalized",
            "confirmed",
            "approved",
            "reject",
            "rejected",
            "moving forward with",
            "plan is to",
            "commitment",
        ]

        decisions = []

        for segment in segments:
            text = segment.get("text", "")
            text_lower = text.lower()
            speaker = segment.get("speaker", "Unknown")

            # Check for decision keywords
            found_keywords = [kw for kw in decision_keywords if kw in text_lower]

            if found_keywords:
                timestamp = self.get_segment_timestamp(segment)

                decisions.append(
                    {
                        "type": "decision",
                        "timestamp": timestamp,
                        "formatted_time": self.format_timestamp(timestamp),
                        "speaker": speaker,
                        "description": text.strip(),
                        "text": text,
                        "keywords": found_keywords,
                        "importance": "high",
                    }
                )

        return decisions

    def detect_disagreements(self, segments: List[Dict]) -> List[Dict]:
        """
        Detect disagreement or conflict language

        Keywords: disagree, but, however, actually, concern, issue, problem, etc.
        """
        disagreement_keywords = [
            "disagree",
            "don't think",
            "not sure",
            "concern",
            "concerned",
            "issue",
            "problem",
            "worried",
            "however",
            "but actually",
            "on the other hand",
            "alternative",
            "pushback",
        ]

        disagreements = []

        for segment in segments:
            text = segment.get("text", "")
            text_lower = text.lower()
            speaker = segment.get("speaker", "Unknown")

            found_keywords = [kw for kw in disagreement_keywords if kw in text_lower]

            if found_keywords:
                timestamp = self.get_segment_timestamp(segment)

                disagreements.append(
                    {
                        "type": "disagreement",
                        "timestamp": timestamp,
                        "formatted_time": self.format_timestamp(timestamp),
                        "speaker": speaker,
                        "description": text.strip(),
                        "text": text,
                        "keywords": found_keywords,
                        "importance": "medium",
                    }
                )

        return disagreements

    def detect_agreements(self, segments: List[Dict]) -> List[Dict]:
        """
        Detect agreement or consensus language

        Keywords: exactly, absolutely, makes sense, good point, I agree, etc.
        """
        agreement_keywords = [
            "exactly",
            "absolutely",
            "agree",
            "makes sense",
            "good point",
            "right",
            "correct",
            "yes",
            "perfect",
            "sounds good",
            "that works",
            "I'm on board",
        ]

        agreements = []

        for segment in segments:
            text = segment.get("text", "")
            text_lower = text.lower()
            speaker = segment.get("speaker", "Unknown")

            found_keywords = [kw for kw in agreement_keywords if kw in text_lower]

            # Only count if multiple agreement keywords or strong agreement
            if len(found_keywords) >= 2 or any(
                kw in ["exactly", "absolutely", "perfect"] for kw in found_keywords
            ):
                timestamp = self.get_segment_timestamp(segment)

                agreements.append(
                    {
                        "type": "agreement",
                        "timestamp": timestamp,
                        "formatted_time": self.format_timestamp(timestamp),
                        "speaker": speaker,
                        "description": text.strip(),
                        "text": text,
                        "keywords": found_keywords,
                        "importance": "low",
                    }
                )

        return agreements

    def detect_action_items(self, segments: List[Dict]) -> List[Dict]:
        """
        Detect action items and commitments

        Keywords: will, should, need to, todo, task, deadline, by [date], etc.
        """
        action_keywords = [
            "will do",
            "i'll",
            "we'll",
            "need to",
            "should",
            "must",
            "have to",
            "responsible for",
            "assigned to",
            "by next week",
            "by tomorrow",
            "deadline",
            "due date",
        ]

        actions = []

        for segment in segments:
            text = segment.get("text", "")
            text_lower = text.lower()
            speaker = segment.get("speaker", "Unknown")

            found_keywords = [kw for kw in action_keywords if kw in text_lower]

            if found_keywords:
                timestamp = self.get_segment_timestamp(segment)

                actions.append(
                    {
                        "type": "action_item",
                        "timestamp": timestamp,
                        "formatted_time": self.format_timestamp(timestamp),
                        "speaker": speaker,
                        "description": text.strip(),
                        "text": text,
                        "keywords": found_keywords,
                        "importance": "high",
                    }
                )

        return actions

    def detect_questions(self, segments: List[Dict]) -> List[Dict]:
        """
        Detect important questions asked
        """
        questions = []

        for segment in segments:
            text = segment.get("text", "")
            speaker = segment.get("speaker", "Unknown")

            # Look for question marks or question words
            if "?" in text or any(
                text.lower().strip().startswith(q)
                for q in ["what", "why", "how", "when", "where", "who"]
            ):
                timestamp = self.get_segment_timestamp(segment)

                questions.append(
                    {
                        "type": "question",
                        "timestamp": timestamp,
                        "formatted_time": self.format_timestamp(timestamp),
                        "speaker": speaker,
                        "description": text.strip(),
                        "text": text,
                        "importance": "low",
                    }
                )

        return questions

    def rank_moments_by_importance(self, moments: List[Dict]) -> List[Dict]:
        """
        Rank moments by importance score

        Scoring:
        - High importance: decisions, sentiment shifts, action items
        - Medium importance: disagreements
        - Low importance: agreements, questions
        """
        importance_scores = {"high": 3, "medium": 2, "low": 1}

        for moment in moments:
            moment["score"] = importance_scores.get(moment.get("importance", "low"), 1)

        # Sort by score (descending) and timestamp
        moments.sort(
            key=lambda x: (
                -x.get("score", 0),
                x.get("timestamp", 0),
            )
        )

        return moments

    def extract_key_moments(
        self,
        diarization_data: Dict,
        sentiment_data: Dict,
        max_moments: int = 20,
    ) -> Dict[str, Any]:
        """
        Main function to extract all key moments

        Args:
            diarization_data: Raw diarization results with speaker segments
            sentiment_data: Sentiment analysis results
            max_moments: Maximum number of moments to return
        """
        logger.info("Extracting key moments from conversation...")

        # Get segments from diarization
        segments = diarization_data.get("speaker_segments", [])
        if not segments:
            logger.warning("No speaker segments found in diarization data")
            return {"moments": [], "summary": {}}

        # Detect different types of moments
        sentiment_shifts = self.detect_sentiment_shifts(sentiment_data, segments)
        decisions = self.detect_decisions(segments)
        disagreements = self.detect_disagreements(segments)
        agreements = self.detect_agreements(segments)
        action_items = self.detect_action_items(segments)
        questions = self.detect_questions(segments)

        # Combine all moments
        all_moments = (
            sentiment_shifts
            + decisions
            + disagreements
            + agreements
            + action_items
            + questions
        )

        # Rank and filter
        ranked_moments = self.rank_moments_by_importance(all_moments)
        top_moments = ranked_moments[:max_moments]

        # Create summary statistics
        summary = {
            "total_moments": len(all_moments),
            "sentiment_shifts": len(sentiment_shifts),
            "decisions": len(decisions),
            "disagreements": len(disagreements),
            "agreements": len(agreements),
            "action_items": len(action_items),
            "questions": len(questions),
            "top_moments_count": len(top_moments),
        }

        logger.info(f"Found {len(all_moments)} total moments")
        logger.info(f"Returning top {len(top_moments)} moments")

        return {
            "moments": top_moments,
            "all_moments": all_moments,
            "summary": summary,
            "by_type": {
                "sentiment_shifts": sentiment_shifts,
                "decisions": decisions,
                "disagreements": disagreements,
                "agreements": agreements,
                "action_items": action_items,
                "questions": questions,
            },
        }

    def save_results(self, results: Dict, request_id: str) -> str:
        """Save key moments results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"key_moments_{request_id}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Key moments saved to: {filepath}")
        return filepath

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process key moments extraction

        Expected state keys:
        - diarization_results: Raw diarization data
        - sentiment: Sentiment analysis results
        - request_id: Unique identifier
        """
        diarization_data = state.get("diarization_results", {})
        sentiment_data = state.get("sentiment", {})
        request_id = state.get("request_id", "unknown")

        # Extract key moments
        results = self.extract_key_moments(diarization_data, sentiment_data)

        # Add metadata
        results["request_id"] = request_id
        results["generated_at"] = datetime.now().isoformat()

        # Save results
        saved_path = self.save_results(results, request_id)

        return {
            "key_moments": results,
            "key_moments_path": saved_path,
        }
