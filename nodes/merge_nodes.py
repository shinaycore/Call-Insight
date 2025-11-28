import json
import os
from datetime import datetime

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML


class MergeNode:
    def __init__(
        self,
        results_dir="merged_results",
        template_dir="templates",
        reports_dir="reports",
    ):
        self.results_dir = results_dir
        self.template_dir = template_dir
        self.reports_dir = reports_dir

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        self.env = Environment(loader=FileSystemLoader(self.template_dir))
        # Add custom filter for time formatting
        self.env.filters["format_time"] = self.format_timestamp

    def load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------------------------------------
    # Format timestamp to MM:SS
    # ---------------------------------------
    def format_timestamp(self, seconds):
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

    # ---------------------------------------
    # Process key moments to ensure proper timestamp formatting
    # ---------------------------------------
    def process_key_moments(self, key_moments):
        """Process key moments to ensure timestamps are properly formatted"""
        if not key_moments:
            return []

        # Handle different possible structures
        moments = key_moments
        if isinstance(key_moments, dict):
            # Try different possible keys
            moments = (
                key_moments.get("moments")
                or key_moments.get("results")
                or key_moments.get("key_moments")
                or []
            )

        if not isinstance(moments, list):
            return []

        processed_moments = []
        for moment in moments:
            if isinstance(moment, dict):
                # Extract timestamp (could be 'timestamp', 'time', 'start_time', etc.)
                timestamp = (
                    moment.get("timestamp")
                    or moment.get("time")
                    or moment.get("start_time")
                    or moment.get("start")
                    or 0
                )

                # Extract description
                description = (
                    moment.get("description")
                    or moment.get("text")
                    or moment.get("content")
                    or moment.get("summary")
                    or ""
                )

                # Extract speaker
                speaker = moment.get("speaker") or moment.get("speaker_id") or ""

                processed_moment = {
                    "timestamp": timestamp,
                    "formatted_time": self.format_timestamp(timestamp),
                    "description": description,
                    "importance": moment.get("importance", "medium"),
                    "speaker": self.normalize_speaker_key(speaker),
                }
                processed_moments.append(processed_moment)

        # Sort by timestamp
        processed_moments.sort(key=lambda x: x["timestamp"])
        return processed_moments

    # ---------------------------------------
    # Normalize speaker key (lowercase, consistent format)
    # ---------------------------------------
    def normalize_speaker_key(self, speaker: str) -> str:
        """Normalize speaker keys to lowercase format like 'speaker 0'"""
        if not speaker:
            return speaker
        return speaker.lower().strip()

    # ---------------------------------------
    # Build aggregated sentiment if missing
    # ---------------------------------------
    def rebuild_aggregated(self, results):
        agg = {}

        for entry in results:
            spk = entry.get("speaker")
            chunks = entry.get("analysis", [])

            if not spk:
                continue

            # Normalize the speaker key
            spk = self.normalize_speaker_key(spk)

            pos = sum(1 for c in chunks if c.get("sentiment") == "POSITIVE")
            neg = sum(1 for c in chunks if c.get("sentiment") == "NEGATIVE")
            neu = sum(1 for c in chunks if c.get("sentiment") == "NEUTRAL")

            total = len(chunks)
            avg_score = sum(c.get("score", 0) for c in chunks) / total if total else 0

            counts = {"POSITIVE": pos, "NEGATIVE": neg, "NEUTRAL": neu}
            overall = max(counts, key=counts.get) if counts else "NEUTRAL"

            agg[spk] = {
                "POSITIVE": pos,
                "NEGATIVE": neg,
                "NEUTRAL": neu,
                "total_chunks": total,
                "average_score": avg_score,
                "overall_sentiment": overall,
            }

        return agg

    # ---------------------------------------
    # Clean HTML tags from text
    # ---------------------------------------
    def clean_html_tags(self, text):
        """Remove HTML tags like <s>, <del>, etc. from text"""
        if not isinstance(text, str):
            return text

        import re

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        return text

    # ---------------------------------------
    # Normalize all speaker keys in data structures
    # ---------------------------------------
    def normalize_data_keys(self, data):
        """Recursively normalize all speaker keys in nested structures and clean HTML tags"""
        if isinstance(data, dict):
            normalized = {}
            for key, value in data.items():
                # Check if key looks like a speaker identifier
                if isinstance(key, str) and "speaker" in key.lower():
                    key = self.normalize_speaker_key(key)
                # Recursively normalize nested values
                normalized[key] = self.normalize_data_keys(value)
            return normalized
        elif isinstance(data, list):
            return [self.normalize_data_keys(item) for item in data]
        elif isinstance(data, str):
            # Clean HTML tags from string values
            return self.clean_html_tags(data)
        else:
            return data

    # ---------------------------------------
    # Render HTML using Jinja2
    # ---------------------------------------
    def render_html(self, merged_data: dict, out_name: str):
        template = self.env.get_template("main.html")

        html_str = template.render(
            request_id=merged_data.get("request_id"),
            generated_at=merged_data.get("generated_at"),
            summary=merged_data.get("summary", {}),
            sentiment=merged_data.get("sentiment", {}),
            topics=merged_data.get("topics", {}),
            key_moments=merged_data.get("key_moments", []),
        )

        html_path = os.path.join(self.reports_dir, out_name + ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_str)

        return html_path

    # ---------------------------------------
    # PDF Generator
    # ---------------------------------------
    def generate_pdf(self, html_path: str, out_name: str):
        pdf_path = os.path.join(self.reports_dir, out_name + ".pdf")
        HTML(html_path).write_pdf(pdf_path)
        return pdf_path

    # ---------------------------------------
    # Main merge function
    # ---------------------------------------
    def merge(
        self,
        summary_path: str,
        sentiment_path: str,
        topic_path: str,
        key_moments_path: str = None,
        request_id: str = None,
    ):
        # Load JSON files
        summary = self.load_json(summary_path) if summary_path else {}
        raw_sentiment = self.load_json(sentiment_path) if sentiment_path else {}
        topics = self.load_json(topic_path) if topic_path else {}
        raw_key_moments = self.load_json(key_moments_path) if key_moments_path else {}

        # Normalize all speaker keys in the data
        summary = self.normalize_data_keys(summary)
        raw_sentiment = self.normalize_data_keys(raw_sentiment)
        topics = self.normalize_data_keys(topics)
        raw_key_moments = self.normalize_data_keys(raw_key_moments)

        # Process key moments with proper timestamp formatting
        key_moments = self.process_key_moments(raw_key_moments)

        # Extract sentiment results
        sentiment_results = raw_sentiment.get("results", [])

        # Build aggregated sentiment (if needed)
        sentiment_aggregated = raw_sentiment.get(
            "aggregated"
        ) or self.rebuild_aggregated(sentiment_results)

        # Ensure aggregated sentiment keys are normalized
        sentiment_aggregated = self.normalize_data_keys(sentiment_aggregated)

        sentiment = {
            "results": sentiment_results,
            "aggregated": sentiment_aggregated,
        }

        merged = {
            "request_id": request_id,
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "sentiment": sentiment,
            "topics": topics,
            "key_moments": key_moments,
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"merged_{request_id}_{ts}" if request_id else f"merged_{ts}"

        # Save merged JSON
        merged_json_path = os.path.join(self.results_dir, base_name + ".json")
        with open(merged_json_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        # Generate HTML
        html_path = self.render_html(merged, base_name)

        # Generate PDF
        pdf_path = self.generate_pdf(html_path, base_name)

        return {
            "merged_path": merged_json_path,
            "html_path": html_path,
            "pdf_path": pdf_path,
            "summary_path": summary_path,
            "sentiment_path": sentiment_path,
            "topic_path": topic_path,
            "key_moments_path": key_moments_path,
        }
