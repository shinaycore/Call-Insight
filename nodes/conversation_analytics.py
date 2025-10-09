# nodes/conversation_analytics.py
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import get_logger
from utils.json_reader import load_json

logger = get_logger(__name__)

class ConversationAnalyticsNode:
    def __init__(self, results_dir: str = "analytics_results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def run_node(self, speaker_keywords: dict, transcript_json: str = None, request_id: str = "test"):
        """
        Analyze conversation:
        - Count keyword occurrences per speaker
        - Determine dominant speaker per topic
        - Identify shared, unique, and neglected topics
        - Generate heatmap
        """
        # Prepare full text per speaker
        speaker_texts = {spk: "" for spk in speaker_keywords}

        if transcript_json:
            transcript = load_json(transcript_json)
            for entry in transcript:
                speaker = entry.get("speaker", "unknown")
                speaker_texts.setdefault(speaker, []).append(entry.get("text", ""))
            # Merge lists into single strings per speaker
            speaker_texts = {spk: " ".join(texts).lower() for spk, texts in speaker_texts.items()}
        else:
            # Fallback: merge keywords as text to count once each
            speaker_texts = {spk: " ".join(speaker_keywords[spk]).lower() for spk in speaker_keywords}

        # Count keyword occurrences per speaker
        topic_counts = {}
        all_keywords = set()
        for speaker, keywords in speaker_keywords.items():
            merged_text = speaker_texts.get(speaker, "")
            counts = {kw: merged_text.count(kw.lower()) for kw in keywords}
            topic_counts[speaker] = counts
            all_keywords.update(keywords)

        # Determine dominant speaker per topic
        dominant_speaker_per_topic = {}
        for kw in all_keywords:
            max_count = 0
            dominant = []
            for speaker, counts in topic_counts.items():
                if counts.get(kw, 0) > max_count:
                    max_count = counts[kw]
                    dominant = [speaker]
                elif counts.get(kw, 0) == max_count and max_count > 0:
                    dominant.append(speaker)
            dominant_speaker_per_topic[kw] = dominant

        # Shared, unique, neglected topics
        shared_topics = [kw for kw, speakers in dominant_speaker_per_topic.items() if len(speakers) > 1]
        unique_topics = {speaker: [] for speaker in speaker_keywords}
        neglected_topics = []
        for kw, speakers in dominant_speaker_per_topic.items():
            if len(speakers) == 1:
                unique_topics[speakers[0]].append(kw)
            if all(topic_counts[spk].get(kw, 0) == 0 for spk in speaker_keywords):
                neglected_topics.append(kw)

        # Generate heatmap
        sorted_keywords = sorted(all_keywords)
        heatmap_data = [[topic_counts[speaker].get(kw, 0) for kw in sorted_keywords] for speaker in speaker_keywords]

        plt.figure(figsize=(max(6, len(sorted_keywords) * 0.5), len(speaker_keywords)))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt="d",
            yticklabels=list(speaker_keywords.keys()),
            xticklabels=sorted_keywords,
            cmap="coolwarm"
        )
        heatmap_path = os.path.join(self.results_dir, f"conversation_heatmap_{request_id}.png")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()

        # Save results to JSON
        result = {
            "topic_counts": topic_counts,
            "dominant_speaker_per_topic": dominant_speaker_per_topic,
            "shared_topics": shared_topics,
            "unique_topics": unique_topics,
            "neglected_topics": neglected_topics,
            "heatmap_path": heatmap_path
        }
        json_path = os.path.join(self.results_dir, f"conversation_analytics_{request_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Conversation analytics saved: {json_path}")

        return result
