"""
LangGraph pipeline runner for Call-Insight (LangGraph 1.x compatible)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# allow importing nodes dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph
from nodes.merge_nodes import MergeNode
from nodes.sentiment_node import SentimentNode
from nodes.summariser_node import SummarizationNode
from nodes.text_preprocessing import TextPreprocessor
from nodes.topic_extractor import llm_topic_extractor_node
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------
# Helpers
# ---------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "sentiment_config.json"


def load_and_fix_diarized_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict) and "speaker_texts" in raw:
        return [{"speaker": k, "text": v} for k, v in raw["speaker_texts"].items()]

    raise ValueError(f"Unsupported transcript format: {type(raw)}")


# ---------------------------
# NODES — state mutations
# ---------------------------
def preprocess_node(state: Dict[str, Any]):
    logger.info("Running TextPreprocessor...")

    tp = TextPreprocessor(results_dir="preprocessed_results", redact_pii=True)
    out = tp.preprocess_transcript(state["transcript_path"], state["request_id"])

    state["preprocessed_txt"] = out["saved_txt_path"]
    state["preprocessed_json"] = out["saved_json_path"]

    logger.info(f"Preprocessing done: {out}")
    return state


def summarizer_node(state: Dict[str, Any]):
    logger.info("Running SummarizationNode...")

    summ = SummarizationNode(results_dir="summarization_results")
    out = summ.summarize_node(
        txt_file=state["preprocessed_txt"],
        transcript_json=None,
        request_id=state["request_id"],
    )

    state["summary"] = out
    logger.info("Summarization completed.")
    return state


def sentiment_node(state: Dict[str, Any]):
    logger.info("Running SentimentNode...")

    diarized = load_and_fix_diarized_json(state["preprocessed_json"])

    sent = SentimentNode(config_path=str(CONFIG_PATH))
    out = sent.sentiment_analysis_node(
        {"diarized_transcript": diarized, "request_id": state["request_id"]}
    )

    state["sentiment"] = out
    logger.info("Sentiment completed.")
    return state


def topics_node(state: Dict[str, Any]):
    logger.info("Running Topic Extraction...")

    summary_text = state["summary"].get("global_notes", "")

    out = llm_topic_extractor_node(
        {"global_summary": summary_text, "request_id": state["request_id"]}
    )

    state["topics"] = out
    logger.info(f"Topics saved: {out.get('topic_path')}")
    return state


def merge_node(state: Dict[str, Any]):
    logger.info("Running MergeNode...")

    merger = MergeNode(results_dir="merged_results")

    out = merger.merge(
        summary_path=state["summary"]["paths"]["json"],
        sentiment_path=state["sentiment"]["saved_path"],
        request_id=state["request_id"],
    )

    state["merged"] = out
    logger.info(f"Merged saved: {out.get('merged_path')}")
    return state


# ---------------------------
# BUILD GRAPH
# ---------------------------
def build_graph():
    graph = StateGraph(dict)

    graph.add_node("preprocess", preprocess_node)
    graph.add_node("summarize", summarizer_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("topics", topics_node)
    graph.add_node("merge", merge_node)

    graph.add_edge("preprocess", "summarize")
    graph.add_edge("summarize", "sentiment")
    graph.add_edge("sentiment", "topics")
    graph.add_edge("topics", "merge")

    graph.set_entry_point("preprocess")
    graph.set_finish_point("merge")

    return graph


# ---------------------------
# RUN
# ---------------------------
def run_pipeline(transcript_path: str, request_id: str):
    graph = build_graph()
    logger.info("Compiling LangGraph pipeline...")
    app = graph.compile()

    init = {"transcript_path": transcript_path, "request_id": request_id}

    logger.info("Invoking pipeline...")
    final_state = app.invoke(init)

    logger.info("Pipeline finished.")
    logger.info(f"Topics: {final_state.get('topics', {}).get('topics')}")
    logger.info(f"Merged file: {final_state.get('merged', {}).get('merged_path')}")

    return final_state


if __name__ == "__main__":
    run_pipeline("../sample_transcript.json", "demo_run")
