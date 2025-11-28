"""
LangGraph pipeline runner for Call-Insight (LangGraph 1.x compatible)
WITH KEY MOMENTS NODE
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# allow importing nodes dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph
from nodes.audio_preprocessing import AudioPreprocessingNode
from nodes.diarization_node import DiarizationNode
from nodes.key_moments_node import KeyMomentsNode  # NEW IMPORT
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
def audio_preprocessing_node(state: Dict[str, Any]):
    logger.info("Running AudioPreprocessingNode...")

    # Validate input file exists
    audio_file_path = state.get("audio_file_path")
    if not audio_file_path:
        error_msg = "No audio_file_path provided in state"
        logger.error(error_msg)
        raise Exception(error_msg)

    # Convert to absolute path if relative
    audio_file_path = os.path.abspath(audio_file_path)

    if not os.path.exists(audio_file_path):
        error_msg = (
            f"Audio file not found: {audio_file_path}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Please provide a valid absolute or relative path"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Found audio file: {audio_file_path}")
    logger.info(f"File size: {os.path.getsize(audio_file_path) / (1024 * 1024):.2f} MB")

    preprocessor = AudioPreprocessingNode(
        default_sr=16000, default_out_folder="preprocessed_audio"
    )

    # Prepare state for audio preprocessing
    audio_state = {
        "input_path": audio_file_path,
        "request_id": state["request_id"],
        "target_sr": state.get("target_sr", 16000),
        "do_noise_reduction": state.get("do_noise_reduction", False),
        "do_trim": state.get("do_trim", True),
        "out_folder": state.get("out_folder", "preprocessed_audio"),
        "messages": [],
    }

    result = preprocessor(audio_state)

    if result.get("error"):
        logger.error(f"Audio preprocessing failed: {result['error']}")
        raise Exception(f"Audio preprocessing error: {result['error']}")

    # Update state with preprocessed audio path
    state["preprocessed_audio_path"] = result["out_path"]
    state["audio_sr"] = result["sr"]
    state["audio_duration"] = result["duration"]
    state["audio_preprocessing_messages"] = result.get("messages", [])

    logger.info(
        f"Audio preprocessing completed: {result['duration']:.2f}s @ {result['sr']}Hz"
    )
    logger.info(f"Preprocessed audio saved to: {result['out_path']}")

    return state


def convert_diarization_to_preprocessing_format(
    speaker_segments: List[Dict],
) -> List[Dict]:
    """
    Convert diarization output format to text preprocessing input format

    Diarization format:
    [
        {
            "speaker": "Speaker 0",
            "text": "Hello world",
            "start": 0.0,
            "end": 2.5,
            "confidence": 0.95
        }
    ]

    Preprocessing format:
    [
        {
            "speaker": "Speaker 0",
            "text": "[00:00 - 00:02] Speaker 0: Hello world"
        }
    ]
    """
    converted = []

    for segment in speaker_segments:
        # Format timestamps
        start_min = int(segment["start"] // 60)
        start_sec = int(segment["start"] % 60)
        end_min = int(segment["end"] // 60)
        end_sec = int(segment["end"] % 60)

        # Create formatted text with timestamp
        formatted_text = (
            f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}] "
            f"{segment['speaker']}: {segment['text']}"
        )

        converted.append({"speaker": segment["speaker"], "text": formatted_text})

    return converted


def diarization_node(state: Dict[str, Any]):
    logger.info("Running DiarizationNode...")

    diarizer = DiarizationNode()

    # Use preprocessed audio path for diarization
    diarization_state = {
        "audio_file_path": state["preprocessed_audio_path"],
        "request_id": state["request_id"],
    }

    result = diarizer(diarization_state)

    if result.get("error"):
        logger.error(f"Diarization failed: {result['error']}")
        raise Exception(f"Diarization error: {result['error']}")

    # Convert speaker segments to preprocessing-compatible format
    converted_segments = convert_diarization_to_preprocessing_format(
        result["speaker_segments"]
    )

    # Save diarization results in BOTH formats
    diarized_data = {
        "full_transcript": result["full_transcript"],
        "speaker_segments": result["speaker_segments"],  # Original format
        "speakers_count": result["speakers_count"],
        "duration": result["duration"],
    }

    # Save preprocessing-compatible format
    preprocessing_data = converted_segments

    # Create diarization results directory
    diarization_dir = Path("diarization_results")
    diarization_dir.mkdir(exist_ok=True)

    # Save original diarization format
    raw_transcript_path = diarization_dir / f"{state['request_id']}_diarized_raw.json"
    with open(raw_transcript_path, "w", encoding="utf-8") as f:
        json.dump(diarized_data, f, indent=2, ensure_ascii=False)

    # Save preprocessing-compatible format
    transcript_path = diarization_dir / f"{state['request_id']}_diarized.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(preprocessing_data, f, indent=2, ensure_ascii=False)

    state["transcript_path"] = str(transcript_path)
    state["diarization_results"] = diarized_data
    state["diarization_converted"] = preprocessing_data

    logger.info(
        f"Diarization completed: {result['speakers_count']} speakers, "
        f"{result['duration']:.2f}s duration"
    )
    logger.info(f"Saved raw diarization to: {raw_transcript_path}")
    logger.info(f"Saved converted transcript to: {transcript_path}")
    logger.info(f"Converted {len(converted_segments)} segments for preprocessing")

    return state


def preprocess_node(state: Dict[str, Any]):
    logger.info("Running TextPreprocessor...")

    # Configure cleaning mode based on use case
    # Options: "light", "moderate", "aggressive"
    cleaning_mode = state.get("cleaning_mode", "light")

    tp = TextPreprocessor(results_dir="preprocessed_results", redact_pii=True)
    out = tp.preprocess_transcript(
        state["transcript_path"], state["request_id"], cleaning_mode=cleaning_mode
    )

    state["preprocessed_txt"] = out["saved_txt_path"]
    state["preprocessed_json"] = out["saved_json_path"]
    state["preprocessed_detailed"] = out.get("saved_detailed_path")

    logger.info(f"Preprocessing done: {out}")
    logger.info(f"Cleaning mode: {cleaning_mode}")
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


# NEW NODE
def key_moments_node(state: Dict[str, Any]):
    logger.info("Running Key Moments Extraction...")

    km_node = KeyMomentsNode(results_dir="key_moments_results")
    out = km_node(state)

    state.update(out)
    logger.info(f"Key moments saved: {out.get('key_moments_path')}")
    logger.info(f"Found {out['key_moments']['summary']['total_moments']} moments")
    return state


def merge_node(state: Dict[str, Any]):
    logger.info("Running MergeNode...")

    merger = MergeNode(results_dir="merged_results")

    out = merger.merge(
        summary_path=state["summary"]["paths"]["json"],
        sentiment_path=state["sentiment"]["saved_path"],
        topic_path=state["topics"]["topic_path"],
        key_moments_path=state["key_moments_path"],  # NEW PARAMETER
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

    # Add all nodes
    graph.add_node("audio_preprocessing", audio_preprocessing_node)
    graph.add_node("diarization", diarization_node)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("summarize", summarizer_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("topics", topics_node)
    graph.add_node("key_moments", key_moments_node)  # NEW NODE
    graph.add_node("merge", merge_node)

    # Define edges (pipeline flow)
    graph.add_edge("audio_preprocessing", "diarization")
    graph.add_edge("diarization", "preprocess")
    graph.add_edge("preprocess", "summarize")
    graph.add_edge("summarize", "sentiment")
    graph.add_edge("sentiment", "topics")
    graph.add_edge("topics", "key_moments")  # NEW EDGE
    graph.add_edge("key_moments", "merge")  # UPDATED EDGE

    # Set entry and finish points
    graph.set_entry_point("audio_preprocessing")
    graph.set_finish_point("merge")

    return graph


# ---------------------------
# RUN
# ---------------------------
def run_pipeline(
    audio_file_path: str,
    request_id: str,
    target_sr: int = 16000,
    do_noise_reduction: bool = False,
    do_trim: bool = True,
    cleaning_mode: str = "light",
):
    """
    Run the complete pipeline from raw audio file to merged results

    Args:
        audio_file_path: Path to the raw audio file (mp3, wav, etc.)
        request_id: Unique identifier for this request
        target_sr: Target sample rate for audio preprocessing (default: 16000)
        do_noise_reduction: Whether to apply noise reduction (default: False)
        do_trim: Whether to trim silences (default: True)
        cleaning_mode: Text cleaning level - "light", "moderate", or "aggressive" (default: "light")
            - "light": Minimal cleaning, preserves natural speech and readability
            - "moderate": Removes fillers, cleans formatting
            - "aggressive": Heavy cleaning (may reduce readability)

    Returns:
        Final state dictionary with all results
    """
    graph = build_graph()
    logger.info("Compiling LangGraph pipeline...")
    app = graph.compile()

    init = {
        "audio_file_path": audio_file_path,
        "request_id": request_id,
        "target_sr": target_sr,
        "do_noise_reduction": do_noise_reduction,
        "do_trim": do_trim,
        "cleaning_mode": cleaning_mode,
    }

    logger.info(f"Starting pipeline for audio: {audio_file_path}")
    logger.info(
        f"Audio preprocessing settings: sr={target_sr}, "
        f"noise_reduction={do_noise_reduction}, trim={do_trim}"
    )
    logger.info(f"Text cleaning mode: {cleaning_mode}")
    logger.info("Invoking pipeline...")

    final_state = app.invoke(init)

    logger.info("=" * 60)
    logger.info("Pipeline finished successfully!")
    logger.info("=" * 60)
    logger.info(f"Audio duration: {final_state.get('audio_duration', 0):.2f}s")
    logger.info(
        f"Speakers detected: {final_state.get('diarization_results', {}).get('speakers_count')}"
    )
    logger.info(
        f"Transcript duration: {final_state.get('diarization_results', {}).get('duration'):.2f}s"
    )
    logger.info(f"Topics: {final_state.get('topics', {}).get('topics')}")
    logger.info(
        f"Key Moments: {final_state.get('key_moments', {}).get('summary', {}).get('total_moments')} found"
    )
    logger.info(f"Merged file: {final_state.get('merged', {}).get('merged_path')}")
    logger.info("=" * 60)

    return final_state


if __name__ == "__main__":
    # Example: Run with audio file
    # Use absolute path or correct relative path
    import os

    # Debug: Check if environment variables are loaded
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    if deepgram_key:
        logger.info(f"✓ DEEPGRAM_API_KEY loaded (length: {len(deepgram_key)})")
    else:
        logger.error("✗ DEEPGRAM_API_KEY not found in environment")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error("Make sure your .env file:")
        logger.error("  1. Is in the project root")
        logger.error("  2. Has no spaces around '=' (DEEPGRAM_API_KEY=yourkey)")
        logger.error("  3. python-dotenv is installed: pip install python-dotenv")
        sys.exit(1)

    # Option 1: Absolute path (recommended)
    audio_path = (
        "/home/shinay/Documents/tutorials/pythonProjects/Call-Insight/test_5min.wav"
    )

    # Option 2: Relative path from script location
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # audio_path = os.path.join(script_dir, "..", "test_5min.wav")

    # Option 3: Check if file exists before running
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found at: {audio_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Please provide the correct absolute path to your audio file")
        sys.exit(1)

    run_pipeline(
        audio_file_path=audio_path,
        request_id="demo_run",
        target_sr=16000,
        do_noise_reduction=True,
        do_trim=True,
        cleaning_mode="light",  # Options: "light", "moderate", "aggressive"
    )
