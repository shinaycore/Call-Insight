import os
import json
from typing import Dict, Any, List
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ProcessPoolExecutor
from utils.logger import get_logger
from utils.json_reader import load_json

logger = get_logger(__name__)

# === Top-level function for multiprocessing (must be top-level to be picklable) ===
def summarize_chunk(args):
    text, model_path = args
    if model_path and os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    else:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # truncate if too long
    max_input_length = 1024
    if len(text.split()) > max_input_length:
        text = " ".join(text.split()[:max_input_length])
    
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]["summary_text"]

# === Node Class ===
class SummarizationNode:
    def __init__(self, config_path: str = "config/summarization_config.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        self.config = load_json(config_path)

        self.max_input_length = self.config.get("max_input_length", 1024)
        self.local_model_path = self.config.get("local_model_path", None)

        self.results_dir = self.config.get("results_dir", "summarization_results")
        os.makedirs(self.results_dir, exist_ok=True)

    def split_text_into_chunks(self, text: str, chunk_size: int = 500):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def global_summarize(self, text_file: str) -> str:
        # Read text from file
        with open(text_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        chunks = self.split_text_into_chunks(content)
        logger.info(f"Text split into {len(chunks)} chunks for parallel summarization.")

        with ProcessPoolExecutor() as executor:
            summaries = list(executor.map(summarize_chunk, [(c, self.local_model_path) for c in chunks]))
        
        # Combine chunk summaries
        return " ".join(summaries)

    def speaker_wise_summarize(self, transcript: List[Dict[str, str]]) -> Dict[str, str]:
        speaker_texts = {}
        for entry in transcript:
            speaker = entry.get("speaker", "unknown")
            speaker_texts.setdefault(speaker, []).append(entry.get("text", ""))

        speaker_summaries = {}
        for spk, texts in speaker_texts.items():
            chunks = self.split_text_into_chunks(" ".join(texts))
            with ProcessPoolExecutor() as executor:
                summaries = list(executor.map(summarize_chunk, [(c, self.local_model_path) for c in chunks]))
            speaker_summaries[spk] = " ".join(summaries)

        return speaker_summaries

    def summarize_node(self, txt_file: str, transcript_json: str, request_id: str = "test") -> Dict[str, Any]:
        try:
            # Global summary from text file
            global_summary = self.global_summarize(txt_file)

            # Speaker-wise summary from JSON
            transcript = load_json(transcript_json)
            speaker_summaries = self.speaker_wise_summarize(transcript)

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = os.path.join(self.results_dir, f"summary_{request_id}_{timestamp}.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({
                    "global_summary": global_summary,
                    "speaker_summaries": speaker_summaries
                }, jf, indent=2, ensure_ascii=False)

            logger.info(f"Summarization results saved: {json_path}")
            return {
                "request_id": request_id,
                "global_summary": global_summary,
                "speaker_summaries": speaker_summaries,
                "saved_path": json_path
            }

        except Exception as e:
            logger.error(f"Summarization node failed: {e}")
            return {"error": str(e), "request_id": request_id}
