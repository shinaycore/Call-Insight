import os
import json
from datetime import datetime
from typing import List, Dict, Any
import requests
from utils.logger import get_logger
from utils.json_reader import load_json

from dotenv import load_dotenv
load_dotenv()  # this will load .env vars


logger = get_logger(__name__)

from dotenv import load_dotenv
import os
from pathlib import Path

class SummarizationNode:
    def __init__(self, results_dir: str = "summarization_results", chunk_size: int = 400):
        """
        Summarization node using OpenRouter free model: openai/gpt-oss-20b
        chunk_size: max words per chunk to prevent memory spikes
        """
        # Load environment variables from .env file
        env_path = Path(__file__).parent / ".env"
        load_dotenv(env_path, override=True)
        
        # Get API key from environment
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Validate API key was loaded
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables. Check your .env file.")
        
        self.model_name = "openai/gpt-oss-20b:free"
        self.chunk_size = chunk_size
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"  # OpenRouter endpoint

        
    def split_text_into_chunks(self, text: str) -> List[str]:
        words = text.split()
        return [" ".join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

    def summarize_text(self, text: str) -> str:
        """
        Summarize a chunk into bullet points using OpenRouter.
        """
        prompt = f"Summarize the following conversation into concise bullet points:\n{text}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that summarizes meetings into clear bullet points."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500
        }
        response = requests.post(self.endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def hierarchical_summarize(self, text: str) -> str:
        chunks = self.split_text_into_chunks(text)
        logger.info(f"Splitting text into {len(chunks)} chunks for summarization.")
        chunk_summaries = [self.summarize_text(c) for c in chunks]
        combined = " ".join(chunk_summaries)
        final_summary = self.summarize_text(combined)
        return final_summary

    def speaker_wise_summarize(self, transcript: List[Dict[str, Any]]) -> Dict[str, str]:
        speaker_texts = {}
        for entry in transcript:
            speaker = entry.get("speaker", "unknown")
            speaker_texts.setdefault(speaker, []).append(entry.get("text", ""))

        speaker_summaries = {}
        for spk, texts in speaker_texts.items():
            merged_text = " ".join(texts)
            speaker_summaries[spk] = self.hierarchical_summarize(merged_text)

        return speaker_summaries

    def summarize_node(self, txt_file: str, transcript_json: str, request_id: str = "test") -> Dict[str, Any]:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()

            global_notes = self.hierarchical_summarize(text)
            transcript = load_json(transcript_json)
            speaker_notes = self.speaker_wise_summarize(transcript)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = os.path.join(self.results_dir, f"summary_{request_id}_{timestamp}.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({
                    "global_notes": global_notes,
                    "speaker_notes": speaker_notes
                }, jf, indent=2, ensure_ascii=False)

            txt_path = os.path.join(self.results_dir, f"summary_{request_id}_{timestamp}.txt")
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write("Global Notes:\n")
                tf.write(global_notes + "\n\nSpeaker Notes:\n")
                for spk, notes in speaker_notes.items():
                    tf.write(f"{spk}:\n{notes}\n\n")

            logger.info(f"Summarization results saved: {json_path} and {txt_path}")
            return {
                "request_id": request_id,
                "global_notes": global_notes,
                "speaker_notes": speaker_notes,
                "saved_paths": {"json": json_path, "txt": txt_path}
            }

        except Exception as e:
            logger.error(f"Summarization node failed: {e}")
            return {"error": str(e), "request_id": request_id}
