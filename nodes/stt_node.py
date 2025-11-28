import json
import operator
import os
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import whisper
from utils.logger import get_logger

logger = get_logger(__name__)


# Define the state schema for LangGraph
class STTState(TypedDict):
    """State schema for STT workflow"""

    request_id: Optional[str]
    chunks: Optional[List[str]]  # List of audio file paths to transcribe
    segments: Optional[List[Dict]]  # From diarization node
    transcripts: Optional[List[Dict[str, str]]]
    full_text: Optional[str]
    num_chunks: Optional[int]
    processing_duration: Optional[float]
    saved_paths: Optional[Dict[str, str]]
    error: Optional[str]
    messages: Annotated[list, operator.add]


class STTNode:
    """LangGraph-compatible Speech-to-Text node using Whisper"""

    def __init__(
        self, model_size: str = "small.en", transcript_folder: str = "transcripts"
    ):
        logger.info(f"Loading Whisper model ({model_size}, device=cpu)...")
        self.model = whisper.load_model(model_size, device="cpu")  # force CPU
        self.transcript_folder = transcript_folder
        os.makedirs(self.transcript_folder, exist_ok=True)
        logger.info("Whisper model loaded successfully.")

    def transcribe_chunk(self, chunk_path: str) -> Dict[str, str]:
        """Transcribe a single audio chunk"""
        try:
            logger.info(f"Transcribing chunk: {chunk_path}")
            result = self.model.transcribe(chunk_path)
            text = result.get("text", "").strip()
            return {"chunk": chunk_path, "text": text}
        except Exception as e:
            logger.error(f"Failed to transcribe {chunk_path}: {e}")
            return {"chunk": chunk_path, "text": "", "error": str(e)}

    def save_transcript(
        self,
        text: str,
        transcripts: List[Dict[str, str]],
        request_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Saves full transcript as .txt and per-chunk results as .json
        Returns dict with both paths.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = (
                f"transcript_{request_id}_{timestamp}"
                if request_id
                else f"transcript_{timestamp}"
            )

            # Save full transcript (txt)
            txt_path = os.path.join(self.transcript_folder, f"{base_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text.replace(". ", ".\n"))

            # Save per-chunk transcripts (json)
            json_path = os.path.join(self.transcript_folder, f"{base_name}.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(transcripts, jf, indent=2, ensure_ascii=False)

            logger.info(f"Transcript saved: {txt_path} and {json_path}")
            return {"txt_path": txt_path, "json_path": json_path}
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
            raise

    def __call__(self, state: STTState) -> STTState:
        """
        LangGraph node function - callable interface
        Takes state, returns updated state
        """
        start_ts = datetime.utcnow()

        try:
            # Get chunks from state - could be from preprocessing or diarization
            chunks = state.get("chunks")

            # If no chunks but we have segments from diarization, use those
            if not chunks and state.get("segments"):
                chunks = [seg["chunk_path"] for seg in state["segments"]]

            if not chunks:
                return {
                    **state,
                    "error": "missing chunks or segments to transcribe",
                    "messages": [
                        {
                            "role": "system",
                            "content": "STT failed: no audio chunks provided",
                        }
                    ],
                }

            logger.info(f"Starting sequential transcription on {len(chunks)} chunks...")

            # Transcribe all chunks
            transcripts = [self.transcribe_chunk(c) for c in chunks]

            # Merge only non-empty texts
            full_text = " ".join([t["text"] for t in transcripts if t.get("text")])

            # Save both full transcript + per-chunk transcripts
            saved_paths = self.save_transcript(
                full_text, transcripts, request_id=state.get("request_id")
            )

            processing_time = (datetime.utcnow() - start_ts).total_seconds()

            logger.info(f"STT node completed for request_id={state.get('request_id')}")

            return {
                **state,
                "transcripts": transcripts,
                "full_text": full_text,
                "num_chunks": len(chunks),
                "processing_duration": processing_time,
                "saved_paths": saved_paths,
                "error": None,
                "messages": [
                    {
                        "role": "system",
                        "content": f"Transcription complete: {len(chunks)} chunks, {len(full_text)} characters",
                    }
                ],
            }

        except Exception as exc:
            logger.error(f"stt_node failed: {exc}")
            return {
                **state,
                "error": f"{type(exc).__name__}: {str(exc)}",
                "transcripts": None,
                "full_text": None,
                "messages": [{"role": "system", "content": f"STT error: {str(exc)}"}],
            }
