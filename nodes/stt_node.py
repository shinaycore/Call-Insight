import os
from datetime import datetime
from typing import List, Dict, Optional, Any
import whisper
from utils.logger import get_logger

logger = get_logger(__name__)

class STTNode:
    def __init__(self, model_size: str = "small.en"):
        """
        Potato-friendly Whisper STT node: single model, CPU only.
        Defaults to small.en for better accuracy on phone call audio.
        """
        logger.info(f"Loading Whisper model ({model_size}, device=cpu)...")
        self.model = whisper.load_model(model_size, device="cpu")  # force CPU
        logger.info("Whisper model loaded successfully.")

    def transcribe_chunk(self, chunk_path: str) -> Dict[str, str]:
        """
        Transcribe a single audio chunk sequentially.
        """
        try:
            logger.info(f"Transcribing chunk: {chunk_path}")
            result = self.model.transcribe(chunk_path)
            text = result.get("text", "").strip()
            return {"chunk": chunk_path, "text": text}
        except Exception as e:
            logger.error(f"Failed to transcribe {chunk_path}: {e}")
            return {"chunk": chunk_path, "text": "", "error": str(e)}

    def stt_node(
        self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Node interface for your pipeline.
        Expects `state["chunks"]` to be a list of file paths.
        Runs **sequentially** for low-spec PCs.
        """
        start_ts = datetime.utcnow()
        try:
            chunks = state.get("chunks")
            if not chunks:
                return {"error": "missing chunks"}

            logger.info(f"Starting sequential transcription on {len(chunks)} chunks...")
            transcripts = [self.transcribe_chunk(c) for c in chunks]

            full_text = " ".join([t["text"] for t in transcripts if "text" in t])

            result = {
                "transcripts": transcripts,        # per-chunk transcripts
                "full_text": full_text,            # merged transcript
                "num_chunks": len(chunks),
                "processing_duration": (datetime.utcnow() - start_ts).total_seconds(),
                "request_id": state.get("request_id")
            }

            logger.info(f"STT node completed for request_id={state.get('request_id')}")
            return result

        except Exception as exc:
            logger.error(f"stt_node failed: {exc}")
            return {"error": f"{type(exc).__name__}: {str(exc)}", "request_id": state.get("request_id")}