import os
from datetime import datetime
from typing import List, Dict, Optional, Any
import whisper
from utils.logger import get_logger

logger = get_logger(__name__)

class STTNode:
    def __init__(self, model_size: str = "small.en"):
        logger.info(f"Loading Whisper model ({model_size}, device=cpu)...")
        self.model = whisper.load_model(model_size, device="cpu")  # force CPU
        logger.info("Whisper model loaded successfully.")

    def transcribe_chunk(self, chunk_path: str) -> Dict[str, str]:
        try:
            logger.info(f"Transcribing chunk: {chunk_path}")
            result = self.model.transcribe(chunk_path)
            text = result.get("text", "").strip()
            return {"chunk": chunk_path, "text": text}
        except Exception as e:
            logger.error(f"Failed to transcribe {chunk_path}: {e}")
            return {"chunk": chunk_path, "text": "", "error": str(e)}

    def save_transcript(self, text: str, request_id: Optional[str] = None, folder: str = "transcripts") -> str:
        try:
            os.makedirs(folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{request_id}_{timestamp}.txt" if request_id else f"transcript_{timestamp}.txt"
            output_path = os.path.join(folder, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Transcript saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
            raise

    def stt_node(
        self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        start_ts = datetime.utcnow()
        try:
            chunks = state.get("chunks")
            if not chunks:
                return {"error": "missing chunks"}

            logger.info(f"Starting sequential transcription on {len(chunks)} chunks...")
            transcripts = [self.transcribe_chunk(c) for c in chunks]

            full_text = " ".join([t["text"] for t in transcripts if "text" in t])

            # --- save full transcript automatically ---
            saved_path = self.save_transcript(full_text, request_id=state.get("request_id"))

            result = {
                "transcripts": transcripts,        # per-chunk transcripts
                "full_text": full_text,            # merged transcript
                "num_chunks": len(chunks),
                "processing_duration": (datetime.utcnow() - start_ts).total_seconds(),
                "request_id": state.get("request_id"),
                "saved_path": saved_path           # path where transcript was saved
            }

            logger.info(f"STT node completed for request_id={state.get('request_id')}")
            return result

        except Exception as exc:
            logger.error(f"stt_node failed: {exc}")
            return {"error": f"{type(exc).__name__}: {str(exc)}", "request_id": state.get("request_id")}
