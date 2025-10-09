import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from pydub import AudioSegment
from utils.logger import get_logger

import torch
from pyannote.audio import Pipeline

logger = get_logger(__name__)


# class DiarizationNode:
    def __init__(self, base_folder: str = "diarized_segments"):
        # unique folder per run
        run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        self.out_folder = os.path.join(base_folder, run_id)
        os.makedirs(self.out_folder, exist_ok=True)

        # force CPU (your i3 laptop has no GPU)
        self.device = torch.device("cpu")

        # load diarization pipeline
        # ⚠️ NOTE: pyannote >=2.0 requires HF login. For local/no-HF,
        # use pyannote.audio==1.1 OR download model weights yourself.
        try:
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        except Exception as e:
            logger.error(
                f"Could not load pyannote pipeline (needs HF models). "
                f"Error: {e}"
            )
            raise

        logger.info(f"Segments will be saved in: {self.out_folder}")

    def _cut_audio(self, audio_file: str, start: float, end: float, out_path: str):
        audio = AudioSegment.from_file(audio_file)
        segment = audio[start * 1000 : end * 1000]  # sec → ms
        segment.export(out_path, format="wav")
        return out_path

    # def diarization_node(
        self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        start_ts = datetime.utcnow()
        try:
            audio_file = state.get("input_path")
            if not audio_file or not os.path.exists(audio_file):
                return {"error": "missing or invalid input_path"}

            logger.info(f"Running pyannote diarization on {audio_file}...")

            diarization = self.pipeline({"uri": "sample", "audio": audio_file})

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                spk = f"Speaker_{speaker}"
                chunk_id = str(uuid.uuid4())[:8]
                out_path = os.path.join(self.out_folder, f"{spk}_{chunk_id}.wav")
                self._cut_audio(audio_file, turn.start, turn.end, out_path)

                segments.append({
                    "speaker": spk,
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "chunk_path": out_path
                })

            result = {
                "segments": segments,
                "num_segments": len(segments),
                "processing_duration": (datetime.utcnow() - start_ts).total_seconds(),
                "request_id": state.get("request_id"),
            }
            logger.info(f"Diarization complete: {len(segments)} segments found.")
            return result

        except Exception as e:
            logger.error(f"diarization_node failed: {e}")
            return {
                "error": f"{type(e).__name__}: {str(e)}",
                "request_id": state.get("request_id"),
            }