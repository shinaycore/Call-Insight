import operator
import os
from datetime import datetime
from typing import Annotated, Any, Dict, Optional, TypedDict

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from utils.logger import get_logger

logger = get_logger(__name__)


# Define the state schema for LangGraph
class AudioPreprocessState(TypedDict):
    """State schema for audio preprocessing workflow"""

    input_path: str
    request_id: Optional[str]
    out_path: Optional[str]
    sr: Optional[int]
    duration: Optional[float]
    processing_duration: Optional[float]
    error: Optional[str]
    # Optional params
    target_sr: Optional[int]
    do_noise_reduction: Optional[bool]
    do_trim: Optional[bool]
    out_folder: Optional[str]
    messages: Annotated[list, operator.add]


class AudioPreprocessingNode:
    """LangGraph-compatible audio preprocessing node"""

    def __init__(
        self, default_sr: int = 16000, default_out_folder: str = "preprocessed_audio"
    ):
        self.default_sr = default_sr
        self.default_out_folder = default_out_folder
        logger.info(f"AudioPreprocessingNode initialized with sr={default_sr}")

    def load_audio(self, file_path: str, target_sr: int = 16000):
        """Load and resample audio file"""
        try:
            logger.info(f"Loading audio file: {file_path}")
            audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
            logger.debug(f"Original sample rate: {sample_rate}, samples: {len(audio)}")

            audio_resampled = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=target_sr
            )
            audio_normalized = audio_resampled / np.max(np.abs(audio_resampled))

            logger.info(f"Audio resampled and normalized to {target_sr} Hz")
            return audio_normalized, target_sr
        except Exception as e:
            logger.error(f"Error while loading audio {file_path}: {e}")
            raise

    def noise_reduction(
        self, audio: np.ndarray, sr: int = 16000, stationary: bool = False
    ) -> np.ndarray:
        """Apply noise reduction to audio"""
        try:
            logger.info(f"Running noise reduction (stationary={stationary})")
            noise_clip = audio[: int(0.5 * sr)] if len(audio) > sr // 2 else audio

            audio_denoised = nr.reduce_noise(
                y=audio,
                sr=sr,
                y_noise=noise_clip,
                stationary=stationary,
                prop_decrease=1.0,
                n_fft=512,
                time_mask_smooth_ms=50,
                freq_mask_smooth_hz=500,
            )
            logger.info("Noise reduction completed")
            return audio_denoised
        except Exception as e:
            logger.error(f"Error during noise reduction: {e}")
            raise

    def remove_long_silences(
        self, audio: np.ndarray, sr: int = 16000, top_db: float = 30
    ) -> np.ndarray:
        """Trim long silences from audio"""
        try:
            logger.info(f"Trimming long silences (top_db={top_db})")
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            logger.info(f"Silence trimmed: {len(audio)} → {len(audio_trimmed)} samples")
            return audio_trimmed
        except Exception as e:
            logger.error(f"Error during silence removal: {e}")
            raise

    def segment_and_save_audio(
        self,
        audio: np.ndarray,
        sr: int,
        chunk_length: int = 60,
        overlap: int = 2,
        base_folder: str = "preprocessed_audio",
    ) -> list:
        """Segment long audio into chunks (optional utility method)"""
        try:
            os.makedirs(base_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_folder = os.path.join(base_folder, f"chunks_{timestamp}")
            os.makedirs(chunk_folder, exist_ok=True)

            logger.info(
                f"Segmenting into {chunk_length}s chunks with {overlap}s overlap"
            )

            samples_per_chunk = chunk_length * sr
            overlap_samples = overlap * sr
            file_paths = []

            start = 0
            chunk_index = 0

            while start < len(audio):
                end = min(start + samples_per_chunk, len(audio))
                chunk = audio[start:end]

                filename = f"{chunk_index:02d}.wav"
                output_path = os.path.join(chunk_folder, filename)
                sf.write(output_path, chunk, sr, subtype="PCM_16")
                file_paths.append(output_path)

                logger.debug(f"Chunk saved: {filename} ({start}:{end})")

                if end == len(audio):
                    break
                start = end - overlap_samples
                chunk_index += 1

            logger.info(
                f"Segmentation complete: {len(file_paths)} chunks saved in {chunk_folder}"
            )
            return file_paths
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            raise

    def __call__(self, state: AudioPreprocessState) -> AudioPreprocessState:
        """
        LangGraph node function - callable interface
        Takes state, returns updated state
        """
        start_ts = datetime.utcnow()

        try:
            # Get input path
            input_path = state.get("input_path")
            if not input_path:
                return {
                    **state,
                    "error": "missing input_path",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Preprocessing failed: missing input_path",
                        }
                    ],
                }

            # Get parameters with defaults
            target_sr = int(state.get("target_sr", self.default_sr))
            do_noise = bool(state.get("do_noise_reduction", False))
            do_trim = bool(state.get("do_trim", True))
            out_folder = state.get("out_folder", self.default_out_folder)
            request_id = state.get("request_id")

            os.makedirs(out_folder, exist_ok=True)

            # Idempotency: if file already exists, skip processing
            if request_id:
                possible = os.path.join(out_folder, f"audio_{request_id}.wav")
                if os.path.exists(possible):
                    duration = librosa.get_duration(filename=possible)
                    logger.info(f"Using cached preprocessed audio: {possible}")
                    return {
                        **state,
                        "out_path": possible,
                        "sr": target_sr,
                        "duration": duration,
                        "processing_duration": 0.0,
                        "error": None,
                        "messages": [
                            {
                                "role": "system",
                                "content": f"Using cached audio: {duration:.2f}s",
                            }
                        ],
                    }

            # 1) Load audio
            audio, sr = self.load_audio(input_path, target_sr=target_sr)

            # 2) Noise reduction
            if do_noise:
                audio = self.noise_reduction(audio, sr=sr)

            # 3) Trim silence
            if do_trim:
                audio = self.remove_long_silences(audio, sr=sr)

            # 4) Save preprocessed file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"audio_{request_id}_{timestamp}.wav"
                if request_id
                else f"audio_{timestamp}.wav"
            )
            out_path = os.path.join(out_folder, filename)
            sf.write(out_path, audio, sr, subtype="PCM_16")
            logger.info(f"Preprocessed audio saved: {out_path}")

            duration = len(audio) / sr
            processing_time = (datetime.utcnow() - start_ts).total_seconds()

            return {
                **state,
                "out_path": out_path,
                "sr": sr,
                "duration": duration,
                "processing_duration": processing_time,
                "error": None,
                "messages": [
                    {
                        "role": "system",
                        "content": f"Audio preprocessed: {duration:.2f}s @ {sr}Hz (noise_reduction={do_noise}, trim={do_trim})",
                    }
                ],
            }

        except Exception as exc:
            logger.error(f"audio_preprocess_node failed: {exc}")
            return {
                **state,
                "error": f"{type(exc).__name__}: {str(exc)}",
                "out_path": None,
                "messages": [
                    {"role": "system", "content": f"Preprocessing error: {str(exc)}"}
                ],
            }
