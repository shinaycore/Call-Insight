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


# ---------------------------------------------------------
# State Schema
# ---------------------------------------------------------
class AudioPreprocessState(TypedDict):
    input_path: str
    request_id: Optional[str]
    out_path: Optional[str]
    sr: Optional[int]
    duration: Optional[float]
    processing_duration: Optional[float]
    error: Optional[str]

    # optional controls
    target_sr: Optional[int]
    do_noise_reduction: Optional[bool]
    do_trim: Optional[bool]
    out_folder: Optional[str]

    messages: Annotated[list, operator.add]


# ---------------------------------------------------------
# Node
# ---------------------------------------------------------
class AudioPreprocessingNode:
    """
    Audio preprocessing node SAFE for diarization.
    - preserves stereo cues
    - does gentle noise reduction
    - trims silence only if explicitly asked
    """

    def __init__(
        self, default_sr: int = 16000, default_out_folder: str = "preprocessed_audio"
    ):
        self.default_sr = default_sr
        self.default_out_folder = default_out_folder
        logger.info(f"AudioPreprocessingNode initialized with sr={default_sr}")

    # -----------------------------------------------------
    # Safe Audio Loader
    # -----------------------------------------------------
    def load_audio(self, file_path: str, target_sr: int = 16000):
        """Load audio without destroying speaker features."""

        try:
            logger.info(f"Loading audio file: {file_path}")

            # DO NOT force mono — diarization needs channel separation
            audio, sample_rate = librosa.load(file_path, sr=None, mono=False)

            # If multichannel, safe downmix (preserves relative loudness)
            if audio.ndim > 1:
                logger.info(
                    f"Input audio has {audio.shape[0]} channels. Downmixing safely."
                )
                audio = np.mean(audio, axis=0)

            # Resample safely
            audio_resampled = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=target_sr
            )

            # Gentle normalization (avoid flattening dynamics)
            peak = np.max(np.abs(audio_resampled))
            if peak > 0:
                audio_resampled = audio_resampled / peak * 0.95

            logger.info(
                f"Audio loaded and resampled to {target_sr} Hz (diarization-safe)."
            )
            return audio_resampled.astype(np.float32), target_sr

        except Exception as e:
            logger.error(f"Error while loading audio {file_path}: {e}")
            raise

    # -----------------------------------------------------
    # Mild Noise Reduction (safe for diarization)
    # -----------------------------------------------------
    def noise_reduction(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        try:
            logger.info("Applying mild noise reduction (safe for diarization).")

            noise_clip = audio[: int(0.3 * sr)]
            cleaned = nr.reduce_noise(
                y=audio,
                sr=sr,
                y_noise=noise_clip,
                prop_decrease=0.25,  # gentle noise suppression
                stationary=False,
                n_fft=1024,
            )
            return cleaned.astype(np.float32)

        except Exception as e:
            logger.error(f"Error during noise reduction: {e}")
            raise

    # -----------------------------------------------------
    # Optional Silence Trim
    # -----------------------------------------------------
    def remove_long_silences(
        self, audio: np.ndarray, sr: int = 16000, top_db: float = 28
    ):
        """Remove silence. Turn this OFF for diarization."""
        try:
            logger.info(f"Trimming long silences (top_db={top_db}).")
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            logger.info(f"Silence trimmed: {len(audio)} → {len(trimmed)} samples")
            return trimmed.astype(np.float32)

        except Exception as e:
            logger.error(f"Error trimming silence: {e}")
            raise

    # -----------------------------------------------------
    # Save + Pipeline
    # -----------------------------------------------------
    def __call__(self, state: AudioPreprocessState) -> AudioPreprocessState:
        start_ts = datetime.utcnow()

        try:
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

            # Read toggles
            target_sr = int(state.get("target_sr", self.default_sr))
            do_noise = bool(state.get("do_noise_reduction", False))
            do_trim = bool(state.get("do_trim", False))  # default OFF for diarization
            out_folder = state.get("out_folder", self.default_out_folder)
            request_id = state.get("request_id")

            os.makedirs(out_folder, exist_ok=True)

            # Idempotency (cache)
            if request_id:
                cached = os.path.join(out_folder, f"audio_{request_id}.wav")
                if os.path.exists(cached):
                    duration = librosa.get_duration(filename=cached)
                    logger.info(f"Using cached preprocessed audio: {cached}")
                    return {
                        **state,
                        "out_path": cached,
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

            # ================================
            #        PROCESSING STEPS
            # ================================
            audio, sr = self.load_audio(input_path, target_sr=target_sr)

            if do_noise:
                audio = self.noise_reduction(audio, sr=sr)

            if do_trim:
                logger.warning(
                    "Silence trimming enabled — may reduce diarization accuracy."
                )
                audio = self.remove_long_silences(audio, sr=sr)
            else:
                logger.info(
                    "Silence trimming OFF — preserving raw timing for diarization."
                )

            # Save file
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
            elapsed = (datetime.utcnow() - start_ts).total_seconds()

            return {
                **state,
                "out_path": out_path,
                "sr": sr,
                "duration": duration,
                "processing_duration": elapsed,
                "error": None,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"Audio preprocessed: {duration:.2f}s @ {sr}Hz "
                            f"(noise_reduction={do_noise}, trim={do_trim})"
                        ),
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
