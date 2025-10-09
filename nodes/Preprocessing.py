# *If we were training a model from scratch, we'd do more. But since we're using Whisper, 
# preprocessing is lighter and mostly cleanup.*

# Load audio → Bring it into Python, any format.
# Resampling → Mono, 16 kHz, 16-bit PCM (Whisper’s favorite).
# Normalization → scales the amplitude of the audio signal to a specific range
# Noise reduction → Kill hiss, hum, café chatter, or your neighbor’s lawnmower. (coming soon)
# Remove long silences → Cut dead air so we don’t waste compute. (coming soon)
# Segmentation/Framing* → Optional, for long recordings.
# Feature Extraction* → Optional, for ML models.
# Save as clean WAV → Ready for Whisper or any downstream magic.

# load_audio → noise_reduction → remove_long_silences → segment_and_save_audio
import os
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from datetime import datetime
from typing import Optional, List, Dict, Any  

from utils.logger import get_logger
logger = get_logger(__name__)

class audio_preprocessing:
    def load_audio(self, file_path: str, target_sr: int = 16000):
        try:
            logger.info(f"Loading audio file: {file_path}")
            audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
            logger.debug(f"Original sample rate: {sample_rate}, samples: {len(audio)}")

            audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
            audio_normalized = audio_resampled / np.max(np.abs(audio_resampled))

            logger.info(f"Audio resampled and normalized to {target_sr} Hz")
            return audio_normalized, target_sr
        except Exception as e:
            logger.error(f"Error while loading audio {file_path}: {e}")
            raise

    def noise_reduction(self, audio: np.ndarray, sr: int = 16000, stationary: bool = False) -> np.ndarray:
        try:
            logger.info(f"Running noise reduction (stationary={stationary})")
            noise_clip = audio[:int(0.5 * sr)] if len(audio) > sr // 2 else audio

            audio_denoised = nr.reduce_noise(
                y=audio,
                sr=sr,
                y_noise=noise_clip,
                stationary=stationary,
                prop_decrease=1.0,
                n_fft=512,
                time_mask_smooth_ms=50,
                freq_mask_smooth_hz=500
            )
            logger.info("Noise reduction completed")
            return audio_denoised
        except Exception as e:
            logger.error(f"Error during noise reduction: {e}")
            raise

    def remove_long_silences(self, audio: np.ndarray, sr: int = 16000, top_db: float = 30) -> np.ndarray:
        try:
            logger.info(f"Trimming long silences (top_db={top_db})")
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            logger.info(f"Silence trimmed: {len(audio)} → {len(audio_trimmed)} samples")
            return audio_trimmed
        except Exception as e:
            logger.error(f"Error during silence removal: {e}")
            raise

    def save_audio(self, audio: np.ndarray, sr: int, folder: str = "preprocessed_audio"):
        try:
            os.makedirs(folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(folder, f"audio_{timestamp}.wav")
            sf.write(output_path, audio, sr, subtype="PCM_16")
            logger.info(f"Audio saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            raise

    def segment_and_save_audio(
        self,
        audio: np.ndarray,
        sr: int,
        chunk_length: int = 60,
        overlap: int = 2,
        base_folder: str = "preprocessed_audio"
    ) -> list:
        try:
            os.makedirs(base_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_folder = os.path.join(base_folder, f"chunks_{timestamp}")
            os.makedirs(chunk_folder, exist_ok=True)

            logger.info(f"Segmenting into {chunk_length}s chunks with {overlap}s overlap")

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

            logger.info(f"Segmentation complete: {len(file_paths)} chunks saved in {chunk_folder}")
            return file_paths
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            raise


_processor = audio_preprocessing()

def audio_preprocess_node(state: Dict[str, Any], config: Optional[Dict[str, Any]] = None, runtime=None) -> Dict[str, Any]:
    start_ts = datetime.utcnow()
    try:
        # required input
        input_path = state.get("input_path")
        if not input_path:
            return {"error": "missing input_path"}

        # params
        target_sr = int(state.get("target_sr", 16000))
        do_noise = bool(state.get("do_noise_reduction", False))
        do_trim = bool(state.get("do_trim", True))
        out_folder = state.get("out_folder", "preprocessed_audio")
        request_id = state.get("request_id")

        os.makedirs(out_folder, exist_ok=True)

        # idempotency: if file already exists, skip processing
        if request_id:
            possible = os.path.join(out_folder, f"audio_{request_id}.wav")
            if os.path.exists(possible):
                duration = librosa.get_duration(filename=possible)
                return {
                    "out_path": possible,
                    "sr": target_sr,
                    "duration": duration,
                    "processing_duration": 0.0,
                    "request_id": request_id
                }

        # 1) load
        audio, sr = _processor.load_audio(input_path, target_sr=target_sr)

        # 2) noise reduction
        if do_noise:
            audio = _processor.noise_reduction(audio, sr=sr)

        # 3) trim silence
        if do_trim:
            audio = _processor.remove_long_silences(audio, sr=sr)

        # 4) save preprocessed file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{request_id}_{timestamp}.wav" if request_id else f"audio_{timestamp}.wav"
        out_path = os.path.join(out_folder, filename)
        sf.write(out_path, audio, sr, subtype="PCM_16")
        logger.info(f"Preprocessed audio saved: {out_path}")

        duration = len(audio) / sr

        return {
            "out_path": out_path,
            "sr": sr,
            "duration": duration,
            "processing_duration": (datetime.utcnow() - start_ts).total_seconds(),
            "request_id": request_id
        }

    except Exception as exc:
        logger.error(f"audio_preprocess_node failed: {exc}")
        return {"error": f"{type(exc).__name__}: {str(exc)}", "request_id": state.get("request_id")}