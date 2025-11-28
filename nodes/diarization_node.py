"""
Deepgram Diarization Node for LangGraph
Standalone node that processes local audio files and returns diarized transcription
Uses direct HTTP API to avoid SDK version issues
"""

import os
from typing import Any, Dict, List, Optional, TypedDict

import requests
from utils.logger import get_logger

logger = get_logger(__name__)


class DiarizationState(TypedDict):
    """Minimal state schema for diarization node"""

    # Inputs
    audio_file_path: Optional[str]

    # Outputs
    full_transcript: Optional[str]
    speaker_segments: Optional[List[Dict[str, Any]]]
    speakers_count: Optional[int]
    duration: Optional[float]
    error: Optional[str]


class DiarizationNode:
    """
    LangGraph node for audio diarization using Deepgram

    Uses direct HTTP API for maximum compatibility
    Processes local audio files only
    """

    def __init__(
        self, api_key: Optional[str] = None, model: str = "nova-2", language: str = "en"
    ):
        """
        Initialize Deepgram client

        Args:
            api_key: Deepgram API key (if None, reads from DEEPGRAM_API_KEY env var)
            model: Deepgram model to use (default: nova-2)
            language: Language code (default: en)
        """
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key not provided. Set DEEPGRAM_API_KEY environment variable "
                "or pass api_key parameter"
            )

        self.model = model
        self.language = language
        self.base_url = "https://api.deepgram.com/v1/listen"
        logger.info(
            f"DiarizationNode initialized with model={model}, language={language}"
        )

    def __call__(self, state: Dict) -> Dict:
        """
        LangGraph node function - processes audio and updates state

        Args:
            state: Current state (must contain 'audio_file_path')

        Returns:
            Updated state with diarization results
        """
        logger.info("Starting diarization process")

        try:
            # Check if we have audio input
            audio_file = state.get("audio_file_path")

            if not audio_file:
                logger.error("No audio source provided in state")
                return {
                    **state,
                    "error": "No audio_file_path provided in state",
                    "full_transcript": None,
                    "speaker_segments": None,
                    "speakers_count": None,
                    "duration": None,
                }

            # Process audio file
            logger.info(f"Processing local file: {audio_file}")
            result = self._process_file(audio_file)

            if result:
                logger.info(
                    f"Diarization complete - Duration: {result['duration']:.2f}s, "
                    f"Speakers: {result['speakers_count']}, "
                    f"Segments: {len(result['speaker_segments'])}"
                )

                # Update state with results
                return {
                    **state,
                    "full_transcript": result["full_transcript"],
                    "speaker_segments": result["speaker_segments"],
                    "speakers_count": result["speakers_count"],
                    "duration": result["duration"],
                    "error": None,
                }
            else:
                logger.error("Diarization returned no results")
                return {
                    **state,
                    "error": "Diarization processing failed - no results returned",
                }

        except FileNotFoundError as e:
            logger.error(f"Audio file not found: {e}")
            return {**state, "error": f"Audio file not found: {str(e)}"}

        except Exception as e:
            logger.exception(f"Unexpected error during diarization: {e}")
            return {**state, "error": f"Diarization error: {str(e)}"}

    def _process_file(self, audio_file_path: str) -> Optional[Dict]:
        """
        Process local audio file

        Args:
            audio_file_path: Path to audio file

        Returns:
            Parsed diarization results
        """
        try:
            logger.debug(f"Reading audio file: {audio_file_path}")
            with open(audio_file_path, "rb") as audio:
                audio_data = audio.read()

            logger.debug(f"Audio file size: {len(audio_data)} bytes")

            # Build query params
            params = {
                "model": self.model,
                "smart_format": "true",
                "diarize": "true",
                "punctuate": "true",
                "utterances": "true",
                "language": self.language,
            }

            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav",  # Adjust based on file type
            }

            logger.debug("Sending request to Deepgram API")
            response = requests.post(
                self.base_url,
                params=params,
                headers=headers,
                data=audio_data,
                timeout=300,  # 5 min timeout
            )

            response.raise_for_status()
            return self._parse_response(response.json())

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request error: {e}")
            if hasattr(e.response, "text"):
                logger.error(f"Response body: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise

    def _parse_response(self, response: Dict) -> Dict:
        """
        Parse Deepgram API response

        Args:
            response: Deepgram API response dict

        Returns:
            Dictionary with parsed results
        """
        try:
            logger.debug("Parsing Deepgram response")

            # Extract full transcript
            full_transcript = response["results"]["channels"][0]["alternatives"][0][
                "transcript"
            ]

            # Extract speaker utterances
            utterances = response["results"]["utterances"]
            logger.debug(f"Found {len(utterances)} utterances")

            speaker_segments = []
            speakers = set()

            for utterance in utterances:
                segment = {
                    "speaker": f"Speaker {utterance['speaker']}",
                    "text": utterance["transcript"],
                    "start": utterance["start"],
                    "end": utterance["end"],
                    "confidence": utterance["confidence"],
                }
                speaker_segments.append(segment)
                speakers.add(utterance["speaker"])

            # Get audio duration
            duration = response["metadata"]["duration"]

            logger.debug(f"Parsed {len(speakers)} unique speakers")

            return {
                "full_transcript": full_transcript,
                "speaker_segments": speaker_segments,
                "speakers_count": len(speakers),
                "duration": duration,
            }

        except KeyError as e:
            logger.error(f"Error parsing response - missing key: {e}")
            logger.error(f"Response structure: {response.keys()}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            raise


# Utility function for formatting
def format_transcript(speaker_segments: List[Dict]) -> str:
    """
    Format speaker segments into readable transcript

    Args:
        speaker_segments: List of speaker segment dicts

    Returns:
        Formatted transcript string
    """
    if not speaker_segments:
        return ""

    formatted = []
    for segment in speaker_segments:
        timestamp = f"[{_format_timestamp(segment['start'])} - {_format_timestamp(segment['end'])}]"
        formatted.append(f"{timestamp} {segment['speaker']}: {segment['text']}")

    return "\n\n".join(formatted)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# Example usage
if __name__ == "__main__":
    # Initialize node (API key read from environment)
    diarizer = DiarizationNode()

    # Process local file
    state = {"audio_file_path": "path/to/meeting.mp3"}
    result = diarizer(state)

    # if not result.get("error"):
    #     print(f"\nDuration: {result['duration']:.2f}s")
    #     print(f"Speakers: {result['speakers_count']}")
    # else:
    #     print(f"Error: {result['error']}")

    # Example: Use in LangGraph
    # from langgraph.graph import StateGraph
    #
    # workflow = StateGraph(dict)
    # workflow.add_node("diarization", diarizer)
    # workflow.set_entry_point("diarization")
    # # ... add more nodes
    # app = workflow.compile()
    # final_state = app.invoke({"audio_file_path": "audio.mp3"})
