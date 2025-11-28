import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import requests
from dotenv import load_dotenv
from utils.logger import get_logger

logger = get_logger(__name__)


class SummarizationNode:
    def __init__(self, results_dir: str = "summarization_results"):
        load_dotenv(Path(__file__).parents[1] / ".env", override=True)

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY missing")

        self.model = "mistralai/mistral-7b-instruct:free"
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.results_dir = Path(results_dir)
        self.cache_dir = Path("cache_summaries")

        self.results_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

    def _parse_txt(self, txt_file: str) -> Dict[str, str]:
        """Parse preprocessed TXT into speaker blocks."""
        speakers = {}
        current = None
        buffer = []
        started = False  # Flag to track if we've started parsing content

        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                original_line = line
                line = line.strip()

                # Skip header lines until we find actual content
                if not started:
                    if line.startswith(
                        "TRANSCRIPT FOR SUMMARIZATION"
                    ) or line.startswith("="):
                        continue
                    started = True

                # Stop at footer sections
                if line.startswith("===") and "End of" in line:
                    break
                if line.startswith("Note: PII redacted"):
                    break

                # Match "Speaker X:" format
                if line.startswith("Speaker") and line.endswith(":"):
                    # Save previous speaker's content
                    if current and buffer:
                        speakers[current] = "\n".join(buffer)
                        logger.debug(
                            f"Saved {current}: {len(buffer)} lines, {len(speakers[current])} chars"
                        )

                    # Start new speaker
                    current = line[:-1].lower()  # Remove colon, lowercase
                    buffer = []
                    logger.debug(f"Found speaker: {current}")

                # Add content to current speaker
                elif current and line and not line.startswith("="):
                    buffer.append(line)

        # Save last speaker
        if current and buffer:
            speakers[current] = "\n".join(buffer)
            logger.debug(f"Saved final {current}: {len(buffer)} lines")

        logger.info(f"Parse complete. Found speakers: {list(speakers.keys())}")
        for spk, text in speakers.items():
            logger.info(f"  {spk}: {len(text)} chars")

        return speakers

    def _get_cached(self, text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()
        cache_file = self.cache_dir / f"{h}.txt"
        return cache_file.read_text() if cache_file.exists() else None

    def _save_cache(self, text: str, result: str):
        h = hashlib.sha256(text.encode()).hexdigest()
        (self.cache_dir / f"{h}.txt").write_text(result)

    def _call_api(self, prompt: str, max_tokens: int = 400) -> str:
        """Call LLM API with retry logic."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional summarizer. Extract key information concisely. "
                        "Never invent details. If content is unclear, state that explicitly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "CallInsightApp",
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    self.endpoint, headers=headers, json=payload, timeout=60
                )

                if resp.status_code == 429:
                    time.sleep(2**attempt)
                    continue

                if resp.status_code == 404:
                    logger.error(
                        f"404 Error - Model or endpoint not found. Model: {self.model}"
                    )
                    logger.error(f"Response: {resp.text[:200]}")
                    return ""

                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()

                # If we got actual content, return it
                if len(content) > 20:
                    return content

                logger.warning(f"API returned too short response: '{content}'")
                return ""

            except Exception as e:
                logger.warning(f"API attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    return ""
                time.sleep(2**attempt)

        return ""

    def _summarize_speaker(self, speaker: str, text: str) -> str:
        """Generate speaker-level summary with proper structure."""
        if not text.strip():
            return "No content available."

        # Check cache
        cached = self._get_cached(f"speaker:{speaker}:{text}")
        if cached:
            return cached

        prompt = f"""Summarize what {speaker} discussed in 4-6 clear bullet points.

Focus on:
- Main topics or concerns raised
- Specific problems mentioned
- Actions taken or suggested
- Decisions or outcomes

Text:
{text[:2000]}

Provide ONLY a bullet point list, nothing else."""

        result = self._call_api(prompt)

        # Better fallback - actually parse the text
        if not result or len(result) < 20:
            logger.info(f"Using smart fallback for {speaker}")
            sentences = [
                s.strip()
                for s in text.replace(" .", ".").split(".")
                if len(s.strip()) > 15
            ]

            # Extract meaningful phrases
            points = []
            for sent in sentences[:8]:
                # Clean up the sentence
                clean = sent.strip()
                if len(clean) > 20 and len(clean) < 200:
                    points.append(f"- {clean}")

            result = "\n".join(points[:6]) if points else "Unable to summarize content."

        self._save_cache(f"speaker:{speaker}:{text}", result)
        return result

    def _global_summary(self, speakers: Dict[str, str]) -> str:
        """Generate overall conversation summary."""
        full_text = "\n\n".join(
            f"{spk.upper()}: {txt}" for spk, txt in speakers.items()
        )

        # Check cache
        cached = self._get_cached(f"global:{full_text}")
        if cached:
            return cached

        prompt = f"""Provide a concise summary of this conversation in 3-5 sentences.

Include:
- Main topic or purpose
- Key problems/challenges discussed
- Important decisions or outcomes
- Action items (if any)

Conversation:
{full_text[:3000]}

Summary:"""

        result = self._call_api(prompt, max_tokens=500)

        # Fallback
        if not result or len(result) < 30:
            result = f"Discussion between {', '.join(speakers.keys())}. Topics covered include technical issues and project work."

        self._save_cache(f"global:{full_text}", result)
        return result

    def summarize_node(
        self, txt_file: str, transcript_json=None, request_id: str = "summary"
    ) -> Dict:
        """Main entry point for summarization."""
        try:
            logger.info(f"Processing: {txt_file}")
            speakers = self._parse_txt(txt_file)

            if not speakers:
                logger.error(f"No speakers found. Parsed speakers dict: {speakers}")
                # Try to read file and show first few lines for debugging
                with open(txt_file, "r") as f:
                    first_lines = [f.readline() for _ in range(10)]
                logger.error(f"First 10 lines of file:\n{''.join(first_lines)}")
                raise ValueError("No speakers found in file")

            logger.info(f"Found {len(speakers)} speakers: {list(speakers.keys())}")

            # Generate summaries
            logger.info("Generating global summary...")
            global_notes = self._global_summary(speakers)

            logger.info("Generating speaker summaries...")
            speaker_notes = {}
            for spk, txt in speakers.items():
                speaker_notes[spk] = self._summarize_speaker(spk, txt)
                time.sleep(0.25)  # Rate limiting

            # Save results
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # JSON output
            json_path = self.results_dir / f"{request_id}_{ts}.json"
            output = {
                "request_id": request_id,
                "global_notes": global_notes,
                "speaker_notes": speaker_notes,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            # Human-readable TXT
            txt_path = self.results_dir / f"{request_id}_{ts}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("CONVERSATION SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(global_notes + "\n\n")
                f.write("SPEAKER CONTRIBUTIONS\n")
                f.write("=" * 60 + "\n")
                for spk, notes in speaker_notes.items():
                    f.write(f"\n{spk.upper()}\n{'-' * 40}\n{notes}\n")

            logger.info(f"Saved JSON: {json_path}")
            logger.info(f"Saved TXT: {txt_path}")

            # Return with paths for compatibility
            return {
                "request_id": request_id,
                "global_notes": global_notes,
                "speaker_notes": speaker_notes,
                "paths": {"json": str(json_path), "txt": str(txt_path)},
            }

        except Exception as e:
            logger.error(f"Summarization failed: {e}", exc_info=True)
            return {"error": str(e), "request_id": request_id}
