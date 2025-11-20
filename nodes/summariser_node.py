import hashlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv
from utils.logger import get_logger

logger = get_logger(__name__)

DEBUG_LOG = Path("summarizer_debug.log")


class SummarizationNode:
    def __init__(self, results_dir: str = "summarization_results"):
        """
        FINAL version:
        - Reads ONLY preprocessed .txt
        - Speaker-level summary
        - Global summary
        - Anti-hallucination fallback logic
        - Debug logging
        """
        root_env = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(root_env, override=True)

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY missing in .env")

        self.model = "mistralai/mistral-7b-instruct:free"
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

        self.results_dir = results_dir
        Path(self.results_dir).mkdir(exist_ok=True)

        self.cache_dir = Path("cache_summaries")
        self.cache_dir.mkdir(exist_ok=True)

    # -----------------------------
    # Debug logger
    # -----------------------------
    def _append_debug(self, text: str):
        try:
            with open(DEBUG_LOG, "a", encoding="utf-8") as df:
                df.write(f"{datetime.now().isoformat()} {text}\n")
        except Exception:
            pass

    # -----------------------------
    # Minimal TXT parser
    # -----------------------------
    def _parse_preprocessed_txt(self, txt_file: str) -> Dict[str, str]:
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()

        speakers = {}
        current = None
        buffer = []

        for line in content.splitlines():
            line = line.strip()

            # Speaker headers like "JAYDEN:"
            if line.endswith(":") and line[:-1].isalpha():
                if current and buffer:
                    speakers[current] = "\n".join(buffer).strip()
                current = line[:-1].lower()
                buffer = []
                continue

            if line.startswith("PII SUMMARY"):
                break

            if current:
                if line:
                    buffer.append(line)

        if current and buffer:
            speakers[current] = "\n".join(buffer).strip()

        return speakers

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _load_cache(self, h: str):
        p = self.cache_dir / f"{h}.txt"
        return p.read_text() if p.exists() else None

    def _save_cache(self, h: str, text: str):
        (self.cache_dir / f"{h}.txt").write_text(text)

    # -----------------------------
    # API Wrapper + Debug
    # -----------------------------
    def _chat_request(self, prompt: str, max_tokens: int = 300) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "CallInsightApp",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Summarize ONLY what is explicitly written.\n"
                        "Never add tasks, projects, meetings, or invented info.\n"
                        "If unclear: return NO CONTENT FOUND.\n"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }

        self._append_debug(
            f"SEND prompt_len={len(prompt)} prompt_snip={prompt[:500]!r}"
        )

        retries, backoff = 5, 2

        for attempt in range(retries):
            try:
                resp = requests.post(
                    self.endpoint, headers=headers, json=payload, timeout=60
                )

                snip = resp.text[:500]
                self._append_debug(f"RESP status={resp.status_code} body_snip={snip!r}")

                if resp.status_code == 429:
                    wait = backoff + random.random()
                    time.sleep(wait)
                    backoff *= 2
                    continue

                resp.raise_for_status()
                data = resp.json()
                msg = (
                    (data.get("choices") or [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                self._append_debug(f"FINAL_CONTENT len={len(msg)} snip={msg[:500]!r}")
                return msg.strip()

            except Exception as e:
                self._append_debug(f"ERROR attempt={attempt} exc={repr(e)}")
                if attempt == retries - 1:
                    return ""
                time.sleep(backoff)
                backoff *= 2

        return ""

    # -----------------------------
    # Fallback summarization logic
    # -----------------------------
    def summarize_speaker_block(self, speaker: str, text: str) -> str:
        if not text.strip():
            return "NO CONTENT FOUND."

        strict_prompt = (
            f"Summarize EVERYTHING said by {speaker}.\n"
            "- Only literal content.\n"
            "- No hallucinations.\n"
            "- 5–8 bullet points.\n\n"
            f"TEXT:\n{text}"
        )

        h = self._hash_text(strict_prompt)
        cached = self._load_cache(h)
        if cached and cached.strip():
            return cached

        result = self._chat_request(strict_prompt)
        if result and result.strip() and "NO CONTENT FOUND" not in result.upper():
            self._save_cache(h, result)
            return result

        # fallback prompt
        fallback_prompt = (
            f"Extract 4–6 literal statements from the text.\n"
            "Write exactly what the speaker says.\n"
            "If impossible: NO CONTENT FOUND.\n\n"
            f"TEXT:\n{text}"
        )

        result2 = self._chat_request(fallback_prompt)
        if result2 and result2.strip() and "NO CONTENT FOUND" not in result2.upper():
            self._save_cache(h, result2)
            return result2

        # literal emergency fallback
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            return "Literal excerpt:\n" + "\n".join(lines[:3])

        return "NO CONTENT FOUND."

    # -----------------------------
    # Global summary
    # -----------------------------
    def global_summary(self, text: str) -> str:
        if not text.strip():
            return "NO CONTENT FOUND."

        prompt = (
            "Summarize this conversation.\n"
            "- Only literal facts.\n"
            "- No hallucinations.\n"
            "- 6–10 bullet points.\n\n"
            f"{text}"
        )

        result = self._chat_request(prompt, max_tokens=400)
        if result and result.strip() and "NO CONTENT FOUND" not in result.upper():
            return result

        # fallback literal extraction
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            return "Literal combined excerpts:\n" + "\n".join(lines[:8])

        return "NO CONTENT FOUND."

    # -----------------------------
    # Main node caller
    # -----------------------------
    def summarize_node(self, txt_file: str, transcript_json=None, request_id="summary"):
        try:
            logger.info(f"Reading TXT: {txt_file}")
            speaker_blocks = self._parse_preprocessed_txt(txt_file)

            if not speaker_blocks:
                raise ValueError("No speakers found in TXT file.")

            self._last_speaker_blocks = speaker_blocks  # for fallbacks

            full_text = "\n".join(speaker_blocks.values())

            logger.info("Generating global summary...")
            global_notes = self.global_summary(full_text)

            logger.info("Generating speaker summaries...")
            speaker_notes = {}
            for spk, text in speaker_blocks.items():
                speaker_notes[spk] = self.summarize_speaker_block(spk, text)
                time.sleep(0.25)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = Path(self.results_dir) / f"{request_id}_{ts}.json"
            txt_path = Path(self.results_dir) / f"{request_id}_{ts}.txt"

            # Save JSON
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(
                    {"global_notes": global_notes, "speaker_notes": speaker_notes},
                    jf,
                    indent=2,
                    ensure_ascii=False,
                )

            # Save TXT
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write("GLOBAL SUMMARY\n")
                tf.write(global_notes + "\n\n")
                tf.write("SPEAKER NOTES\n")
                for spk, notes in speaker_notes.items():
                    tf.write(f"\n[{spk.upper()}]\n{notes}\n")

            return {
                "request_id": request_id,
                "global_notes": global_notes,
                "speaker_notes": speaker_notes,
                "paths": {"json": str(json_path), "txt": str(txt_path)},
            }

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {"error": str(e)}
