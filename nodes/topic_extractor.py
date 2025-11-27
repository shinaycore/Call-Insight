import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv
from utils.logger import get_logger

logger = get_logger(__name__)


def llm_topic_extractor_node(state: dict):
    summary_text = state.get("global_summary", "")
    request_id = state.get("request_id", "default")

    if not summary_text.strip():
        return {
            "topics": [],
            "request_id": request_id,
            "topic_path": None,
            "error": "empty summary",
        }

    # Load .env
    root_env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(root_env, override=True)
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY missing in .env")

    # Best FREE models (try in order) - updated for 2024
    free_models = [
        "mistralai/mistral-7b-instruct:free",  # Most reliable
        "meta-llama/llama-3.2-3b-instruct:free",  # Updated Llama
        "google/gemma-2-9b-it:free",
        "qwen/qwen-2-7b-instruct:free",  # Backup option
    ]

    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    # Caching
    cache_dir = Path("cache_topics")
    cache_dir.mkdir(exist_ok=True)

    # UNIVERSAL PROMPT - works for ANY domain (tech, sports, cooking, finance, etc.)
    prompt = f"""Extract key topics from this text that someone could search online to learn more.

Requirements:
- Output ONLY a JSON array: ["topic1", "topic2", ...]
- 5-8 topics, each 1-4 words
- Topics should be specific terms, concepts, techniques, or subjects mentioned
- Avoid generic words like: meeting, discussion, team, project, deadline, management

Examples (adapt to the actual content):
Tech: ["API authentication", "Docker containers", "SQL indexing"]
Sports: ["batting technique", "spin bowling", "pitch conditions"]
Cooking: ["dum cooking", "marination process", "spice balancing"]
Finance: ["mutual funds", "portfolio diversification", "tax optimization"]

Text:
{summary_text[:1200]}

Return only the JSON array:"""

    h = hashlib.sha256(prompt.encode()).hexdigest()
    cache_path = cache_dir / f"{h}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    # Try multiple free models
    def call_openrouter_multi_model():
        for model in free_models:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 250,
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "CallInsight-TopicExtractor",
            }

            retries = 2
            backoff = 1

            for attempt in range(retries):
                try:
                    resp = requests.post(
                        endpoint, headers=headers, json=payload, timeout=25
                    )

                    if resp.status_code == 429:
                        logger.warning(f"Rate limit on {model}, trying next...")
                        break

                    if resp.status_code != 200:
                        logger.warning(f"Error {resp.status_code} on {model}")
                        break

                    data = resp.json()
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )

                    if content:
                        logger.info(f"✓ Success with {model}")
                        return content

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed on {model}: {e}")
                    if attempt < retries - 1:
                        time.sleep(backoff)
                        backoff *= 1.5

            # Try next model if this one failed
            time.sleep(0.5)

        return ""

    raw = call_openrouter_multi_model()

    # Enhanced JSON parsing
    topics = []
    if raw:
        try:
            # Clean up common LLM output patterns
            cleaned = raw.strip()

            # Remove markdown
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", cleaned)

            # Remove explanatory text before/after JSON
            json_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            topics = json.loads(cleaned)

            # Validate and clean
            if isinstance(topics, list):
                topics = [
                    str(t).strip() for t in topics if t and len(str(t).strip()) >= 3
                ][:10]
        except Exception as e:
            logger.warning(f"JSON parse error: {e}")
            # Try line-by-line extraction
            lines = raw.split("\n")
            for line in lines:
                matches = re.findall(r'"([^"]{3,40})"', line)
                topics.extend(matches)
            topics = topics[:10]

    # ENHANCED FALLBACK with multiple strategies
    if not topics or len(topics) < 3:
        logger.info("Using enhanced fallback extraction")
        topics = extract_topics_smart_fallback(summary_text)

    # Final result
    result = {
        "topics": topics,
        "request_id": request_id,
        "topic_path": f"topic_results/{request_id}_topics.json",
    }

    # Save result
    out_dir = Path("topic_results")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{request_id}_topics.json").write_text(json.dumps(result, indent=2))

    cache_path.write_text(json.dumps(result, indent=2))

    return result


def extract_topics_smart_fallback(text: str) -> list:
    """
    Multi-strategy fallback for when LLM fails
    """
    topics = []

    # Strategy 1: Technical patterns (highest priority)
    # Capitalized multi-word terms
    capitalized_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", text)
    topics.extend(capitalized_phrases[:4])

    # Acronyms (2-5 letters)
    acronyms = list(set(re.findall(r"\b[A-Z]{2,5}\b", text)))
    topics.extend(acronyms[:3])

    # Technical terms with special chars
    technical = re.findall(r"\b[a-zA-Z]+(?:[-_./][a-zA-Z0-9]+)+\b", text)
    topics.extend(technical[:3])

    # camelCase or PascalCase
    camel = re.findall(r"\b[a-z]+[A-Z][a-zA-Z]+\b|\b[A-Z][a-z]+[A-Z][a-zA-Z]+\b", text)
    topics.extend(camel[:2])

    # Strategy 2: Look for repeated meaningful terms (domain-agnostic)
    # Instead of hardcoded keywords, find words that appear multiple times
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)

    blacklist = {
        "this",
        "that",
        "with",
        "from",
        "they",
        "them",
        "have",
        "been",
        "will",
        "just",
        "like",
        "know",
        "your",
        "their",
        "were",
        "said",
        "into",
        "before",
        "about",
        "there",
        "which",
        "when",
        "where",
        "would",
        "could",
        "should",
        "these",
        "those",
        "what",
        "some",
        "make",
        "than",
        "then",
        "more",
        "other",
        "such",
        "also",
        "very",
        "after",
        "through",
        "during",
        "does",
        "doing",
        "made",
        "over",
        "each",
        "most",
        "many",
        "much",
        "both",
        "here",
        "only",
        "need",
        "work",
        "call",
        "give",
        "time",
        "good",
        "want",
        "thing",
        "look",
        "meeting",
        "discussion",
        "team",
        "project",
        "deadline",
    }

    word_freq = Counter(
        w.lower() for w in words if w.lower() not in blacklist and len(w) >= 4
    )

    # Get words that appear 2+ times (indicates importance)
    frequent = [w for w, count in word_freq.most_common(20) if count >= 2]
    topics.extend(frequent[:5])

    # Strategy 3: Noun phrases (domain-agnostic patterns)
    # Look for adjective + noun patterns that work across domains
    noun_phrases = re.findall(
        r"\b(?:[a-z]+(?:ing|ed|ive|ous|al|ic))\s+[a-z]{4,}\b", text.lower()
    )
    topics.extend(list(set(noun_phrases))[:3])

    # Strategy 4: High-frequency meaningful words (only if needed)
    if len(topics) < 5:
        blacklist = {
            "this",
            "that",
            "with",
            "from",
            "they",
            "them",
            "have",
            "been",
            "will",
            "just",
            "like",
            "know",
            "your",
            "their",
            "were",
            "said",
            "into",
            "before",
            "about",
            "there",
            "which",
            "when",
            "where",
            "would",
            "could",
            "should",
            "these",
            "those",
            "what",
            "some",
            "make",
            "than",
            "then",
            "more",
            "other",
            "such",
            "also",
            "very",
            "after",
            "through",
            "during",
            "does",
            "doing",
            "made",
            "over",
            "each",
            "most",
            "many",
            "much",
            "both",
            "here",
            "only",
            "need",
            "work",
            "call",
            "give",
            "time",
            "good",
            "want",
            "thing",
            "look",
        }

        words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
        word_freq = Counter(
            w.lower() for w in words if w.lower() not in blacklist and len(w) >= 4
        )

        # Get top words that appear 2+ times
        frequent = [w for w, count in word_freq.most_common(15) if count >= 2]
        topics.extend(frequent[:5])

    # Clean and deduplicate
    seen = set()
    clean_topics = []

    for topic in topics:
        topic_clean = str(topic).strip()
        topic_lower = topic_clean.lower()

        # Skip if too short, already seen, or generic
        if (
            len(topic_clean) < 3
            or topic_lower in seen
            or topic_lower in {"meeting", "discussion", "team", "project", "work"}
        ):
            continue

        seen.add(topic_lower)
        clean_topics.append(topic_clean)

        if len(clean_topics) >= 8:
            break

    return clean_topics if clean_topics else ["discussion topics", "key points"]
