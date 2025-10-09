# keyword_extraction.py
import os, sys
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import Counter
import re
from pathlib import Path

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.logger import get_logger
from utils.json_reader import load_json

logger = get_logger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("Run: python -m spacy download en_core_web_sm")

CUSTOM_NOISE = {"thanks", "good", "well", "yeah", "hello", "bye", "one", "thing", "time", "way"}
STOPWORDS = STOP_WORDS.union(CUSTOM_NOISE)


class KeywordExtractionNode:
    """Extract search-optimized keywords and generate resource queries"""
    
    def __init__(self, results_dir: str = "keyword_results", top_n: int = 10):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.top_n = top_n
    
    def extract_topics_and_questions(self, text: str) -> Dict[str, Any]:
        """
        Extract main topics, problems, and questions that need resources
        Returns search-optimized keywords and suggested queries
        """
        doc = nlp(text.lower())
        
        # Extract key entities and concepts
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "TECH", "GPE", "EVENT", "LAW"]:
                entities.append(ent.text)
        
        # Extract noun phrases (main topics)
        topics = []
        for chunk in doc.noun_chunks:
            # Filter for meaningful multi-word topics
            if len(chunk.text.split()) >= 2 and chunk.root.pos_ in ["NOUN", "PROPN"]:
                clean = chunk.text.strip()
                if clean not in STOPWORDS and len(clean) > 3:
                    topics.append(clean)
        
        # Extract problems/pain points (verb + object patterns)
        problems = []
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp"]:
                # Get verb with its object
                obj = [child.text for child in token.children if child.dep_ in ["dobj", "pobj"]]
                if obj:
                    problems.append(f"{token.text} {' '.join(obj)}")
        
        # Extract questions (sentences with question words)
        questions = []
        for sent in doc.sents:
            if any(token.text.lower() in ["what", "how", "why", "when", "where", "which"] for token in sent):
                questions.append(sent.text.strip())
        
        # TF-IDF for importance scoring
        candidate_keywords = list(set(entities + topics + [p.split()[1] if len(p.split()) > 1 else p for p in problems]))
        
        if candidate_keywords and len(text) > 50:
            vectorizer = TfidfVectorizer(
                vocabulary=candidate_keywords,
                use_idf=True,
                smooth_idf=True,
                ngram_range=(1, 3)
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
            top_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.top_n]
            ranked_keywords = [kw for kw, score in top_keywords if score > 0]
        else:
            ranked_keywords = candidate_keywords[:self.top_n]
        
        return {
            "main_topics": list(set(topics))[:self.top_n],
            "entities": list(set(entities)),
            "problems": list(set(problems))[:5],
            "questions": questions[:5],
            "search_keywords": ranked_keywords
        }
    
    def generate_search_queries(self, keywords_data: Dict[str, Any]) -> List[str]:
        """
        Generate web search queries from extracted keywords
        These are optimized for finding helpful resources
        """
        queries = []
        
        # Topic-based queries
        for topic in keywords_data.get("main_topics", [])[:5]:
            queries.append(f"{topic} guide")
            queries.append(f"how to {topic}")
        
        # Problem-solving queries
        for problem in keywords_data.get("problems", []):
            queries.append(f"solve {problem}")
            queries.append(f"{problem} solution")
        
        # Direct questions
        for question in keywords_data.get("questions", []):
            queries.append(question)
        
        # Entity/product resources
        for entity in keywords_data.get("entities", [])[:3]:
            queries.append(f"{entity} documentation")
            queries.append(f"{entity} best practices")
        
        # General search keywords
        if keywords_data.get("search_keywords"):
            top_kw = " ".join(keywords_data["search_keywords"][:3])
            queries.append(f"{top_kw} tutorial")
            queries.append(f"{top_kw} resources")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        
        return unique_queries[:15]  # Limit to top 15 queries
    
    def extract_from_transcript(self, transcript_json: str) -> Dict[str, Any]:
        """Extract keywords speaker-wise and generate search queries"""
        transcript = load_json(transcript_json)
        
        # Merge all text for global analysis
        all_text = " ".join([entry.get("text", "") for entry in transcript])
        global_keywords = self.extract_topics_and_questions(all_text)
        global_queries = self.generate_search_queries(global_keywords)
        
        # Speaker-wise analysis
        speaker_texts = {}
        for entry in transcript:
            speaker = entry.get("speaker", "unknown")
            speaker_texts.setdefault(speaker, []).append(entry.get("text", ""))
        
        speaker_keywords = {}
        speaker_queries = {}
        for spk, texts in speaker_texts.items():
            merged = " ".join(texts)
            keywords = self.extract_topics_and_questions(merged)
            queries = self.generate_search_queries(keywords)
            
            speaker_keywords[spk] = keywords
            speaker_queries[spk] = queries
        
        return {
            "global": {
                "keywords": global_keywords,
                "search_queries": global_queries
            },
            "by_speaker": {
                spk: {
                    "keywords": speaker_keywords[spk],
                    "search_queries": speaker_queries[spk]
                }
                for spk in speaker_keywords
            }
        }
    
    def extract_node(
        self,
        txt_file: str = None,
        transcript_json: str = None,
        request_id: str = "test"
    ) -> Dict[str, Any]:
        """Run keyword extraction and generate web search queries"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {"request_id": request_id}
            
            # From plain text
            if txt_file:
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read()
                keywords = self.extract_topics_and_questions(text)
                queries = self.generate_search_queries(keywords)
                
                results["global_keywords"] = keywords
                results["global_search_queries"] = queries
                
                # Save readable output
                txt_path = self.results_dir / f"keywords_{request_id}_{timestamp}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("🔍 SEARCH-OPTIMIZED KEYWORDS\n")
                    f.write("=" * 60 + "\n\n")
                    
                    f.write("Main Topics:\n")
                    for topic in keywords.get("main_topics", []):
                        f.write(f"  • {topic}\n")
                    
                    f.write("\nProblems/Issues:\n")
                    for prob in keywords.get("problems", []):
                        f.write(f"  • {prob}\n")
                    
                    f.write("\nQuestions Asked:\n")
                    for q in keywords.get("questions", []):
                        f.write(f"  • {q}\n")
                    
                    f.write("\n" + "=" * 60 + "\n")
                    f.write("🌐 SUGGESTED WEB SEARCHES\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for i, query in enumerate(queries, 1):
                        f.write(f"{i}. {query}\n")
                
                results["saved_txt_path"] = str(txt_path)
                logger.info(f"Keywords saved: {txt_path}")
            
            # From transcript JSON
            if transcript_json:
                transcript_results = self.extract_from_transcript(transcript_json)
                results.update(transcript_results)
            
            # Save JSON
            json_path = self.results_dir / f"keywords_{request_id}_{timestamp}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            results["saved_json_path"] = str(json_path)
            logger.info(f"Keyword extraction complete: {json_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return {"error": str(e), "request_id": request_id}

# Example usage
from pathlib import Path
import sys

if __name__ == "__main__":
    # --- Make project root visible for imports ---
    project_root = Path(__file__).parent.parent.resolve()  # <- note .parent.parent
    sys.path.append(str(project_root))
    
    from nodes.keyword_extraction import KeywordExtractionNode

    extractor = KeywordExtractionNode(top_n=10)
    result = extractor.extract_node(
        transcript_json="/home/shinaycore/PycharmProjects/Call-Insight/diarized_transcript.json",
        request_id="example_001"
    )

    print("\n🔍 Search Queries Generated:")
    for query in result.get("global", {}).get("search_queries", [])[:5]:
        print(f" • {query}")
