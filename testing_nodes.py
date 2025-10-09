# # main.py (Call-Insight root)
# import sys
# from pathlib import Path
# import os
# from dotenv import load_dotenv
# from nodes.keyword_extraction import KeywordExtractionNode
# from nodes.conversation_analytics import ConversationAnalyticsNode  # <-- new import

# # --- Make project root visible for imports ---
# project_root = Path(__file__).parent.resolve()
# sys.path.append(str(project_root))

# # --- Load environment variables ---
# env_path = project_root / ".env"
# load_dotenv(env_path, override=True)

# def main():
#     # --- Paths to your files ---
#     txt_file = "diarized_transcript.json"
#     transcript_json = "diarized_transcript.json"
#     summary_json_path = Path("summarization_results/summary_test123_20251009_201507.json")  # existing summary

#     # --- Load Existing Summary ---
#     if summary_json_path.exists():
#         print(f"✅ Using existing summary: {summary_json_path}")
#         import json
#         with open(summary_json_path, "r", encoding="utf-8") as f:
#             summary_result = json.load(f)
#     else:
#         raise FileNotFoundError(f"Summary file not found: {summary_json_path}. Run summarization first.")

#     print("Global Notes:\n", summary_result.get("global_notes", ""))
#     print("\nSpeaker Notes:\n", summary_result.get("speaker_notes", {}))
#     print("\nSaved files:", summary_result.get("saved_paths", {}))

#     # --- Initialize Keyword Extraction Node ---
#     keyword_node = KeywordExtractionNode(
#         results_dir=str(project_root / "keyword_results"),
#         top_n=20
#     )

#     # --- Run Keyword Extraction ---
#     keyword_result = keyword_node.extract_node(
#         txt_file=txt_file,
#         transcript_json=transcript_json,
#         request_id="test123"
#     )

#     print("\n✅ Keyword extraction complete!")
#     print("Global Keywords / Action Points:\n", keyword_result.get("global_keywords", []))
#     print("\nSpeaker Keywords:\n", keyword_result.get("speaker_keywords", {}))
#     print("\nSaved files:", {
#         "json": keyword_result.get("saved_json_path"),
#         "txt": keyword_result.get("saved_txt_path")
#     })

#     # --- Initialize Conversation Analytics Node ---
#     analytics_node = ConversationAnalyticsNode(results_dir=str(project_root / "analytics_results"))

#     # --- Run Conversation Analytics on speaker keyword counts ---
#     speaker_keyword_counts = {}
#     for speaker, keywords in keyword_result.get("speaker_keywords", {}).items():
#         # Count frequency of each keyword for each speaker
#         counts = {}
#         for kw in keywords:
#             counts[kw] = keywords.count(kw)
#         speaker_keyword_counts[speaker] = counts

#     analytics_result = analytics_node.run_node(speaker_keyword_counts, request_id="test123")

#     print("\n✅ Conversation analytics complete!")
#     print("Dominant speaker per topic:\n", analytics_result.get("dominant_speaker_per_topic", {}))
#     print("Neglected topics:\n", analytics_result.get("neglected_topics", []))
#     print("Shared topics:\n", analytics_result.get("shared_topics", []))
#     print("Unique topics per speaker:\n", analytics_result.get("unique_topics", {}))
#     print("Heatmap saved at:", analytics_result.get("heatmap_path"))
#     print("JSON results saved at:", analytics_result.get("saved_json_path"))

# if __name__ == "__main__":
#     main()


# main.py (Call-Insight root)
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from nodes.text_preprocessing import TextPreprocessor

# --- Make project root visible for imports ---
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

# --- Load environment variables ---
env_path = project_root / ".env"
load_dotenv(env_path, override=True)


def main():
    # --- Paths to your files ---
    transcript_json = "diarized_transcript.json"  # Whisper STT output JSON

    # --- Initialize Preprocessor ---
    preprocessor = TextPreprocessor(results_dir=str(project_root / "preprocessed_results"))

    # --- Run preprocessing ---
    preprocessed_result = preprocessor.preprocess_transcript(
        transcript_json=transcript_json,
        request_id="test123"
    )

    print("✅ Transcript preprocessing complete!")
    print("\nSpeaker-wise cleaned text:\n")
    for speaker, text in preprocessed_result["speaker_texts"].items():
        print(f"{speaker}: {text[:200]}{'...' if len(text) > 200 else ''}")

    print("\nSaved preprocessed transcript JSON:", preprocessed_result["saved_json_path"])


if __name__ == "__main__":
    main()
