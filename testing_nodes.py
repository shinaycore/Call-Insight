# main.py

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from nodes.merge_nodes import MergeNode  # <-- ADD THIS
from nodes.sentiment_node import SentimentNode
from nodes.summariser_node import SummarizationNode
from nodes.text_preprocessing import TextPreprocessor


def load_and_fix_diarized_json(path: str):
    """
    Fixes the preprocessed JSON into the ONLY format
    the sentiment node will accept:
    [
        {"speaker": "jayden", "text": "..."},
        {"speaker": "mia", "text": "..."}
    ]
    """

    with open(path, "r") as f:
        raw = json.load(f)

    # Case 1: already diarized correctly
    if isinstance(raw, list) and all(isinstance(x, dict) for x in raw):
        return raw

    if isinstance(raw, dict) and "speaker_texts" in raw:
        speaker_texts = raw["speaker_texts"]

        return [
            {"speaker": speaker, "text": text}
            for speaker, text in speaker_texts.items()
        ]

    raise ValueError(f"Unsupported JSON format for diarized transcript: {type(raw)}")


def main():
    transcript_path = "sample_transcript.json"
    request_id = "demo_run"

    # 1. Preprocessing
    tp = TextPreprocessor(results_dir="preprocessed_results", redact_pii=True)
    preprocess_output = tp.preprocess_transcript(transcript_path, request_id=request_id)

    preprocessed_txt_path = preprocess_output["saved_txt_path"]
    preprocessed_json_path = preprocess_output["saved_json_path"]

    # 2. Fix diarized JSON
    diarized_transcript = load_and_fix_diarized_json(preprocessed_json_path)

    # 3. Initialize processing nodes
    summarizer = SummarizationNode(results_dir="summarization_results")
    sentiment = SentimentNode(config_path="config/sentiment_config.json")

    # 4. Run summarizer + sentiment in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                summarizer.summarize_node,
                txt_file=preprocessed_txt_path,
                transcript_json=None,
                request_id=request_id,
            ): "summary",
            executor.submit(
                sentiment.sentiment_analysis_node,
                {
                    "diarized_transcript": diarized_transcript,
                    "request_id": request_id,
                },
            ): "sentiment",
        }

        results = {}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {"error": str(e)}

    # 5. MERGE RESULTS
    merger = MergeNode(results_dir="merged_results")

    summary_path = results["summary"].get("paths", {}).get("json")
    sentiment_path = results["sentiment"].get("saved_path")

    merged_output = merger.merge(
        summary_path=summary_path, sentiment_path=sentiment_path, request_id=request_id
    )

    print("\n=== FINAL OUTPUT ===")
    print(
        json.dumps(
            {
                "summary": results["summary"],
                "sentiment": results["sentiment"],
                "merged": merged_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
