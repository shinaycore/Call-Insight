import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from nodes.merge_nodes import MergeNode
from nodes.sentiment_node import SentimentNode
from nodes.summariser_node import SummarizationNode
from nodes.text_preprocessing import TextPreprocessor
from nodes.topic_extractor import llm_topic_extractor_node


def load_and_fix_diarized_json(path: str):
    with open(path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, list) and all(isinstance(x, dict) for x in raw):
        return raw

    if isinstance(raw, dict) and "speaker_texts" in raw:
        return [
            {"speaker": speaker, "text": text}
            for speaker, text in raw["speaker_texts"].items()
        ]

    raise ValueError(f"Unsupported JSON format for diarized transcript: {type(raw)}")


def main():
    transcript_path = "sample_transcript.json"
    request_id = "demo_run"

    tp = TextPreprocessor(results_dir="preprocessed_results", redact_pii=True)
    preprocess_output = tp.preprocess_transcript(transcript_path, request_id=request_id)

    preprocessed_txt_path = preprocess_output["saved_txt_path"]
    preprocessed_json_path = preprocess_output["saved_json_path"]

    diarized_transcript = load_and_fix_diarized_json(preprocessed_json_path)

    summarizer = SummarizationNode(results_dir="summarization_results")
    sentiment = SentimentNode(config_path="config/sentiment_config.json")

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
                {"diarized_transcript": diarized_transcript, "request_id": request_id},
            ): "sentiment",
        }

        results = {}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = {"error": str(e)}

    global_summary_text = results["summary"].get("global_notes", "")

    topic_output = llm_topic_extractor_node(
        {"global_summary": global_summary_text, "request_id": request_id}
    )

    results["topics"] = topic_output

    print("\n=== TOPIC EXTRACTOR RESULT ===\n")
    print(topic_output)

    summary_path = results["summary"].get("paths", {}).get("json")
    sentiment_path = results["sentiment"].get("saved_path")

    merger = MergeNode(results_dir="merged_results")
    merged_output = merger.merge(
        summary_path=summary_path,
        sentiment_path=sentiment_path,
        request_id=request_id,
    )


if __name__ == "__main__":
    main()
