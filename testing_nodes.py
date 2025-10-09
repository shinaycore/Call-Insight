from nodes.summariser_node import SummarizationNode

if __name__ == "__main__":
    # Paths
    txt_file = "/home/shinay/Documents/PROJECTS/Call_Insight/transcripts/transcript_test123_20250914_140137.txt"
    transcript_json = "/home/shinay/Documents/PROJECTS/Call_Insight/diarized_transcript.json"

    # Initialize summarizer node
    summarizer = SummarizationNode()

    # Run summarization
    result = summarizer.summarize_node(txt_file=txt_file, transcript_json=transcript_json, request_id="test123")

    # Global summary
    print("\n=== Global Summary ===\n")
    print(result.get("global_summary", "No summary generated."))

    # Speaker-wise summaries
    print("\n=== Speaker-wise Summaries ===\n")
    speaker_summaries = result.get("speaker_summaries", {})
    if speaker_summaries:
        for spk, summary in speaker_summaries.items():
            print(f"[{spk}] {summary}")
    else:
        print("No speaker-wise summaries generated.")

    # Saved JSON
    print(f"\nSaved JSON path: {result.get('saved_path', 'No file saved.')}")
