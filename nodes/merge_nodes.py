# merge_nodes.py

import json
import os
from datetime import datetime


class MergeNode:
    def __init__(self, results_dir="merged_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def load_json(self, path: str):
        with open(path, "r") as f:
            return json.load(f)

    def merge(self, summary_path: str, sentiment_path: str, request_id: str = None):
        # Load JSON files
        summary = self.load_json(summary_path)
        sentiment = self.load_json(sentiment_path)

        # Build merged output
        merged = {
            "request_id": request_id,
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "sentiment": sentiment,
        }

        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (
            f"merged_{request_id}_{timestamp}.json"
            if request_id
            else f"merged_{timestamp}.json"
        )
        out_path = os.path.join(self.results_dir, name)

        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        return {
            "merged_path": out_path,
            "summary_path": summary_path,
            "sentiment_path": sentiment_path,
        }
