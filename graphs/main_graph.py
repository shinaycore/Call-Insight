"""
main_graph.py
LangGraph workflow connecting all processing nodes.
"""

import os
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

# IMPORTING NODES
from nodes.Preprocessing import audio_preprocess_node
from nodes.stt_node import STTNode

# Set PWD to your project root
os.chdir("/home/shinaycore/PycharmProjects/Call-Insight")
print("[DEBUG] New working directory:", os.getcwd())

def build_main_graph():
    """
    Build the LangGraph workflow with audio_preprocess → stt_node.
    """
    workflow = StateGraph(state_schema=Dict[str, Any])

    # 1️⃣ Add audio preprocessing node
    workflow.add_node("audio_preprocess", audio_preprocess_node)

    # 2️⃣ Add STT node
    stt = STTNode(model_size="small.en")  # initialize Whisper
    workflow.add_node("stt_node", stt.stt_node)

    # Define flow
    workflow.add_edge(START, "audio_preprocess")
    workflow.add_edge("audio_preprocess", "stt_node")
    workflow.add_edge("stt_node", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_main_graph()

    test_state = {
        "input_path": "samples/_DayBreak__with_Jay_Young_on_the_USA_Radio_Network.ogg",
        "target_sr": 16000,
        "do_noise_reduction": True,
        "do_trim": True,
        "request_id": "sample_001"
    }

    print("🚀 Running graph with audio_preprocess → stt_node...")
    result = graph.invoke(test_state)
    print(f"✅ Output:\n{result}")
