from nodes.Preprocessing import audio_preprocess_node
from nodes.stt_node import STTNode
import os

if __name__ == "__main__":
    # --- 1️⃣ Audio preprocessing ---
    input_audio = "/home/shinay/Downloads/How to stay calm when you know you'll be stressed Daniel Levitin TED.mp3"

    preproc_state = audio_preprocess_node({
        "input_path": input_audio,
        "target_sr": 16000,
        "do_noise_reduction": True,
        "do_trim": True,
        "do_segment": True,        # segment into chunks
        "chunk_length": 60,        # seconds per chunk
        "overlap": 2,              # seconds overlap
        "out_folder": "test_output",
        "request_id": "test123"
    })

    if "error" in preproc_state:
        print("Preprocessing failed:", preproc_state["error"])
        exit(1)

    chunks = preproc_state.get("chunks", [])
    print(f"Preprocessing complete. {len(chunks)} chunks ready for transcription.")
    if not chunks:
        print("No chunks were created. Exiting.")
        exit(1)

    # --- 2️⃣ Initialize STT node ---
    stt_node = STTNode(model_size="small.en")  # small.en for English-only phone-call audio

    # --- 3️⃣ Run sequential STT ---
    stt_state = {
        "chunks": chunks,
        "request_id": "test123"
    }

    stt_result = stt_node.stt_node(stt_state)

    if "error" in stt_result:
        print("STT failed:", stt_result["error"])
    else:
        print("STT complete!")
        print("\nFull transcript:")
        print(stt_result["full_text"])
        print("\nPer-chunk transcripts:")
        for t in stt_result["transcripts"]:
            chunk_name = os.path.basename(t['chunk'])
            text_preview = t['text'][:50] + ("..." if len(t['text']) > 50 else "")
            print(f"{chunk_name} → {text_preview}")