"""
Test script for DiarizationNode
Run this to test your diarization setup
"""

import os
import sys
from pathlib import Path

from nodes.diarization_node import DiarizationNode, format_transcript

# If your diarization node is in a different file, import it
# from your_module import DiarizationNode, format_transcript
# For now, assuming it's in the same directory


def test_diarization_node():
    """Test the diarization node with sample audio"""

    print("=" * 60)
    print("DIARIZATION NODE TEST")
    print("=" * 60)

    # 1. Get API key
    api_key = "0e09c2dabe3feaf801a09c6633b1b19433ece13a"
    if not api_key:
        print("\n❌ ERROR: DEEPGRAM_API_KEY not found in environment")
        print("\nSet it using:")
        print("  export DEEPGRAM_API_KEY='your_key_here'  # Linux/Mac")
        print("  set DEEPGRAM_API_KEY=your_key_here       # Windows")
        return

    print(f"\n✅ API Key found: {api_key[:10]}...")

    # 2. Import the node
    try:
        from nodes.diarization_node import DiarizationNode, format_transcript

        print("✅ DiarizationNode imported successfully")
    except ImportError as e:
        print(f"\n❌ ERROR: Could not import DiarizationNode: {e}")
        print("\nMake sure diarization_node.py is in the same directory")
        return

    # 3. Initialize the node
    try:
        diarizer = DiarizationNode(api_key)
        print("✅ DiarizationNode initialized")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize node: {e}")
        return

    # 4. Test with local file
    print("\n" + "-" * 60)
    print("TEST 1: Local Audio File")
    print("-" * 60)

    audio_file = input("\nEnter path to audio file (or press Enter to skip): ").strip()

    if audio_file:
        if not os.path.exists(audio_file):
            print(f"❌ File not found: {audio_file}")
        else:
            print(f"\n🎤 Processing: {audio_file}")
            state = {"audio_file_path": audio_file}

            try:
                result = diarizer(state)

                if result.get("error"):
                    print(f"\n❌ Error: {result['error']}")
                else:
                    print("\n✅ SUCCESS!")
                    print(f"\n📊 Results:")
                    print(f"  Duration: {result['duration']:.2f} seconds")
                    print(f"  Speakers: {result['speakers_count']}")
                    print(f"  Segments: {len(result['speaker_segments'])}")

                    print(f"\n📝 Full Transcript Preview (first 200 chars):")
                    print(f"  {result['full_transcript'][:200]}...")

                    print(f"\n🗣️  Speaker Segments (first 3):")
                    for i, segment in enumerate(result["speaker_segments"][:3]):
                        print(
                            f"\n  [{i + 1}] {segment['speaker']} ({segment['start']:.1f}s - {segment['end']:.1f}s)"
                        )
                        print(f'      "{segment["text"]}"')
                        print(f"      Confidence: {segment['confidence']:.2%}")

                    if len(result["speaker_segments"]) > 3:
                        print(
                            f"\n  ... and {len(result['speaker_segments']) - 3} more segments"
                        )

                    # Offer to save full transcript
                    save = (
                        input("\n💾 Save full formatted transcript? (y/n): ")
                        .strip()
                        .lower()
                    )
                    if save == "y":
                        output_file = "transcript_output.txt"
                        with open(output_file, "w") as f:
                            f.write(format_transcript(result["speaker_segments"]))
                        print(f"✅ Saved to: {output_file}")

            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                import traceback

                traceback.print_exc()

    # 5. Test with URL
    print("\n" + "-" * 60)
    print("TEST 2: Audio URL")
    print("-" * 60)

    test_url = input("\nEnter audio URL (or press Enter to skip): ").strip()

    if test_url:
        print(f"\n🌐 Processing URL: {test_url}")
        state = {"audio_url": test_url}

        try:
            result = diarizer(state)

            if result.get("error"):
                print(f"\n❌ Error: {result['error']}")
            else:
                print("\n✅ SUCCESS!")
                print(f"\n📊 Results:")
                print(f"  Duration: {result['duration']:.2f} seconds")
                print(f"  Speakers: {result['speakers_count']}")
                print(f"  Segments: {len(result['speaker_segments'])}")

                print(f"\n📝 Full Transcript Preview:")
                print(f"  {result['full_transcript'][:300]}...")

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback

            traceback.print_exc()

    # 6. Test with sample URL (if user skipped both)
    if not audio_file and not test_url:
        print("\n" + "-" * 60)
        print("TEST 3: Sample Audio URL")
        print("-" * 60)
        print("\n📌 Testing with public sample audio...")

        # Public domain sample audio
        sample_url = "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"

        state = {"audio_url": sample_url}

        try:
            print(f"\n🌐 Processing: {sample_url}")
            result = diarizer(state)

            if result.get("error"):
                print(f"\n❌ Error: {result['error']}")
                print(
                    "\nNote: This is just a music sample (no speech), so transcription may be empty."
                )
            else:
                print("\n✅ Node executed successfully!")
                print(f"  Duration: {result['duration']:.2f} seconds")
                print(f"  Speakers: {result.get('speakers_count', 0)}")

                if result.get("full_transcript"):
                    print(f"\n📝 Transcript: {result['full_transcript']}")
                else:
                    print("\n📝 No speech detected (this is just music)")

        except Exception as e:
            print(f"\n❌ Error: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    print("\n💡 Tips:")
    print("  - Use MP3, WAV, M4A, or other common audio formats")
    print("  - For best results, use clear audio with distinct speakers")
    print("  - Check logs in utils.logger for detailed debug info")
    print("  - Test audio samples: https://www.kozco.com/tech/organfinale.mp3")


if __name__ == "__main__":
    test_diarization_node()
