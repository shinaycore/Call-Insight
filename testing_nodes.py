import re
from collections import defaultdict


def split_by_speaker(transcript: str):
    """
    Splits the transcript into sections based on speaker number.
    Returns a dict mapping 'Speaker X' to list of their lines.
    """

    # Pattern to match:
    # [00:02 - 00:05] Speaker 0: Hi...
    pattern = re.compile(r"\[(.*?)\]\s*Speaker\s+(\d+):\s*(.*)")

    speakers = defaultdict(list)

    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            time_range = match.group(1)
            speaker_id = f"Speaker {match.group(2)}"
            text = match.group(3)

            speakers[speaker_id].append({"time": time_range, "text": text})

    return speakers
