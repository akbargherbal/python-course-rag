"""
SRT subtitle parser with timestamp preservation.
"""
from dataclasses import dataclass
from datetime import timedelta
import re
from typing import List, Tuple
from pathlib import Path


@dataclass
class SubtitleBlock:
    """Single subtitle entry with timing."""
    index: int
    start_time: timedelta
    end_time: timedelta
    text: str

    def to_seconds(self) -> Tuple[float, float]:
        """Convert timedeltas to seconds for easier handling."""
        return (
            self.start_time.total_seconds(),
            self.end_time.total_seconds()
        )


def parse_srt(file_path: Path) -> List[SubtitleBlock]:
    """
    Parse SRT file into structured subtitle blocks.
    """
    blocks = []

    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Fallback for older encodings
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()

    # Split by double newline (subtitle separator)
    raw_blocks = re.split(r'\n\s*\n', content.strip())

    for raw_block in raw_blocks:
        lines = raw_block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            # Parse index
            index = int(lines[0].strip())

            # Parse timestamps: 00:00:01,000 --> 00:00:03,500
            time_line = lines[1].strip()
            time_match = re.match(
                r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                time_line
            )

            if not time_match:
                continue

            start_time = timedelta(
                hours=int(time_match.group(1)),
                minutes=int(time_match.group(2)),
                seconds=int(time_match.group(3)),
                milliseconds=int(time_match.group(4))
            )

            end_time = timedelta(
                hours=int(time_match.group(5)),
                minutes=int(time_match.group(6)),
                seconds=int(time_match.group(7)),
                milliseconds=int(time_match.group(8))
            )

            # Join remaining lines as text
            text = ' '.join(lines[2:])

            blocks.append(SubtitleBlock(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text
            ))

        except (ValueError, IndexError):
            continue

    return blocks


def group_blocks_by_time(
    blocks: List[SubtitleBlock],
    max_chars: int = 1000
) -> List[dict]:
    """
    Group subtitle blocks into semantic chunks.
    """
    chunks = []
    current_chunk_text = []
    current_chunk_start = None
    current_chunk_end = None
    current_length = 0

    for block in blocks:
        if current_chunk_start is None:
            current_chunk_start = block.start_time

        current_chunk_text.append(block.text)
        current_chunk_end = block.end_time
        current_length += len(block.text)

        # Create chunk if size threshold reached
        if current_length >= max_chars:
            chunks.append({
                'text': ' '.join(current_chunk_text),
                'start_time': current_chunk_start.total_seconds(),
                'end_time': current_chunk_end.total_seconds()
            })

            # Reset for next chunk
            current_chunk_text = []
            current_chunk_start = None
            current_length = 0

    # Handle final chunk
    if current_chunk_text:
        chunks.append({
            'text': ' '.join(current_chunk_text),
            'start_time': current_chunk_start.total_seconds() if current_chunk_start else 0.0,
            'end_time': current_chunk_end.total_seconds() if current_chunk_end else 0.0
        })

    return chunks
