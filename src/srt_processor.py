"""
Convert SRT files into LlamaIndex Documents with timestamp metadata.
"""
from pathlib import Path
from llama_index.core import Document
from typing import List
import re
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.srt_parser import parse_srt, group_blocks_by_time

def extract_video_metadata(srt_path: Path) -> dict:
    """
    Extract metadata from file path structure.
    Expected: .../Section Name/001 Video Title_en.srt
    """
    # Extract section (immediate parent directory)
    section = srt_path.parent.name

    # Extract video ID and title from filename
    # Example: "001 Introduction_en.srt" -> ID: "001", Title: "Introduction"
    filename = srt_path.stem.replace('_en', '')
    match = re.match(r'(\d+)\s+(.+)', filename)

    if match:
        video_id = match.group(1)
        video_title = match.group(2)
    else:
        video_id = "unknown"
        video_title = filename

    return {
        'section': section,
        'video_id': video_id,
        'video_title': video_title,
        'file_path': str(srt_path)
    }

def srt_to_documents(
    srt_path: Path,
    max_chars: int = 1000
) -> List[Document]:
    """
    Parse SRT file and create LlamaIndex Documents with timestamp metadata.
    """
    try:
        blocks = parse_srt(srt_path)
    except Exception as e:
        print(f"⚠️  Error reading {srt_path}: {e}")
        return []

    if not blocks:
        return []

    # Group blocks into semantic chunks
    chunks = group_blocks_by_time(blocks, max_chars=max_chars)
    
    # Extract metadata once
    metadata = extract_video_metadata(srt_path)

    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            text=chunk['text'],
            metadata={
                **metadata,
                'start_time_seconds': chunk['start_time'],
                'end_time_seconds': chunk['end_time'],
                'chunk_index': i,
                'file_type': 'srt'
            }
        )
        documents.append(doc)

    return documents

def process_all_srt_files(
    course_root: Path,
    max_chars: int = 1000,
    limit: int = None
) -> List[Document]:
    """
    Recursively find and process all SRT files in course directory.
    """
    srt_files = sorted(list(course_root.rglob("*.srt")))

    if limit:
        srt_files = srt_files[:limit]

    all_documents = []
    print(f"📂 Found {len(srt_files)} SRT files. Processing...")

    for i, srt_file in enumerate(srt_files, 1):
        docs = srt_to_documents(srt_file, max_chars)
        all_documents.extend(docs)
        
        if i % 50 == 0:
            print(f"  ...processed {i}/{len(srt_files)} files ({len(all_documents)} chunks so far)")

    print(f"✅ Finished processing. Total chunks: {len(all_documents)}")
    return all_documents
