"""
Test SRT parser with real course files.
"""
from pathlib import Path
import sys
import os

# Add project root to path so we can import src
sys.path.append(os.getcwd())

from src.srt_parser import parse_srt, group_blocks_by_time

def test_single_file():
    """Test parsing a single SRT file."""
    # Find the first available SRT file
    course_root = Path("./course_content")
    try:
        test_file = next(course_root.rglob("*.srt"))
    except StopIteration:
        print("❌ No SRT files found in ./course_content")
        return

    print(f"✅ Testing: {test_file.name}")

    # Parse file
    blocks = parse_srt(test_file)
    print(f"  📊 Parsed {len(blocks)} subtitle blocks")

    if not blocks:
        print("  ❌ Failed to parse blocks")
        return

    # Show first 3 blocks
    for block in blocks[:3]:
        start, end = block.to_seconds()
        print(f"  [{start:.1f}s - {end:.1f}s]: {block.text[:80]}...")

    # Group into chunks
    chunks = group_blocks_by_time(blocks, max_chars=500)
    print(f"\n  📦 Created {len(chunks)} chunks")

    # Show first chunk
    if chunks:
        chunk = chunks[0]
        print(f"  First chunk ({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s):")
        print(f"    {chunk['text'][:200]}...")

def test_multiple_files():
    """Test parsing multiple files to check consistency."""
    course_root = Path("./course_content")
    srt_files = list(course_root.rglob("*.srt"))[:5]

    print(f"\n🔄 Testing {len(srt_files)} files for consistency...")

    results = []
    for srt_file in srt_files:
        try:
            blocks = parse_srt(srt_file)
            chunks = group_blocks_by_time(blocks, max_chars=1000)
            results.append(True)
            print(f"  ✅ {srt_file.name}: {len(blocks)} blocks → {len(chunks)} chunks")
        except Exception as e:
            results.append(False)
            print(f"  ❌ {srt_file.name}: {e}")

    success_rate = sum(results) / len(results) * 100
    print(f"\n  Success rate: {success_rate:.0f}%")

if __name__ == '__main__':
    print("=" * 60)
    print("SRT PARSER TEST SUITE")
    print("=" * 60)
    test_single_file()
    test_multiple_files()
    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
