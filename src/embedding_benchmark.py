"""
Benchmark embedding models on CPU with course content.
"""
from sentence_transformers import SentenceTransformer
import time
from pathlib import Path
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from src.srt_parser import parse_srt, group_blocks_by_time

# UPDATED: Testing Higher Quality Models
MODELS_TO_TEST = [
    'BAAI/bge-small-en-v1.5',           # Baseline (Fastest)
    'BAAI/bge-base-en-v1.5',            # Step up in quality (~400MB)
    'Snowflake/snowflake-arctic-embed-m', # Very popular mid-sized model (~400MB)
    'BAAI/bge-m3',                      # The Heavyweight (~2.2GB) - Multi-function
]

def load_sample_chunks(n_files: int = 5) -> list[str]:
    """Load sample chunks from course for testing."""
    course_root = Path("./course_content")
    srt_files = list(course_root.rglob("*.srt"))[:n_files]

    all_chunks = []
    for srt_file in srt_files:
        try:
            blocks = parse_srt(srt_file)
            chunks = group_blocks_by_time(blocks, max_chars=1000)
            all_chunks.extend([c['text'] for c in chunks])
        except Exception:
            continue

    # Return up to 50 chunks
    return all_chunks[:50]

def benchmark_model(model_name: str, texts: list[str]) -> dict:
    """Benchmark a single embedding model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Load model
    start = time.time()
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None
        
    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.2f}s")

    # Embed texts
    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True)
    embed_time = time.time() - start

    chunks_per_second = len(texts) / embed_time

    print(f"✅ Embedded {len(texts)} chunks in {embed_time:.2f}s")
    print(f"   Speed: {chunks_per_second:.1f} chunks/second")
    
    # Estimate time for full course (assuming ~10,000 chunks for safety)
    estimated_full_time = (10000 / chunks_per_second) / 60
    print(f"   Estimated full course (10k chunks): {estimated_full_time:.1f} minutes")

    return {
        'model_name': model_name,
        'chunks_per_second': chunks_per_second,
        'estimated_full_course_minutes': estimated_full_time,
        'model_obj': model,
        'embeddings': embeddings
    }

def test_search_quality(model, embeddings, texts):
    """Test search quality with a sample query."""
    query = "How do I handle exceptions?"
    query_vec = model.encode([query])[0]
    
    # Cosine similarity
    scores = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    )
    
    top_idx = np.argsort(scores)[-1]
    print(f"\n🔎 Test Query: '{query}'")
    print(f"   Best Match (Score: {scores[top_idx]:.3f}):")
    print(f"   \"{texts[top_idx][:100]}...\"")

if __name__ == '__main__':
    print("EMBEDDING MODEL BENCHMARK (TIER 2)")
    print("="*60)

    # Load sample data
    print("\n📁 Loading sample chunks...")
    texts = load_sample_chunks(n_files=10)
    print(f"✅ Loaded {len(texts)} chunks for testing")

    if not texts:
        print("❌ No text chunks found.")
        sys.exit(1)

    results = []
    for model_name in MODELS_TO_TEST:
        result = benchmark_model(model_name, texts)
        if result:
            test_search_quality(result['model_obj'], result['embeddings'], texts)
            results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    for res in results:
        print(f"{res['model_name']:<35} | {res['chunks_per_second']:>5.1f} chunks/s | ~{res['estimated_full_course_minutes']:>4.1f} min full build")
