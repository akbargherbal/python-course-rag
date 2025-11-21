"""
Test query engine with representative sample queries.
"""
from pathlib import Path
from src.query_engine import CourseQueryEngine
import json
import time

# Representative test queries
TEST_QUERIES = [
    "How do I raise and handle exceptions?",
    "Explain list comprehensions with examples",
    "What are decorators and how do I use them?",
    "How does memory management work in Python?",
    "What's the difference between args and kwargs?",
    "How do I work with generators?",
    "Explain the difference between class and static methods",
    "How do I create and use context managers?",
]

def test_query_engine():
    """Test query engine on multiple queries."""
    print("🧪 QUERY ENGINE TEST SUITE")
    print("=" * 70)

    DB_PATH = Path("./data/lancedb")

    # Initialize engine
    print("\n🔧 Initializing query engine...")
    try:
        engine = CourseQueryEngine(
            db_path=DB_PATH,
            use_reranking=True
        )
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return False

    # Run test queries
    print(f"\n🔍 Running {len(TEST_QUERIES)} test queries...")
    print("=" * 70)

    results_summary = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Query: '{query}'")

        try:
            result = engine.query(query, top_k=10, verbose=True)

            print(f"  Top Result: {result['results'][0]['video_title']}")
            print(f"  Score: {result['results'][0]['score']:.3f}")
            print(f"  Time: {result['results'][0]['start_timestamp']}")

            results_summary.append({
                'query': query,
                'num_results': result['total_results'],
                'top_result': result['results'][0]['video_title'] if result['results'] else "None",
                'top_score': result['results'][0]['score'] if result['results'] else 0,
                'query_time': result['query_time']
            })

        except Exception as e:
            print(f"❌ Query failed: {e}")
            results_summary.append({
                'query': query,
                'error': str(e)
            })

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    successful = len([r for r in results_summary if 'error' not in r])
    avg_time = sum(r.get('query_time', 0) for r in results_summary) / successful if successful > 0 else 0

    print(f"\n✅ Successful queries: {successful}/{len(TEST_QUERIES)}")
    print(f"⏱️  Average query time: {avg_time:.2f}s")

    # Save results
    output_file = Path("./docs/PHASE4_QUERY_RESULTS.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n📄 Results saved to {output_file}")

    return successful == len(TEST_QUERIES)

if __name__ == "__main__":
    test_query_engine()