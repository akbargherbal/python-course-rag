"""
Comprehensive validation suite to ensure system quality.
"""
from pathlib import Path
from src.query_engine import CourseQueryEngine
import json
import time
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def create_validation_queries() -> list[dict]:
    """
    Create test queries with known relevance criteria.
    """
    return [
        {
            "query": "How do I use list comprehensions?",
            "should_contain": ["comprehension", "list"],
            "description": "Syntax: List Comprehensions"
        },
        {
            "query": "Explain decorators with examples",
            "should_contain": ["decorator", "wrap", "function"],
            "description": "Concept: Decorators"
        },
        {
            "query": "What is the difference between args and kwargs?",
            "should_contain": ["args", "kwargs"],
            "description": "Syntax: *args and **kwargs"
        },
        {
            "query": "How do I handle exceptions?",
            "should_contain": ["exception", "try", "except", "raise"],
            "description": "Concept: Exception Handling"
        },
        {
            "query": "What are context managers?",
            "should_contain": ["context", "manager", "with", "enter"],
            "description": "Concept: Context Managers"
        }
    ]

def validate_result_quality(result: dict, test_case: dict) -> dict:
    """
    Validate if top result matches expected content keywords.
    """
    # Get text from top 3 results to be generous
    top_texts = " ".join([r['full_text'].lower() for r in result['results'][:3]])
    
    # Check matches
    matches = [term for term in test_case['should_contain'] if term.lower() in top_texts]
    match_rate = len(matches) / len(test_case['should_contain'])
    
    # Pass if at least 50% of keywords are found in top 3 results
    passed = match_rate >= 0.5

    return {
        'passed': passed,
        'match_rate': match_rate,
        'found_terms': matches,
        'missing_terms': [t for t in test_case['should_contain'] if t not in matches],
        'reason': f"Found {len(matches)}/{len(test_case['should_contain'])} keywords"
    }

def run_validation_suite():
    print("🧪 VALIDATION TEST SUITE")
    print("=" * 70)

    db_path = Path("./data/lancedb")
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return

    # Load engine
    engine = CourseQueryEngine(db_path=db_path, use_reranking=True)
    test_cases = create_validation_queries()
    results_log = []

    print(f"\nRunning {len(test_cases)} validation queries...\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {test_case['description']}")
        print(f"  Query: '{test_case['query']}'")

        start = time.time()
        result = engine.query(test_case['query'], top_k=5)
        elapsed = time.time() - start

        validation = validate_result_quality(result, test_case)
        
        status = "✅" if validation['passed'] else "⚠️"
        print(f"  {status} {validation['reason']}")
        print(f"     Time: {elapsed:.2f}s")
        
        if not validation['passed']:
            print(f"     Missing: {validation['missing_terms']}")

        results_log.append(validation)
        print()

    # Summary
    passed = sum(1 for r in results_log if r['passed'])
    print("=" * 70)
    print(f"PASSED: {passed}/{len(test_cases)} ({100*passed/len(test_cases):.0f}%)")
    print("=" * 70)

if __name__ == "__main__":
    run_validation_suite()