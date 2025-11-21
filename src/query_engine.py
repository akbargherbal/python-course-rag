"""
Query engine with two-stage retrieval: vector search + cross-encoder reranking.
"""
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from typing import List, Dict, Optional
import time

# CONSTANTS
DEFAULT_DB_PATH = Path("./data/lancedb")
DEFAULT_EMBED_MODEL = "Snowflake/snowflake-arctic-embed-m"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class CourseQueryEngine:
    """Semantic search engine for course content with reranking."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        use_reranking: bool = True
    ):
        """
        Initialize query engine.
        """
        self.db_path = Path(db_path)
        self.use_reranking = use_reranking

        # 1. DISABLE LLM (We want local search, not generation)
        Settings.llm = None

        # 2. Configure embedding model
        print(f"⚙️  Configuring embedding model: {embedding_model}")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            trust_remote_code=True
        )

        # 3. Load existing index from LanceDB
        if not self.db_path.exists():
            raise ValueError(f"Database not found at {self.db_path}")

        print(f"📖 Loading index from {self.db_path}...")
        vector_store = LanceDBVectorStore(
            uri=str(self.db_path),
            table_name="course_content"
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        print("✅ Index loaded successfully")

        # 4. Setup reranker
        if self.use_reranking:
            print(f"📊 Loading reranker: {rerank_model}")
            self.reranker = SentenceTransformerRerank(
                model=rerank_model,
                top_n=5  # Keep top 5 after reranking
            )
        else:
            self.reranker = None

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        verbose: bool = False
    ) -> Dict:
        """
        Execute semantic search.
        """
        start_time = time.time()

        # Configure the engine
        # response_mode="no_text" ensures we just get nodes, no LLM generation
        if self.reranker and self.use_reranking:
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                node_postprocessors=[self.reranker],
                response_mode="no_text"
            )
        else:
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="no_text"
            )

        # Execute
        response = query_engine.query(query_text)
        query_time = time.time() - start_time

        # Format results
        results = []
        for i, node in enumerate(response.source_nodes, 1):
            meta = node.metadata
            
            # Handle potential missing keys safely
            start_seconds = int(meta.get('start_time_seconds', 0))
            end_seconds = int(meta.get('end_time_seconds', 0))
            
            start_ts = f"{start_seconds // 60}:{start_seconds % 60:02d}"
            end_ts = f"{end_seconds // 60}:{end_seconds % 60:02d}"

            results.append({
                'rank': i,
                'video_id': meta.get('video_id', 'N/A'),
                'video_title': meta.get('video_title', 'Unknown'),
                'section': meta.get('section', 'Unknown'),
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
                'score': float(node.score) if node.score else 0.0,
                'text_snippet': node.text[:300].replace('\n', ' '),
                'full_text': node.text
            })

        if verbose:
            print(f"Query time: {query_time:.2f}s")
            print(f"Results returned: {len(results)}")

        return {
            'query': query_text,
            'results': results,
            'query_time': query_time,
            'total_results': len(results),
            'reranking_used': self.use_reranking
        }

if __name__ == "__main__":
    # Quick sanity check
    print("🧪 Running quick sanity check...")
    try:
        engine = CourseQueryEngine()
        result = engine.query("What is a function?", top_k=3, verbose=True)
        print("\nTop Result:")
        if result['results']:
            r = result['results'][0]
            print(f"[{r['video_title']}] @ {r['start_timestamp']}")
            print(f"Snippet: {r['text_snippet']}...")
        else:
            print("No results found.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()