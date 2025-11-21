"""
Build LanceDB index from processed SRT documents.
"""
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
import sys
import os
import time
import shutil

# Add project root to path
sys.path.append(os.getcwd())

from src.srt_processor import process_all_srt_files

def build_index(
    course_root: Path,
    db_path: Path = Path("./data/lancedb"),
    embedding_model: str = "Snowflake/snowflake-arctic-embed-m", # UPDATED MODEL
    max_chars: int = 1000,
    limit_files: int = None
):
    """Build complete LanceDB index."""
    print("=" * 70)
    print("BUILDING LANCEDB INDEX")
    print("=" * 70)

    # 1. Configure Embedding Model
    print(f"\n📦 Configuring model: {embedding_model}")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embedding_model,
        trust_remote_code=True
    )

    # 2. Load Documents
    print(f"\n📂 Loading SRT files from {course_root}")
    start_time = time.time()
    documents = process_all_srt_files(
        course_root,
        max_chars=max_chars,
        limit=limit_files
    )
    print(f"✅ Loaded {len(documents)} chunks in {time.time() - start_time:.1f}s")

    if not documents:
        print("❌ No documents to index!")
        return None

    # 3. Setup LanceDB
    print(f"\n💾 Initializing LanceDB at {db_path}")
    
    # Clean up existing DB to ensure fresh build
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    vector_store = LanceDBVectorStore(
        uri=str(db_path),
        table_name="course_content",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Build Index
    print(f"\n🚀 Generating embeddings and building index...")
    embed_start = time.time()
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    embed_time = time.time() - embed_start
    print(f"\n✅ Index built in {embed_time:.1f}s")
    print(f"   Avg time per chunk: {embed_time/len(documents):.3f}s")

    return index

if __name__ == "__main__":
    # Configuration
    COURSE_ROOT = Path("./course_content")
    DB_PATH = Path("./data/lancedb")
    
    # Set limit=None for full build
    LIMIT = None 
    
    build_index(
        course_root=COURSE_ROOT,
        db_path=DB_PATH,
        limit_files=LIMIT
    )
