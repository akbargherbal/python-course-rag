# 🎓 Python Course Semantic Search

A local RAG (Retrieval Augmented Generation) system for searching 100+ hours of Python video course content.

## 🚀 Quick Start

1.  **Activate Environment**
    ```bash
    source venv/bin/activate
    ```

2.  **Run Web Interface**
    ```bash
    streamlit run app.py
    ```

3.  **Run CLI**
    ```bash
    python -m src.cli search -q "your query"
    ```

## 🛠️ Architecture

-   **Embeddings**: `Snowflake/snowflake-arctic-embed-m` (384-dim)
-   **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
-   **Database**: LanceDB (Local vector store)
-   **Framework**: LlamaIndex

## 📂 Project Structure

-   `src/`: Core logic (parser, indexing, query engine)
-   `data/lancedb/`: The vector database files
-   `app.py`: Streamlit web application
-   `course_content/`: The raw SRT files

## 🔄 Updating the Index

If you add new SRT files to `course_content/`, rebuild the index:

```bash
python -m src.build_index
```
```

### Step 6.3: Final Verification

Run the validation suite to confirm everything is green.

```bash
python tests/validation.py
```

If that passes, **congratulations!** You have successfully built, optimized, and deployed a local semantic search engine for your course. 🚀