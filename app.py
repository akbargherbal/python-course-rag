"""
Streamlit web interface for course semantic search.
Run with: streamlit run app.py
"""
import streamlit as st
from pathlib import Path
from src.query_engine import CourseQueryEngine
import time

# Page configuration
st.set_page_config(
    page_title="🎓 Python Course Search",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .timestamp-badge {
        background-color: #f0f2f6;
        color: #31333F;
        padding: 4px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-weight: bold;
        border: 1px solid #d6d6d8;
    }
    .score-badge {
        color: #0068c9;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_query_engine():
    """Load query engine (cached to prevent reloading on every search)."""
    db_path = Path("./data/lancedb")
    
    if not db_path.exists():
        st.error(f"❌ Database not found at {db_path}")
        st.stop()
        
    return CourseQueryEngine(db_path=db_path, use_reranking=True)

def main():
    # Header
    st.title("🎓 Python Course Semantic Search")
    st.markdown("Search your local video course content with natural language.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        num_results = st.slider(
            "Results to show", 
            min_value=1, 
            max_value=20, 
            value=5
        )
        
        st.divider()
        
        st.subheader("ℹ️ About")
        st.markdown("""
        **Tech Stack:**
        - **Embeddings:** Snowflake Arctic (m)
        - **Reranker:** MS-MARCO MiniLM
        - **DB:** LanceDB
        - **UI:** Streamlit
        """)

    # Load Engine
    with st.spinner("🚀 Loading search engine..."):
        try:
            engine = load_query_engine()
        except Exception as e:
            st.error(f"Failed to load engine: {e}")
            st.stop()

    # Search Interface
    query = st.text_input(
        "Search Query", 
        placeholder="e.g., How do I use decorators?",
        label_visibility="collapsed"
    )

    # Example Queries
    if not query:
        st.markdown("### 📌 Try these examples:")
        examples = [
            "How do I raise and handle exceptions?",
            "Explain list comprehensions",
            "What is the difference between args and kwargs?",
            "How does memory management work?",
            "What are generators?"
        ]
        
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex):
                query = ex
                st.rerun()

    # Execute Search
    if query:
        start_time = time.time()
        try:
            response = engine.query(query, top_k=num_results)
            elapsed = time.time() - start_time
            
            # Results Header
            st.markdown(f"### Found {len(response['results'])} results in `{elapsed:.2f}s`")
            
            # Display Results
            for res in response['results']:
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                            <h3 style="margin:0; color:#0f1116;">{res['rank']}. {res['video_title']}</h3>
                            <span class="score-badge">Score: {res['score']:.2f}</span>
                        </div>
                        <div style="margin-bottom:10px; color:#555;">
                            <b>📂 Section:</b> {res['section']}
                        </div>
                        <div style="margin-bottom:15px;">
                            <span class="timestamp-badge">⏱️ {res['start_timestamp']} - {res['end_timestamp']}</span>
                            <span style="margin-left:10px; font-family:monospace; color:#666;">ID: {res['video_id']}</span>
                        </div>
                        <div style="background-color:#f8f9fa; padding:10px; border-left:3px solid #0068c9; font-style:italic; color:#444;">
                            "{res['text_snippet']}..."
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Search failed: {e}")

if __name__ == "__main__":
    main()