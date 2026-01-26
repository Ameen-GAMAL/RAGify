"""
Modern Streamlit UI for E-commerce Q&A RAG System
WITH CUSTOM INPUT MODE FEATURE

Features:
- Database Mode: Use pre-built index
- Custom Input Mode: User provides Q&A pairs dynamically
- Beautiful, responsive design
- Feedback collection + self-learning
- Analytics dashboard
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import html
import json
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.pipeline import RAGPipeline, RAGConfig
from src.rag.dynamic_index import DynamicIndexBuilder
from src.rag.retriever import RetrievedChunk
from src.rag.generator import generate_answer, GenerationConfig
from src.learn.feedback_store import (
    FeedbackPaths,
    make_feedback_record,
    append_feedback,
    update_boosts,
    get_feedback_stats,
    get_top_performing_chunks,
)

# Page configuration
st.set_page_config(
    page_title="E-commerce Q&A Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .answer-box {
        background: #e8f4f8;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2ecc71;
        margin: 1.5rem 0;
        font-size: 16px;
        line-height: 1.6;
    }

    .custom-mode-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }

    .result-card,
    .result-card p,
    .result-card strong {
    color: #111827 !important;  /* dark text */
    }

    .result-card p { 
    margin: 0.25rem 0; 
    }

    .result-card {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .result-card h4 {
        color: #1f77b4 !important;
        font-weight: 600;
        margin-top: 0;
    }

    .qa-input-box {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bae6fd;
        margin: 0.5rem 0;
    }

    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background-color: #145a8c;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline() -> RAGPipeline:
    """Load RAG pipeline (cached)."""
    return RAGPipeline()


@st.cache_resource
def get_dynamic_builder() -> DynamicIndexBuilder:
    """Get dynamic index builder (cached)."""
    return DynamicIndexBuilder()


def init_session_state():
    """Initialize session state variables."""
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    if "mode" not in st.session_state:
        st.session_state.mode = "database"  # 'database' or 'custom'
    if "custom_qa_pairs" not in st.session_state:
        st.session_state.custom_qa_pairs = []
    if "index_built" not in st.session_state:
        st.session_state.index_built = False
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False


def _sanitize_for_html(text: str) -> str:
    """Prevent HTML injection"""
    safe = text.replace("```", "\\`\\`\\`")
    safe = html.escape(safe, quote=False)
    return safe


def display_answer(answer: str):
    """Display answer in a styled box"""
    safe = _sanitize_for_html(answer)
    st.markdown(
        f"""
    <div class="answer-box">
        <h3 style="color: #1f2937; margin-top: 0;">💬 Answer</h3>
        <div style="margin: 0; color: #111827; font-size: 16px; line-height: 1.7;">
            {safe}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def display_source(source: dict, index: int):
    """Display a single retrieved Q&A in a styled card."""
    question = source.get("question", "N/A")
    answer = source.get("answer", "N/A")
    score = float(source.get("score", 0))

    question_display = question[:150] + "..." if len(question) > 150 else question
    answer_display = answer[:200] + "..." if len(answer) > 200 else answer

    question_display = _sanitize_for_html(question_display)
    answer_display = _sanitize_for_html(answer_display)

    st.markdown(
        f"""
    <div class="result-card">
        <h4>[{index}] Similar Q&A (Score: {score:.3f})</h4>
        <p><strong>Q:</strong> {question_display}</p>
        <p><strong>A:</strong> {answer_display}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def custom_input_mode():
    """UI for custom Q&A input mode"""
    st.markdown(
        """
    <div class="custom-mode-box">
        <h3 style="color: #856404; margin-top: 0;">📝 Custom Input Mode</h3>
        <p style="color: #856404; margin: 0;">
            Add your own Q&A pairs to create a temporary knowledge base
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    
    # Display current Q&A pairs
    st.subheader(f"📚 Your Q&A Pairs ({len(st.session_state.custom_qa_pairs)})")
    
    if st.session_state.custom_qa_pairs:
        for i, pair in enumerate(st.session_state.custom_qa_pairs, 1):
            with st.expander(f"Q&A #{i}: {pair['question'][:50]}..."):
                st.markdown(f"**Question:** {pair['question']}")
                st.markdown(f"**Answer:** {pair['answer']}")
                if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.custom_qa_pairs.pop(i-1)
                    st.session_state.index_built = False
                    st.success(f"✅ Removed Q&A #{i}")
                    # Force rebuild of this section
                    time.sleep(0.1)
                    st.rerun()
    else:
        st.info("No Q&A pairs added yet. Add your first pair below!")
    
    st.markdown("---")
    
    # Add new Q&A pair
    st.subheader("➕ Add New Q&A Pair")
    
    # Method 1: Manual Entry
    with st.expander("✍️ Manual Entry", expanded=True):
        question = st.text_input(
            "Question",
            placeholder="e.g., What is the battery life of this product?",
            key="new_question"
        )
        answer = st.text_area(
            "Answer",
            placeholder="e.g., The battery life is approximately 10-12 hours with normal use.",
            key="new_answer",
            height=100
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("➕ Add Pair", type="primary"):
                if question.strip() and answer.strip():
                    st.session_state.custom_qa_pairs.append({
                        'question': question.strip(),
                        'answer': answer.strip()
                    })
                    st.session_state.index_built = False
                    st.success("✅ Q&A pair added!")
                    # Clear the inputs by setting a flag
                    st.session_state.clear_inputs = True
                else:
                    st.error("Both question and answer are required!")
    
    # Method 2: JSON Upload
    with st.expander("📤 Upload JSON File"):
        st.markdown("""
        Upload a JSON file with this format:
        ```json
        {
          "qa_pairs": [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"}
          ]
        }
        ```
        """)
        
        uploaded_file = st.file_uploader(
            "Choose JSON file",
            type=['json'],
            key="json_uploader"
        )
        
        # Process file only once
        if uploaded_file is not None:
            # Create a unique identifier for this file
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            
            # Only process if it's a new file
            if st.session_state.last_uploaded_file != file_id:
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    data = json.load(uploaded_file)
                    pairs = data.get('qa_pairs', [])
                    
                    if pairs:
                        st.session_state.custom_qa_pairs.extend(pairs)
                        st.session_state.index_built = False
                        st.session_state.last_uploaded_file = file_id
                        st.success(f"✅ Added {len(pairs)} Q&A pairs from JSON!")
                        # Don't rerun - let the user see the success message
                    else:
                        st.error("No 'qa_pairs' found in JSON file")
                except Exception as e:
                    st.error(f"Error loading JSON: {e}")
            else:
                st.info(f"📁 File already loaded: {uploaded_file.name}")
    
    # Method 3: Paste Multiple Pairs
    with st.expander("📋 Paste Multiple Pairs (Tab-separated)"):
        st.markdown("Paste Q&A pairs in this format (one per line):")
        st.code("Question 1\\tAnswer 1\nQuestion 2\\tAnswer 2")
        
        bulk_text = st.text_area(
            "Paste here",
            placeholder="What is X?\tX is...\nHow to Y?\tTo Y, you...",
            height=150,
            key="bulk_qa"
        )
        
        if st.button("➕ Add Bulk Pairs"):
            if bulk_text.strip():
                lines = bulk_text.strip().split('\n')
                added = 0
                
                for line in lines:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        q = parts[0].strip()
                        a = parts[1].strip()
                        if q and a:
                            st.session_state.custom_qa_pairs.append({
                                'question': q,
                                'answer': a
                            })
                            added += 1
                
                if added > 0:
                    st.session_state.index_built = False
                    st.success(f"✅ Added {added} Q&A pairs!")
                else:
                    st.error("No valid pairs found. Check format.")
    
    st.markdown("---")
    
    # Build Index Button
    if st.session_state.custom_qa_pairs:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🔨 Build Search Index", type="primary", use_container_width=True):
                with st.spinner("Building index..."):
                    builder = get_dynamic_builder()
                    builder.clear()
                    builder.add_qa_pairs(st.session_state.custom_qa_pairs)
                    success = builder.build_index()
                    
                    if success:
                        st.session_state.index_built = True
                        st.success("✅ Index built! You can now search your Q&A pairs.")
                    else:
                        st.error("Failed to build index")
        
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                if st.session_state.custom_qa_pairs:  # Only clear if there are pairs
                    st.session_state.custom_qa_pairs = []
                    st.session_state.index_built = False
                    st.session_state.last_uploaded_file = None
                    get_dynamic_builder().clear()
                    st.success("✅ All Q&A pairs cleared")
                    time.sleep(0.5)
                    st.rerun()
        
        with col3:
            # Export functionality
            if st.session_state.custom_qa_pairs:
                export_data = {
                    'qa_pairs': st.session_state.custom_qa_pairs
                }
                st.download_button(
                    label="💾 Export",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"custom_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )


def search_custom_mode(query: str, top_k: int, config: RAGConfig):
    """Search in custom input mode"""
    if not st.session_state.index_built:
        st.error("⚠️ Please build the index first by clicking '🔨 Build Search Index'")
        return None
    
    builder = get_dynamic_builder()
    stats = builder.get_stats()
    
    if stats['total_vectors'] == 0:
        st.error("⚠️ No Q&A pairs in index. Please add some pairs first.")
        return None
    
    # Retrieve from custom index
    retrieved_dicts = builder.search(query, top_k=top_k)
    
    if not retrieved_dicts:
        return {
            "query": query,
            "answer": "No relevant results found in your custom Q&A pairs.",
            "sources": [],
            "metadata": {"used_llm": False}
        }
    
    # Convert to RetrievedChunk format
    retrieved = [
        RetrievedChunk(
            chunk_id=r['chunk_id'],
            score=r['score'],
            text=f"Question: {r['question']}\nAnswer: {r['answer']}",
            metadata=r['metadata']
        )
        for r in retrieved_dicts
    ]
    
    # Generate answer using LLM if enabled
    if config.use_generation:
        try:
            gen_config = GenerationConfig(
                temperature=float(config.temperature),
                hallucination_control=bool(config.hallucination_control),
            )
            result = generate_answer(query, retrieved, gen_config)
            answer = result["answer"]
            used_llm = True
        except Exception as e:
            st.warning(f"LLM generation failed: {e}. Using simple answer.")
            answer = f"{retrieved[0].metadata['answer']}\n\n*Source: {retrieved[0].metadata['question']}*"
            used_llm = False
    else:
        # Simple answer from best match
        answer = f"{retrieved[0].metadata['answer']}\n\n*Source: {retrieved[0].metadata['question']}*"
        used_llm = False
    
    # Format sources
    sources = [
        {
            "chunk_id": r['chunk_id'],
            "score": r['score'],
            "question": r['question'],
            "answer": r['answer'],
        }
        for r in retrieved_dicts
    ]
    
    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "metadata": {
            "total_retrieved": len(sources),
            "used_llm": used_llm,
            "mode": "custom",
        }
    }


def main():
    init_session_state()

    # Header
    st.title("🛍️ E-commerce Q&A Assistant")
    
    # Mode selector in sidebar
    st.sidebar.title("⚙️ Settings")
    
    mode = st.sidebar.radio(
        "🔀 Mode",
        ["📊 Database Mode", "📝 Custom Input Mode"],
        index=0 if st.session_state.mode == "database" else 1,
        help="Choose between pre-built database or custom input"
    )
    
    # Update mode
    if "Database" in mode:
        st.session_state.mode = "database"
    else:
        st.session_state.mode = "custom"
    
    st.markdown("---")
    
    # Show appropriate interface based on mode
    if st.session_state.mode == "custom":
        custom_input_mode()
        
        if not st.session_state.index_built:
            st.info("ℹ️ Add Q&A pairs above and build the index to start searching")
            return
    
    # Common search settings
    st.sidebar.markdown("### Search Configuration")
    
    top_k = st.sidebar.slider(
        "Number of results",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of similar Q&A pairs to retrieve",
    )
    
    st.sidebar.markdown("### Generation Configuration")
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.9,
        step=0.1,
        help="Higher = more varied. 0.0 = deterministic.",
    )
    
    hallucination_control = st.sidebar.checkbox(
        "Hallucination control (strict grounding)",
        value=True,
        help="If enabled, the LLM must only use retrieved references",
    )
    
    show_sources = st.sidebar.checkbox(
        "Show source Q&A pairs",
        value=True,
        help="Display the similar Q&A pairs used to generate the answer",
    )
    
    # Analytics (only in database mode)
    if st.session_state.mode == "database":
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Analytics")
        paths = FeedbackPaths()
        stats = get_feedback_stats(paths)
        total = stats.get("total_feedback", 0)
        positive_rate = stats.get("positive_rate", 0)
        st.sidebar.metric("Total Feedback", f"{total:,}")
        st.sidebar.metric("Positive Rate", f"{positive_rate:.1f}%")
    
    # Main search interface
    st.markdown("---")
    
    # Mode indicator
    if st.session_state.mode == "custom":
        builder = get_dynamic_builder()
        stats = builder.get_stats()
        st.info(f"🔍 Custom Mode Active | {stats['total_pairs']} Q&A pairs | Index: {'✅ Built' if stats['index_built'] else '⏳ Not Built'}")
    else:
        st.info("📊 Database Mode Active | Searching pre-built knowledge base")
    
    query = st.text_input(
        "🔍 Ask your question:",
        placeholder="e.g., What is the battery health life expectancy?",
        help="Type your question",
        key="search_query",
    )
    
    col1, col2, _ = st.columns([2, 1, 1])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.last_result = None
        st.session_state.feedback_submitted = False
        st.rerun()
    
    # Search action
    if search_button and query:
        with st.spinner("🔄 Searching..."):
            try:
                cfg = RAGConfig(
                    top_k=top_k,
                    use_generation=True,
                    temperature=float(temperature),
                    hallucination_control=bool(hallucination_control),
                )
                
                # Route based on mode
                if st.session_state.mode == "database":
                    pipeline = load_pipeline()
                    result = pipeline.run(query, top_k=top_k, config=cfg)
                else:
                    result = search_custom_mode(query, top_k, cfg)
                
                if result:
                    st.session_state.last_result = result
                    st.session_state.feedback_submitted = False
                    
                    st.session_state.search_history.append(
                        {"query": query, "timestamp": datetime.now().isoformat(), "mode": st.session_state.mode}
                    )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    # Display results
    if st.session_state.last_result:
        result = st.session_state.last_result
        
        st.markdown("---")
        display_answer(result["answer"])
        
        if show_sources and result.get("sources"):
            st.markdown("### 📚 Source Q&A Pairs")
            st.caption(f"Retrieved {len(result['sources'])} similar question-answer pairs")
            
            for i, source in enumerate(result["sources"], 1):
                display_source(source, i)
        
        # Feedback (only in database mode)
        if st.session_state.mode == "database":
            st.markdown("---")
            st.subheader("👍 Was this answer helpful?")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("👍 Yes, helpful"):
                    paths = FeedbackPaths()
                    chunk_ids = [s["chunk_id"] for s in result.get("sources", []) if "chunk_id" in s]
                    record = make_feedback_record(
                        query=result["query"],
                        answer=result["answer"],
                        sources=result.get("sources", []),
                        rating=1,
                        comment="Helpful",
                    )
                    append_feedback(paths, record)
                    update_boosts(paths, used_chunk_ids=chunk_ids, rating=1)
                    st.session_state.feedback_submitted = True
            
            with col2:
                if st.button("👎 Not helpful"):
                    paths = FeedbackPaths()
                    chunk_ids = [s["chunk_id"] for s in result.get("sources", []) if "chunk_id" in s]
                    record = make_feedback_record(
                        query=result["query"],
                        answer=result["answer"],
                        sources=result.get("sources", []),
                        rating=-1,
                        comment="Not helpful",
                    )
                    append_feedback(paths, record)
                    update_boosts(paths, used_chunk_ids=chunk_ids, rating=-1)
                    st.session_state.feedback_submitted = True
            
            with col3:
                if st.session_state.feedback_submitted:
                    st.success("✅ Thank you for your feedback!")
    
    # Recent searches
    if st.session_state.search_history:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🕒 Recent Searches")
        
        for i, item in enumerate(reversed(st.session_state.search_history[-5:]), 1):
            qtxt = item["query"][:40] + "..." if len(item["query"]) > 40 else item["query"]
            mode_icon = "📊" if item.get("mode") == "database" else "📝"
            st.sidebar.text(f"{mode_icon} {i}. {qtxt}")
    
    # Footer
    st.markdown(
        """
<div style="text-align: center; padding: 2rem 0; color: #7f8c8d; border-top: 1px solid #ecf0f1; margin-top: 3rem;">
    <p>Built with ❤️ using • RAG • Sentence Transformers</p>
    <p style="font-size: 12px;">Model: sentence-transformers/all-mpnet-base-v2</p>
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()