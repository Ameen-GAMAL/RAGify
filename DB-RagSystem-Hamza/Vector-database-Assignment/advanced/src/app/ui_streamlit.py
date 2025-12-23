"""
Modern Streamlit UI for E-commerce Q&A RAG System

- Beautiful, responsive design
- Retrieval + LLM generation via pipeline.run() (keeps terminal logs)
- Feedback collection + self-learning boosts
- Analytics dashboard
- UI controls: Temperature + Hallucination Control
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import html

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.pipeline import RAGPipeline, RAGConfig
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

    .result-card {
        background: #f8f9fa;
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

    .result-card p {
        color: #1f2937 !important;
        font-size: 15px;
        line-height: 1.6;
        margin: 0.5rem 0;
    }

    .result-card strong {
        color: #111827 !important;
        font-weight: 600;
    }

    /* Main container */
    .main {
        padding: 2rem;
    }

    /* Headers */
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }

    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 2rem;
    }

    h3 {
        color: #34495e;
        font-weight: 500;
    }

    /* Search box */
    .stTextInput > div > div > input {
        font-size: 16px;
        padding: 12px;
    }

    /* Buttons */
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

    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
        color: #111827;
    }

    .css-1d391kg h2, .css-1d391kg h3 {
        color: #1f2937;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #7f8c8d;
        border-top: 1px solid #ecf0f1;
        margin-top: 3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline() -> RAGPipeline:
    """Load RAG pipeline (cached)."""
    return RAGPipeline()


def init_session_state():
    """Initialize session state variables."""
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False


def _sanitize_for_html(text: str) -> str:
    """
    Prevent:
    - unclosed markdown fences eating closing HTML tags
    - model output injecting HTML into our template
    """
    safe = text.replace("```", "\\`\\`\\`")
    safe = html.escape(safe, quote=False)
    return safe


def display_answer(answer: str):
    """Display answer in a styled box (HTML-safe)."""
    safe = _sanitize_for_html(answer)
    st.markdown(
        f"""
    <div class="answer-box">
        <h3 style="color: #1f2937; margin-top: 0;">💬 Answer</h3>
        <div style="
            margin: 0;
            color: #111827;
            font-size: 16px;
            line-height: 1.7;
        ">
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
        <h4>
            [{index}] Similar Q&A (Score: {score:.3f})
        </h4>
        <p><strong>Q:</strong> {question_display}</p>
        <p><strong>A:</strong> {answer_display}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def submit_feedback(result: dict, rating: int, comment: str):
    """Submit user feedback + update boosts."""
    paths = FeedbackPaths()
    chunk_ids = [s["chunk_id"] for s in result.get("sources", []) if "chunk_id" in s]

    record = make_feedback_record(
        query=result["query"],
        answer=result["answer"],
        sources=result.get("sources", []),
        rating=rating,
        comment=comment,
    )

    append_feedback(paths, record)
    update_boosts(paths, used_chunk_ids=chunk_ids, rating=rating)

    st.session_state.feedback_submitted = True


def display_feedback_form(result: dict):
    """Display feedback collection form."""
    st.markdown("---")
    st.subheader(" Was this answer helpful?")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("👍 Yes, helpful"):
            submit_feedback(result, rating=1, comment="Helpful")

    with col2:
        if st.button("👎 Not helpful"):
            submit_feedback(result, rating=-1, comment="Not helpful")

    with col3:
        if st.session_state.feedback_submitted:
            st.success("✅ Thank you for your feedback!")


def display_analytics():
    """Display analytics dashboard in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Analytics")

    paths = FeedbackPaths()
    stats = get_feedback_stats(paths)

    total = stats.get("total_feedback", 0)
    positive_rate = stats.get("positive_rate", 0)

    st.sidebar.metric("Total Feedback", f"{total:,}")
    st.sidebar.metric("Positive Rate", f"{positive_rate:.1f}%")

    if total > 0:
        st.sidebar.progress(positive_rate / 100)

        st.sidebar.markdown("#### 🏆 Top Performing")
        top_chunks = get_top_performing_chunks(paths, top_k=5)

        for i, chunk in enumerate(top_chunks, 1):
            boost = chunk["boost_score"]
            emoji = "🔥" if boost > 0.2 else "⭐" if boost > 0 else "📉"
            st.sidebar.text(f"{emoji} Chunk {i}: {boost:+.3f}")


def main():
    init_session_state()

    # Header
    st.title(" E-commerce Q&A Assistant")
    st.markdown(
        """
*Powered by RAG + MPNet Embeddings + Self-Learning*

Ask questions about products, compatibility, shipping, and more!
"""
    )

    # Sidebar: Settings
    st.sidebar.title(" Settings")
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
        value=0.7,
        step=0.1,
        help="Higher = more varied. 0.0 = deterministic.",
    )

    hallucination_control = st.sidebar.checkbox(
        "Hallucination control (strict grounding)",
        value=True,
        help="If enabled, the LLM must only use retrieved references; if insufficient, it must say so.",
    )

    show_sources = st.sidebar.checkbox(
        "Show source Q&A pairs",
        value=True,
        help="Display the similar Q&A pairs used to generate the answer",
    )

    # Sidebar analytics
    display_analytics()

    # Main search interface
    st.markdown("---")

    query = st.text_input(
        "🔍 Ask your question:",
        placeholder="e.g., What is the battery health life expectancy?",
        help="Type your question about products",
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
                pipeline = load_pipeline()

                # Use pipeline.run() so terminal prints show again.
                cfg = RAGConfig(
                    top_k=top_k,
                    use_generation=True,
                    temperature=float(temperature),
                    hallucination_control=bool(hallucination_control),
                )

                result = pipeline.run(query, top_k=top_k, config=cfg)

                st.session_state.last_result = result
                st.session_state.feedback_submitted = False

                st.session_state.search_history.append(
                    {"query": query, "timestamp": datetime.now().isoformat()}
                )

            except Exception as e:
                st.error(f" Error: {str(e)}")
                st.exception(e)

    # Display results
    if st.session_state.last_result:
        result = st.session_state.last_result

        st.markdown("---")
        display_answer(result["answer"])

        if show_sources and result.get("sources"):
            st.markdown("### Source Q&A Pairs")
            st.caption(f"Retrieved {len(result['sources'])} similar question-answer pairs")

            for i, source in enumerate(result["sources"], 1):
                display_source(source, i)

        display_feedback_form(result)

    # Recent searches
    if st.session_state.search_history:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🕒 Recent Searches")

        for i, item in enumerate(reversed(st.session_state.search_history[-5:]), 1):
            qtxt = item["query"][:40] + "..." if len(item["query"]) > 40 else item["query"]
            st.sidebar.text(f"{i}. {qtxt}")

    # Footer
    st.markdown(
        """
<div class="footer">
    <p>Built with ❤️ using • RAG • Sentence Transformers</p>
    <p style="font-size: 12px;">Model: sentence-transformers/all-mpnet-base-v2</p>
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
