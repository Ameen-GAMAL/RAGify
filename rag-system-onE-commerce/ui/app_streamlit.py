import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so `from src...` works when running Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase4_generation import GenerationModule, GenerationConfig, ProductRecord  # Phase 4 generator


# -----------------------------
# Utilities
# -----------------------------

def _trim(text: str, limit: int = 240) -> str:
    text = "" if text is None else str(text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _ensure_outputs_dirs() -> Path:
    out_dir = PROJECT_ROOT / "outputs" / "ui"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _feedback_path() -> Path:
    out_dir = _ensure_outputs_dirs()
    return out_dir / "feedback.jsonl"


def _log_feedback(payload: Dict[str, Any]) -> None:
    path = _feedback_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# -----------------------------
# Cached loader (fast UI)
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_generator(
    top_k: int,
    retrieval_method: str,
    rerank: bool,
    strict_grounding: bool,
    llm_backend: str,
    openai_model: str,
    hf_model: str,
    temperature: float,
) -> GenerationModule:
    """
    Streamlit caches this resource so model/vector db loading doesn't repeat each run.
    """
    cfg = GenerationConfig(
        top_k=top_k,
        retrieval_method=retrieval_method,
        rerank=rerank,
        strict_grounding=strict_grounding,
        llm_backend=llm_backend,
        openai_model=openai_model,
        hf_model=hf_model,
        temperature=temperature,
    )
    return GenerationModule(config=cfg)


# -----------------------------
# UI
# -----------------------------

st.set_page_config(
    page_title="RAG Shopping Assistant (Phase 5)",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 RAG Shopping Assistant")
st.caption(
    "Phase 5 UI: query input, result display, and interactive controls "
    "(retrieval method, top-k, rerank, backend, grounding)."
)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")

    retrieval_method = st.selectbox(
        "Retrieval method",
        options=["hybrid", "semantic", "bm25"],
        index=0,
        help="Controls how results are retrieved before generation.",
    )

    top_k = st.slider(
        "Top-K results",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of retrieved products used to build the answer context.",
    )

    rerank = st.toggle(
        "Use re-ranking (cross-encoder)",
        value=False,
        help="Slower but can improve ranking quality.",
    )

    strict_grounding = st.toggle(
        "Strict grounding (no hallucinations)",
        value=True,
        help="Forces the generator to only use retrieved content.",
    )

    st.divider()
    st.subheader("LLM Backend")

    llm_backend = st.selectbox(
        "Backend",
        options=["auto", "fallback", "hf", "openai"],
        index=0,
        help="auto: OpenAI if key exists else HF else fallback.",
    )

    openai_model = st.text_input(
        "OpenAI model",
        value="gpt-4.1-mini",
        help="Only used if backend is openai/auto with OPENAI_API_KEY set.",
    )

    hf_model = st.text_input(
        "HF model",
        value="google/flan-t5-base",
        help="Only used if backend is hf/auto when transformers is available.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Higher = more creative, lower = more deterministic.",
    )

    st.divider()
    show_sources = st.toggle("Show sources & raw retrieved products", value=True)

# Main input area
query = st.text_input(
    "Enter your shopping query (e.g., 'educational toys for toddlers')",
    value="",
)

colA, colB = st.columns([1, 1])
with colA:
    run_btn = st.button("🔍 Search & Generate", type="primary", use_container_width=True)
with colB:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.session_state.pop("last_result", None)
    st.rerun()

# Run pipeline
if run_btn:
    if not query.strip():
        st.warning("Please type a query first.")
    else:
        with st.spinner("Running retrieval + generation..."):
            gen = load_generator(
                top_k=top_k,
                retrieval_method=retrieval_method,
                rerank=rerank,
                strict_grounding=strict_grounding,
                llm_backend=llm_backend,
                openai_model=openai_model,
                hf_model=hf_model,
                temperature=temperature,
            )

            t0 = time.time()
            out = gen.answer(query.strip())
            out["ui_latency_sec"] = round(time.time() - t0, 4)
            out["ui_settings"] = {
                "retrieval_method": retrieval_method,
                "top_k": top_k,
                "rerank": rerank,
                "strict_grounding": strict_grounding,
                "llm_backend": llm_backend,
                "openai_model": openai_model,
                "hf_model": hf_model,
                "temperature": temperature,
            }

            st.session_state["last_result"] = out

# Display results (if any)
result: Optional[Dict[str, Any]] = st.session_state.get("last_result")

if result:
    st.subheader("✅ Generated Answer")
    st.write(result["response"])

    st.caption(
        f"Retrieval: {result['retrieval_method']} | "
        f"Top-K: {result['top_k']} | "
        f"Rerank: {result['rerank']} | "
        f"Latency: {result.get('latency_sec', 'N/A')}s (gen) / {result.get('ui_latency_sec','N/A')}s (ui)"
    )

    # Feedback buttons (stored for Phase 6 self-learning)
    st.divider()
    st.subheader("👍 Feedback (stored for Phase 6)")

    fcol1, fcol2, fcol3 = st.columns([1, 1, 2])

    with fcol1:
        if st.button("👍 Helpful"):
            _log_feedback({
                "query": result.get("query"),
                "label": "helpful",
                "timestamp": time.time(),
                "settings": result.get("ui_settings", {}),
                "sources": result.get("sources", []),
            })
            st.success("Thanks! Feedback saved.")

    with fcol2:
        if st.button("👎 Not helpful"):
            _log_feedback({
                "query": result.get("query"),
                "label": "not_helpful",
                "timestamp": time.time(),
                "settings": result.get("ui_settings", {}),
                "sources": result.get("sources", []),
            })
            st.success("Thanks! Feedback saved.")

    with fcol3:
        comment = st.text_input("Optional comment (why?)", value="")
        if st.button("💾 Save comment"):
            _log_feedback({
                "query": result.get("query"),
                "label": "comment",
                "comment": comment,
                "timestamp": time.time(),
                "settings": result.get("ui_settings", {}),
                "sources": result.get("sources", []),
            })
            st.success("Comment saved.")

    # Sources + Product cards
    if show_sources:
        st.divider()
        st.subheader("📦 Retrieved Products (Sources)")

        # For display, load processed_data and show full info for each doc_id
        df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "processed_data.csv")

        sources: List[Dict[str, Any]] = result.get("sources", [])
        if not sources:
            st.info("No sources returned.")
        else:
            # Create product cards
            for s in sources:
                source_id = s.get("source_id")
                doc_id = s.get("doc_id")
                if doc_id is None or doc_id < 0 or doc_id >= len(df):
                    continue

                row = df.iloc[int(doc_id)]
                name = str(row.get("productName", "")).strip()
                desc = str(row.get("description", "")).strip()
                details = str(row.get("productDetails", "")).strip()
                price = str(row.get("price", "")).strip()
                was_price = str(row.get("withoutDiscountPrice", "")).strip()
                merchant = str(row.get("merchantName", "")).strip()
                p_link = str(row.get("productLink", "")).strip()
                m_link = str(row.get("merchantLink", "")).strip()
                rating = str(row.get("reviewsScore", "")).strip()
                reviews = str(row.get("reviewsCount", "")).strip()

                with st.expander(f"[{source_id}] {name or '(missing name)'}", expanded=(source_id == 1)):
                    left, right = st.columns([2, 1])

                    with left:
                        st.markdown(f"**Description:** {_trim(desc, 450)}")
                        if details and details != "nan":
                            st.markdown(f"**Details:** {_trim(details, 450)}")

                    with right:
                        if price and price != "nan":
                            st.markdown(f"**Price:** {price}")
                        if was_price and was_price != "nan" and was_price != price:
                            st.markdown(f"**Was:** {was_price}")
                        if merchant and merchant != "nan":
                            st.markdown(f"**Merchant:** {merchant}")
                        if rating and rating != "nan":
                            st.markdown(f"**Rating:** {rating}")
                        if reviews and reviews != "nan":
                            st.markdown(f"**Reviews:** {reviews}")

                        if p_link and p_link != "nan":
                            st.link_button("Open product link", p_link)
                        elif m_link and m_link != "nan":
                            st.link_button("Open merchant link", m_link)

        # Raw JSON (helpful for debugging / screenshots in report)
        st.divider()
        st.subheader("🧾 Debug: Raw Output JSON")
        st.json(result)

else:
    st.info("Enter a query and click **Search & Generate** to see results.")
