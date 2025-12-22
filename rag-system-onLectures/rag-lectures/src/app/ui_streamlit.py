import os
import requests
import streamlit as st

# Allow overriding the API URL without editing code
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Lectures Chat", layout="wide")
st.title("RAG Lectures Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about the lectures...")  # Streamlit chat input UI


def render_sources(sources):
    with st.expander("Sources"):
        if not sources:
            st.write("No sources returned.")
            return

        for i, s in enumerate(sources, start=1):
            chunk_id = s.get("chunk_id", "N/A")
            lecture_id = s.get("lecture_id", "N/A")
            p_start = s.get("page_start", "?")
            p_end = s.get("page_end", "?")
            snippet = s.get("snippet", "")
            score = s.get("score", None)

            score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"

            st.markdown(
                f"**S{i}** — `{chunk_id}` | {lecture_id} p.{p_start}-{p_end} | score={score_txt}\n\n"
                f"> {snippet}"
            )


def send_feedback(query, answer, sources, rating, comment=""):
    payload = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "rating": rating,
        "comment": comment,
    }
    try:
        requests.post(f"{API_URL}/feedback", json=payload, timeout=10)
    except Exception:
        # Don't crash the UI if feedback fails
        pass


if prompt:
    # 1) Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Call API and show assistant response (or errors) inside assistant bubble
    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            try:
                r = requests.post(
                    f"{API_URL}/query",
                    json={"query": prompt, "top_k": 6},
                    timeout=120,
                )
            except requests.exceptions.RequestException as e:
                err_msg = (
                    "Could not reach the API server.\n\n"
                    f"- Check that FastAPI is running on `{API_URL}`\n"
                    f"- Error: `{e}`"
                )
                st.error("API connection error")
                st.markdown(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
                st.stop()

        # If API returned non-2xx, show body (often includes traceback or JSON error)
        if not r.ok:
            st.error(f"API error {r.status_code}")
            st.code(r.text)

            # Store something readable in chat history
            assistant_text = f"API error {r.status_code}\n\n{r.text}"
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            st.stop()

        # Try JSON parse
        try:
            resp = r.json()
        except Exception:
            st.error("API did not return JSON.")
            st.code(r.text)

            assistant_text = "API did not return JSON.\n\n" + r.text
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            st.stop()

        # If API returned JSON error object
        if isinstance(resp, dict) and resp.get("error"):
            st.error("Server exception")
            st.code(resp.get("traceback", resp["error"]))

            assistant_text = f"Server exception:\n\n{resp.get('traceback', resp['error'])}"
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            st.stop()

        answer = resp.get("answer", "")
        sources = resp.get("sources", [])

        st.markdown(answer)
        render_sources(sources)

        col1, col2, col3 = st.columns([1, 1, 6])
        with col1:
            if st.button("👍 Helpful"):
                send_feedback(prompt, answer, sources, rating=1)
                st.success("Thanks! Logged.")
        with col2:
            if st.button("👎 Not helpful"):
                send_feedback(prompt, answer, sources, rating=-1)
                st.info("Logged. We'll improve retrieval next time.")

    # 3) Store assistant answer in history ONLY if we successfully produced one
    st.session_state.messages.append({"role": "assistant", "content": answer})
