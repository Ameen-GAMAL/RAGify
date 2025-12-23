import streamlit as st
from datetime import datetime

def get_css_styles():
    return """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        color: #F24236;
        font-size: 1.5rem;
        margin-top: 20px;
    }
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #155724;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0c5460;
        margin: 10px 0;
    }
    </style>
    """

def render_header():
    st.markdown('<h1 class="main-header">E-commerce FAQ Assistant</h1>', unsafe_allow_html=True)

def render_query_section(rag_system):
    st.markdown('<h2 class="sub-header">Ask Questions</h2>', unsafe_allow_html=True)
    query = st.text_input(
        "Enter your question:",
        placeholder="Example: How can I track my order? or What is your return policy?",
        key="query_input"
    )
    col1, col2 = st.columns(2)
    with col1:
        search_clicked = st.button("Search for Answer", type="primary")
    with col2:
        clear_clicked = st.button("Clear Chat")
    return query, search_clicked, clear_clicked

def display_results(query, chat_history, rag_system):
    if query and chat_history and chat_history[0]['query'] == query:
        latest_chat = chat_history[0]
        st.markdown("---")
        st.markdown("### Results:")
        with st.container():
            st.markdown("**Your Question:**")
            st.info(latest_chat['query'])
            st.markdown("**Answer:**")
            st.success(latest_chat['response'])
            with st.expander("Search Details (Similar Questions)"):
                for i, doc in enumerate(latest_chat['retrieved_docs'][:3]):
                    similarity_score = 1 - doc['score']/10
                    st.markdown(f"**Question {i+1}** (similarity: {similarity_score:.2%}):")
                    st.markdown(f"*{doc['question']}*")
                    if doc['answer'] and len(doc['answer']) > 0:
                        st.markdown(f"**Original answer:** {doc['answer'][:150]}...")
                    st.markdown("---")
        st.markdown("### Rate this answer:")
        rating = st.slider("How helpful was this answer?", 1, 5, 3, key="rating_slider")
        if st.button("Submit Rating", key="submit_rating_btn"):
            rag_system.learning_layer.add_feedback(
                latest_chat['query'],
                latest_chat['retrieved_docs'],
                0,
                rating/5.0
            )
            st.markdown('<div class="success-box">Thank you for your feedback!</div>', unsafe_allow_html=True)
    if len(chat_history) > 1:
        st.markdown("### Previous Conversations:")
        for i, chat in enumerate(chat_history[1:6]):
            with st.expander(f"Question: {chat['query'][:50]}..." if len(chat['query']) > 50 else f"Question: {chat['query']}", key=f"exp_{i}"):
                st.markdown(f"**Question:** {chat['query']}")
                st.markdown(f"**Answer:** {chat['response'][:200]}...")

def render_system_controls(rag_system):
    st.markdown('<h2 class="sub-header">System Controls</h2>', unsafe_allow_html=True)
    st.markdown("### Add New Q&A Pair:")
    with st.form("add_qna_form"):
        new_question = st.text_input("New Question:", key="new_question_input")
        new_answer = st.text_area("Answer:", height=100, key="new_answer_input")
        submit_clicked = st.form_submit_button("Add to Database")
        if submit_clicked:
            if new_question and new_answer:
                if rag_system.add_new_qna(new_question, new_answer):
                    st.success("Added successfully!")
                else:
                    st.error("Error adding to database")
            else:
                st.warning("Please fill all fields")