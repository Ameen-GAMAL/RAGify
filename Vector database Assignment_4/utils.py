import os
import streamlit as st
from datetime import datetime

def initialize_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'csv_loaded' not in st.session_state:
        st.session_state.csv_loaded = False

def handle_search_action(query, rag_system):
    if query:
        with st.spinner("Searching the FAQ database..."):
            retrieved_docs = rag_system.retrieve(query)
            response = rag_system.generate_response(query, retrieved_docs)
            
            st.session_state.chat_history.insert(0, {
                'query': query,
                'response': response,
                'retrieved_docs': retrieved_docs,
                'timestamp': datetime.now()
            })
        return True
    return False

def handle_clear_action():
    st.session_state.chat_history = []
    st.rerun()

def check_csv_file(csv_path="Ecommerce_FAQs.csv"):
    if not os.path.exists(csv_path):
        st.warning(f"CSV file '{csv_path}' not found. Using default data.")
        return False
    return True

def get_similarity_percentage(score):
    return max(0, 1 - score/10) * 100