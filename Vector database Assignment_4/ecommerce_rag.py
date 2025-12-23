import streamlit as st
from models import EcommerceRAGSystem
from ui_components import get_css_styles, render_header, render_query_section, display_results, render_system_controls
from utils import initialize_session_state, handle_search_action, handle_clear_action, check_csv_file

def main():
    st.set_page_config(page_title="E-commerce FAQ Assistant", layout="wide")
    st.markdown(get_css_styles(), unsafe_allow_html=True)
    initialize_session_state()
    render_header()
    
    csv_exists = check_csv_file()
    
    if st.session_state.rag_system is None:
        if csv_exists:
            st.session_state.rag_system = EcommerceRAGSystem("Ecommerce_FAQs.csv")
            st.success("System initialized with CSV data!")
        else:
            st.session_state.rag_system = EcommerceRAGSystem()
            st.info("System initialized with default data.")
    
    rag = st.session_state.rag_system
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query, search_clicked, clear_clicked = render_query_section(rag)
        if search_clicked:
            handle_search_action(query, rag)
        if clear_clicked:
            handle_clear_action()
        display_results(query, st.session_state.chat_history, rag)
    
    with col2:
        render_system_controls(rag)

if __name__ == "__main__":
    main()