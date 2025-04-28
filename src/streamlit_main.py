import os, sys
# 1) put project root on path so "src." imports work
parent = os.path.dirname(os.path.abspath(__file__))
project = os.path.dirname(parent)
sys.path.append(project)

# 2) patch asyncio so nested runs don’t crash
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from src.streamlit.query_streamlit import set_streamlit_page_config, run_streamlit_app
from src.main import initialize_components

set_streamlit_page_config()

@st.cache_resource
def get_components():
    st.info("Initializing RAG components…")
    comps = initialize_components(llm_model_name="deepseek-r1:7b")
    st.success("Component initialization ready")
    return comps

comps = get_components()
if comps:
    run_streamlit_app(comps)
else:
    st.error("Component initialization failed")
