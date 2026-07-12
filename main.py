from retrival_methods import main_retrival_pipeline
import streamlit as st
from pathlib import Path
import time
import streamlit as st
from textwrap import dedent
from ingestion_pipeline import ingestion_pipeline


mode= st.segmented_control(
    "choose Knowledge Source",
    ["Easy Build", "Upload Document"],
    default="Easy Build",width="stretch"
)
if mode=="Easy Build":
    st.set_page_config(
        page_title="Easy Build Industry RAG",
        page_icon="🤖",
        layout="wide"
    )

    st.markdown(
        """
        <div style="text-align:center; padding:20px 0;">
            <h1 style="color:#4F8BF9; margin-bottom:10px;">
                🤖 Industry-Grade RAG Pipeline Demo
            </h1>
            <h3 style="color:#666; font-weight:400;">
                Easy Build AI Knowledge Assistant
            </h3>
            <p style="font-size:18px; color:#888;">
                Ask questions about AI Easy Build documents indexed in the vector database.
                Responses are generated using Retrieval-Augmented Generation (RAG).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    query=st.chat_input("Enter Your Query")
    if query:
        st.write("your query in process")
        response=main_retrival_pipeline(query)
        time.sleep(5)
        st.title("working:")
        st.write(f"Answer for your query : {response}")
if mode == "Upload Document":
    st.markdown(dedent("""
    <div style="text-align:center; padding:2rem 1rem 2.5rem 1rem;">
        <h1 style="
            font-size:2.4rem;
            font-weight:800;
            background:linear-gradient(90deg, #6366f1, #ec4899);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            margin-bottom:0.5rem;
        ">
            🤖 Industry-Grade RAG Pipeline
        </h1>
        <p style="font-size:1.05rem; color:#6c757d; max-width:600px; margin:0 auto 1.2rem auto;">
            Upload a document and interact with it using an AI-powered RAG pipeline.
        </p>
        <div style="display:flex; justify-content:center; gap:0.5rem; flex-wrap:wrap; margin-bottom:1rem;">
            <span style="background:#eef2ff; color:#4338ca; padding:0.3rem 0.8rem; border-radius:999px; font-size:0.85rem; font-weight:600;">📄 PDF</span>
            <span style="background:#eef2ff; color:#4338ca; padding:0.3rem 0.8rem; border-radius:999px; font-size:0.85rem; font-weight:600;">📝 DOCX</span>
            <span style="background:#eef2ff; color:#4338ca; padding:0.3rem 0.8rem; border-radius:999px; font-size:0.85rem; font-weight:600;">📊 PPTX</span>
            <span style="background:#eef2ff; color:#4338ca; padding:0.3rem 0.8rem; border-radius:999px; font-size:0.85rem; font-weight:600;">📈 XLSX</span>
            <span style="background:#eef2ff; color:#4338ca; padding:0.3rem 0.8rem; border-radius:999px; font-size:0.85rem; font-weight:600;">📑 CSV</span>
        </div>
        <p style="font-size:0.9rem; color:#999;">
            Documents are indexed into the vector database after upload. Processing may take a few moments.
        </p>
    </div>
    """), unsafe_allow_html=True)

    st.divider()
    UPLOAD_DIR=Path("uploads")
    UPLOAD_DIR.mkdir(exist_ok=True)
    uploaded_file=st.file_uploader("Upload")
    if uploaded_file:
        
        file_path= UPLOAD_DIR/uploaded_file.name 
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        response = ingestion_pipeline(str(file_path))
        st.write(f"chunks:{response}")
    
    query=st.chat_input("Enter Your Query")
    if query:
        st.write("your query in process")
        response=main_retrival_pipeline(query)
        time.sleep(5)
        st.title("working:")
        st.write(f"Answer for your query : {response}")
    



#upload dir and logic 
# UPLOAD_DIR= Path("uploads")
# UPLOAD_DIR.mkdir(exist_ok=True)
# uploaded_file=st.file_uploader("upload your documents for here (optional)")
# if uploaded_file:
#     file_path= UPLOAD_DIR/uploaded_file.name
#     with open(file_path,"wb") as f:
#         f.write(uploaded_file.getbuffer())
#         st.success(f"Saved: {file_path}")

# uploaded Retrival to 

