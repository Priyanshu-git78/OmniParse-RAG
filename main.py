from retrival_methods import main_retrival_pipeline
import streamlit as st
from pathlib import Path


# Title of the APP
st.title("This is RAG Pipeline Build for Demo Purpose")


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

query=st.chat_input("Enter Your Query")
if query:
    st.write("your query in process")
response=main_retrival_pipeline(query)
st.title("working:")
st.write(f"Answer for your query : {response}")