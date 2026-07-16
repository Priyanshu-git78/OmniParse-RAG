from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import streamlit as st


@st.cache_resource
def reranker_model():
    model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base", model_kwargs={"device": "cpu"}
    )
    return model


@st.cache_resource
def get_embedding_model():
    model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return model
