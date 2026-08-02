from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline,ChatHuggingFace
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import streamlit as st
from langchain.chat_models import init_chat_model
import os
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import torch
from functools import lru_cache
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
load_dotenv()

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


# @lru_cache(maxsize=1)
# def _load_local_llm():
#     """only called the first time the local model is actually needed."""
#     model_name= "Qwen/Qwen2.5-3B-Instruct"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model= AutoModelForCausalLM.from_pretrained(
#         model_name,
#         dtype= "auto",
#         device_map="auto",
#     )
#     hf_pipeline= pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=512
#     )

#     return ChatHuggingFace(llm= HuggingFacePipeline(pipeline= hf_pipeline))

# def _local_llm_runnable(input_,**kwargs):
#     """Wrapper downloded only happens on actual invocation""" 
#     llm = _load_local_llm()
#     return llm.invoke(input_, **kwargs)


def build_llms():
    """Call this exactly once when your app starts."""
    llm_grok = init_chat_model(
        model="qwen/qwen3.6-27b",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.environ["GROQ_API_KEY"],
        model_provider="openai",
        temperature=0.0,
    )
    llm_open_router = init_chat_model(
        model="qwen/qwen3.6-27b",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.environ["openrouter_api_key".upper()],
        model_provider="openai",
        temperature=0.0,
    )
    vllm = init_chat_model(
        model="Qwen/Qwen2-VL-7B-Instruct-AWQ",
        openai_api_base="http://localhost:8005/v1",
        openai_api_key="pranshu123",
        model_provider="openai",
        temperature=0.0,
    )
    

    # local_llm_lazy = RunnableLambda(_local_llm_runnable)

    return llm_grok, llm_open_router, vllm


def build_structured_llm(llm_grok, llm_open_router, local_vllm,schema, method=None):
    kwargs = {"method": method} if method else {}
    return llm_grok.with_structured_output(schema, **kwargs).with_fallbacks(
        [llm_open_router.with_structured_output(schema, **kwargs),local_vllm.with_structured_output(schema, **kwargs)]
    )