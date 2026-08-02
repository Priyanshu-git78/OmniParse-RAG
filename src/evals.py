from langsmith import traceable
from src.retrival_methods import retrival_pipeline

import re


class Rag_eval():
    



@traceable(name="rag_pipeline")
def main_retrival_pipelines(inputs: dict) -> dict:
    query = inputs["question"]
    collection = inputs.get("collection", "DemoRAG")
    pipeline = retrival_pipeline(collection=collection)
    all_retrieval_results = pipeline.multiquery_RRM(query)
    fused_results = pipeline.reciprocal_rank_fusion(
        all_retrieval_results, k=60, verbose=False
    )
    reranked_docs_c = pipeline.reranker_chunks()
    response = pipeline.generate_final_answer(chunks=reranked_docs_c, query=query)

    response_str = str(response)
    thinking_match = re.search(r"<think>(.*?)</think>", response_str, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else None
    clean_answer = re.sub(r"<think>.*?</think>", "", response_str, flags=re.DOTALL).strip()
    contexts= [getattr(d, "page_content", str(d)) for d in reranked_docs_c]
    print(f"contexts:{contexts}")
    print(f"type:{type(contexts)}")
    print(f"thinking: {thinking}")
    print(f"answer:{clean_answer}")
    return {
        "answer": clean_answer,
        "context": [getattr(d, "page_content", str(d)) for d in reranked_docs_c],
        "thinking": thinking,  from langsmith.evaluation import evaluate

import os
from langsmith import Client
from dotenv import load_dotenv
load_dotenv()
client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

try:   
    dataset = client.create_dataset(
        dataset_name="Rag-pipeline",
        description="Golden QA pairs for RAG evaluation"
    )
    examples = [
    {"question": "What is the refund policy?", "answer": "Refunds are issued within 14 days of purchase."},
    {"question": "How do I reset my password?", "answer": "Go to Settings > Security > Reset Password."},
    # ... add more, ideally pulled from real user logs / support tickets
    ]

    client.create_examples(
        inputs=[{"question": e["question"]} for e in examples],
        outputs=[{"answer": e["answer"]} for e in examples],
        dataset_id=dataset.id,
    )
except Exception as e:
    dataset = client.read_dataset(dataset_name="rag-eval-golden-setsing")




