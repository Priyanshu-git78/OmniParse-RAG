from datasets import Dataset
from retrival_methods import retrival_pipeline 
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from models import build_llms
from ragas.run_config import RunConfig
from ragas.llms import llm_factory
from models import get_embedding_model
from openai import OpenAI


#making the llm_with_fallback
llm_grok, llm_openrouter, lazy_llm=build_llms()

client = OpenAI(api_key="pranshu123", base_url="http://localhost:8005/v1")
evaluator_llm = llm_factory("Qwen/Qwen3-14B-AWQ", client=client)
run_config = RunConfig(timeout=180, max_retries=5,max_wait=60,max_workers=10)



def main_retrival_pipeline(query: str,collection="DemoRAG"):
    
    
    pipeline = retrival_pipeline(collection=collection)
    all_retrieval_results = pipeline.multiquery_RRM(query)

    fused_results = pipeline.reciprocal_rank_fusion(
        all_retrieval_results, k=60, verbose=False
    )
    
    reranked_docs_c = pipeline.reranker_chunks()
    
    response = pipeline.generate_final_answer(chunks=reranked_docs_c, query=query)
    
    return response,reranked_docs_c
questions = [
    "What is Easy Build?",
]

ground_truths = [
    "A B2B2C online channel platform.",
]
rows =[]
for question, ground_truth in zip(questions, ground_truths):
    answer,contexts= main_retrival_pipeline(query=question)
    rows.append({
        "question":question,
        "contexts":[docs.page_content for docs in contexts],
        "answer": answer,
        "reference":ground_truth,
    }
    )


evaluation_dataset= Dataset.from_list(rows)

evaluator_embeddings = RagasHuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5",device="cpu")

from ragas import evaluate
from ragas.metrics.collections import (
    AnswerCorrectness,
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy
)

import json
answer_relevancy_1 = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    strictness=1,
)
answer_correctness = AnswerCorrectness(llm=evaluator_llm)

faithfulness = Faithfulness()
context_precision = ContextPrecision()
context_recall = ContextRecall()


scores = evaluate(
    evaluation_dataset,
    metrics=[
        answer_correctness,
        answer_relevancy_1,
        faithfulness,
        context_precision,
        context_recall,
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=run_config
)

aggregate_scores = dict(scores)

scores_df = scores.to_pandas()
print(scores_df)
per_row_scores = scores_df.to_dict(orient="records")

merged =[]

for row,score_row  in zip(rows,per_row_scores):
    merged.append({**row,**score_row})

output = {
    "aggregate_scores":aggregate_scores,
    "results":merged,
}

with open("eval_results.json",'w') as f :
    json.dump(output,f,indent=2,ensure_ascii=False )
print(f"Saved {len(merged)} results to eval_resuls.json")