from datasets import Dataset
from retrival_methods import retrival_pipeline 

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
    "what is Easy Build?",
    "what are goals of Easy Build?",
    "where is Easy Build HO located?",
]

ground_truths =[
    "B2B2C online channel platfrom",
    "Easy Build is capture the raw material market in Noida"
    "Noida Sector 16"
]

pipe =retrival_pipeline()

rows =[]
for question, ground_truth in zip(questions, ground_truths):
    answer,contexts= main_retrival_pipeline(question=question)
    rows.append({
        "question":question,
        "contexts":contexts,
        "answer": answer,
        "reference":ground_truth,
    }
    )

evaluation_dataset= Dataset.fromlist(rows)
