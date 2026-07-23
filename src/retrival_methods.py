from dotenv import load_dotenv

# import mlflow
import re
from models import get_embedding_model, reranker_model, build_llms ,build_structured_llm
import json
import time
from pydantic import BaseModel
from collections import defaultdict

# Load a local open-source cross-encoder model

# Lanchain dependencies
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings


from langchain.chat_models import init_chat_model
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st
import os
from langchain_postgres import PGVector

load_dotenv()

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment('Rag Prompts')
# mlflow.langchain.autolog()


class retrival_pipeline:

    def __init__(self,collection="DemoRAG"):
        #llm for fallbacking system
        self.llm_grok, self.llm_open_router, self.local_vllm = build_llms()
        self.llm = self.llm_grok.with_fallbacks([self.llm_open_router, self.local_vllm])


        self.original_query = "what is Easy Build revenue, profit and Sales"
        
        
        #initializing the embedding model
        self.embedding_model = get_embedding_model()

        self.db = PGVector(
            embeddings=self.embedding_model,
            collection_name=collection,
            connection = os.environ["postgres_url"],
            use_jsonb= True
        )
        from sqlalchemy import text

        with self.db._engine.connect() as conn:
            result = conn.execute(text("""
            SELECT document,cmetadata
            FROM langchain_pg_embedding
            WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection WHERE name=:collection)
            """),{'collection':collection}
            )
            rows = result.fetchall()
        self.documents = [
            Document(page_content= row[0],metadata=row[1] or {})
            for row in rows
        ]
            

        # self.raw = self.db.get(include=["documents", "metadatas"])
        # self.documents = []
        # for text, meta in zip(self.raw["documents"], self.raw["metadatas"]):
        #     meta = meta or {}
        #     for k, v in meta.items():
        #         if isinstance(v, str):
        #             try:
        #                 meta[k] = json.loads(v)
        #             except json.JSONDecodeError:
        #                 pass
        #     self.documents.append(Document(page_content=text, metadata=meta))

        # number of the chunks
        self.k = 5  # number of the chunks from hybrid retrival and mmr
        self.reranker_k_no = 3  # number of the chunks from reranker

        # intialize the reranker model from transformers
        self.cross_encoder = reranker_model()
    
    
    def get_structured_llm(self,schema, method=None):
        return build_structured_llm(self.llm_grok,self.llm_open_router,self.local_vllm,schema,method=method)

    
    def hybrid_retriver(self):
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = self.k
        vector_retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": 20, "lambda_mult": 0.5},
        )

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever], weights=[0.7, 0.3]
        )
        print("-----Hybrid Retriever intialised")
        return self.hybrid_retriever

    def multiquery_RRM(self, original_query):

        if original_query:
            self.original_query = original_query

        class Queryvariations(BaseModel):
            queries: list[str]

        llm_with_tool = self.get_structured_llm(
            Queryvariations, method="json_mode"
        )

        prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

        Original query: {self.original_query}

        Return 3 alternative queries that rephrase or approach the same question from different angles.
        Respond ONLY with a valid JSON object matching this format, no extra text:
        {{"queries": ["variation 1", "variation 2", "variation 3"]}}
        """

        response = llm_with_tool.invoke(prompt)

        query_variations = response.queries

        retriever = self.hybrid_retriver()

        self.all_retrieval_results = []

        for i, query in enumerate(query_variations, 1):
            print(f"\n=====Results for Query {i}:{query}===")
            docs = retriever.invoke(query)
            self.all_retrieval_results.append(docs)  # Store for RRF calculation
            print(f"Retrived {len(docs)} documents:\n")
            for j, doc in enumerate(docs, 1):
                print(f"Document{j}:")
                print(f"{doc.page_content[:150]}")
            print("-" * 50)
        print("Multi-Quewry Retrivel Complete")
        return self.all_retrieval_results

    def reciprocal_rank_fusion(self, chunk_lists, k=60, verbose=True):
        if verbose:
            print("\n" + "=" * 60)
            print("Applying Reciprocal Rank Fusion")
            print(f"\nUsing k+{k}")
            print(f"calculating RRF scores... \n")

        # Data structures for RRF calculation
        rrf_scores = defaultdict(float)
        all_unique_chunks = {}

        # For verbose output - track chunk IDs
        chunk_id_map = {}
        chunk_counter = 1
        for query_indx, chunks in enumerate(chunk_lists, 1):
            if verbose:
                print(f"Processing Query {query_indx} results")
            # Go through each chunk in this query's results
            for position, chunk in enumerate(chunks, 1):
                # Use chunk content as unique identifier
                chunk_content = chunk.page_content

                # Assign a simple ID if we haven't seen this chunk before
                if chunk_content not in chunk_id_map:
                    chunk_id_map[chunk_content] = f"chunk_{chunk_counter}"
                    chunk_counter += 1

                chunk_id = chunk_id_map[chunk_content]

                # Store the chunk object (in case we haven't seen it before)
                all_unique_chunks[chunk_content] = chunk

                # Calculate posistion score: 1/(k+ position)
                position_score = 1 / (k + position)

                # Add to RRF Score
                rrf_scores[chunk_content] += position_score

                if verbose:
                    print(
                        f" Position {position}:{chunk_id}+{position_score:4f}(ruuning Total: {rrf_scores[chunk_content]:4f})"
                    )
                    print(f" Preview: {chunk_content[:80]}...")

            # sort chunks by RRF score (highest first)
        self.sorted_chunks = sorted(
            [
                (all_unique_chunks[chunk_content], score)
                for chunk_content, score in rrf_scores.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        if verbose:
            print(
                f" RRF Complete Processed {len(self.sorted_chunks)} unique chunks from {len(chunk_lists)} queries"
            )
        return [document for document, score in self.sorted_chunks]

        # Apply RRF to our retrieval results

    def reranker_chunks(self):
        reranker = CrossEncoderReranker(
            model=self.cross_encoder, top_n=self.reranker_k_no
        )
        documents_only = [item[0] for item in self.sorted_chunks]

        # DEBUG: confirm the relevant doc is even in the candidate pool
        print(f"\n=== Candidates going into reranker ({len(documents_only)} total) ===")
        for i, doc in enumerate(documents_only, 1):
            print(f"{i}. {doc.page_content[:80]}")

        self.reranked_docs = reranker.compress_documents(
            documents_only, self.original_query
        )

        print(f"\n=== Reranked output ({len(self.reranked_docs)} total) ===")
        for i, doc in enumerate(self.reranked_docs, 1):
            print(f"{i:2d}. {doc.page_content[:80]}")

        return self.reranked_docs

    def generate_final_answer(self, chunks=None, query=None):
        """Generate final answer using multimodal content and local Qwen-VL model"""
        if chunks is None:
            chunks = self.reranked_docs

        if query is None:
            query = self.original_query
        try:
            # Build the base prompt
            prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
            # Append retrieved documents, tables and text
            for i, chunk in enumerate(chunks):
                prompt_text += f"--- Document {i+1} ---\n"

                if "original_content" in chunk.metadata:
                    original_data = chunk.metadata["original_content"]

                    if isinstance(original_data, str):
                        original_data = json.loads(original_data)

                    raw_text = original_data.get("raw_text", "")

                    if raw_text:
                        prompt_text += f"TEXT:\n{raw_text}\n\n"

                    # Add tables as HTML
                    tables_html = original_data.get("tables_html", [])
                    if tables_html:
                        prompt_text += "TABLES:\n"
                        for j, table in enumerate(tables_html):
                            prompt_text += f"Table {j+1}:\n{table}\n\n"

                prompt_text += "\n"

            prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

            # Build message content starting with text
            message_content = [{"type": "text", "text": prompt_text}]

            # Add all images from all chunks and prepend placeholders
            for chunk in chunks:
                if "original_content" in chunk.metadata:
                    raw = chunk.metadata["original_content"]
                    if isinstance(raw, str):
                        raw = json.loads(raw)
                    images_base64 = raw.get("images_base64", [])
                    for image_base64 in images_base64:
                        message_content[0]["text"] = (
                            "<image>\n" + message_content[0]["text"]
                        )
                        message_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            }
                        )

            message = HumanMessage(content=message_content)
            response = self.llm.invoke([message])
            return response.content
        except json.JSONDecodeError as e:
            print("not sucessful{e}")


def main_retrival_pipeline(query: str,collection="DemoRAG"):
    start = time.perf_counter()
    process_text = st.empty()
    pipeline = retrival_pipeline(collection=collection)
    all_retrieval_results = pipeline.multiquery_RRM(query)
    process_text.write(f"Process : multi-queryies:{all_retrieval_results[:2][:100]}")
    fused_results = pipeline.reciprocal_rank_fusion(
        all_retrieval_results, k=60, verbose=False
    )
    process_text.write(
        f"Process : Hybrid BM25 and MMR retrival chunks:{fused_results[:2][:100]}"
    )
    reranked_docs_c = pipeline.reranker_chunks()
    process_text.write(f"Process : Reranker {reranked_docs_c[:2][:100]}")
    response = pipeline.generate_final_answer(chunks=reranked_docs_c, query=query)
    thinking = re.search(r"<think>(.*?)</think>", str(response), re.DOTALL)
    with st.expander("think", expanded=False):
        st.write(thinking.group(1).strip())
    response = re.sub(
        pattern=r"<think>.*?</think>", repl="", string=str(response), flags=re.DOTALL
    )
    process_text.empty()
    end = time.perf_counter()
    time_taken = start - end
    return response, time_taken


if __name__ == "__main__":
    main_retrival_pipeline(query="what is Easy Build")
