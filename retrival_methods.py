
from dotenv import load_dotenv


import json

from pydantic import BaseModel
from collections import defaultdict
# Load a local open-source cross-encoder model

#Lanchain dependencies
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from langchain_core.messages import SystemMessage ,HumanMessage

load_dotenv()

class retrival_pipeline():


    def __init__(self):
        self.llm=init_chat_model(
            model="Qwen/Qwen2-VL-7B-Instruct-AWQ",
            openai_api_base="http://localhost:8005/v1",
            openai_api_key="pranshu123",
            model_provider="openai",
            temperature=0.0,
        )
        
        
        self.original_query ="what is Easy Build revenue, profit and Sales"
        self.persistent_directory = "dbs/chroma"
        self.embedding_model = OllamaEmbeddings(model="qwen3-embedding:4b")
        
        self.db = Chroma(
            persist_directory=self.persistent_directory,
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        self.raw=self.db.get(include=['documents','metadatas'])
        self.documents=[]
        print(self.raw)
        for text, meta in zip(self.raw["documents"],self.raw["metadatas"]):
            meta=meta or {}
            for k,v  in meta.items():
                if isinstance(v,str):
                    try:
                        meta[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
            self.documents.append(Document(page_content=text,meta_data=meta))
        
        # number of the chunks
        self.k=10 # number of the chunks from hybrid retrival and mmr
        self.reranker_k_no=3 # number of the chunks from reranker 


        # intialize the reranker model from transformers
        self.cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base",model_kwargs={"device":"cpu"})

    def hybrid_retriver(self):
        bm25_retriever=BM25Retriever.from_documents(self.documents)
        bm25_retriever.k=self.k
        vector_retriever=self.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":self.k,
                "fetch_k":20,
                "lambda_mult":0.5
            }
        )
        
        self.hybrid_retriever=EnsembleRetriever(
            retrievers=[vector_retriever,bm25_retriever],
            weights=[0.7,0.3]
        )
        print("-----Hybrid Retriever intialised")
        return self.hybrid_retriever

    def multiquery_RRM(self,original_query):
        
        if original_query:
            self.original_query=original_query
        
        class Queryvariations(BaseModel):
            queries : list[str]

        llm_with_tool = self.llm.with_structured_output(Queryvariations)

        prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

        Original query: {self.original_query}

        Return 3 alternative queries that rephrase or approach the same question from different angles."""

        response=llm_with_tool.invoke(prompt)

        query_variations=response.queries


        retriever = self.hybrid_retriver()
        

        self.all_retrieval_results=[]

        for i , query in enumerate(query_variations,1):
            print(f"\n=====Results for Query {i}:{query}===")
            docs = retriever.invoke(query)
            self.all_retrieval_results.append(docs) # Store for RRF calculation
            print(f"Retrived {len(docs)} documents:\n")
            for j, doc in enumerate(docs,1):
                print(f"Document{j}:")
                print(f"{doc.page_content[:150]}")
            print("-"*50)
        print("Multi-Quewry Retrivel Complete")
        return self.all_retrieval_results


    def reciprocal_rank_fusion(self,chunk_lists, k=60, verbose=True):
        if verbose:
            print("\n"+"="*60)
            print("Applying Reciprocal Rank Fusion")
            print(f"\nUsing k+{k}")
            print(f"calculating RRF scores... \n")

        # Data structures for RRF calculation
        rrf_scores= defaultdict(float)
        all_unique_chunks={}

        # For verbose output - track chunk IDs
        chunk_id_map ={}
        chunk_counter=1
        for query_indx, chunks in enumerate(chunk_lists,1):
            if verbose:
                print(f"Processing Query {query_indx} results")
            # Go through each chunk in this query's results
            for position, chunk in enumerate(chunks,1):
                # Use chunk content as unique identifier
                chunk_content= chunk.page_content

                # Assign a simple ID if we haven't seen this chunk before 
                if chunk_content not in chunk_id_map:
                    chunk_id_map[chunk_content]=f"chunk_{chunk_counter}"
                    chunk_counter +=1
                
                chunk_id= chunk_id_map[chunk_content]
                
                # Store the chunk object (in case we haven't seen it before)
                all_unique_chunks[chunk_content]=chunk

                # Calculate posistion score: 1/(k+ position)
                position_score= 1/(k+position) 

                # Add to RRF Score
                rrf_scores[chunk_content]+= position_score

                if verbose:
                    print(f" Position {position}:{chunk_id}+{position_score:4f}(ruuning Total: {rrf_scores[chunk_content]:4f})")
                    print(f" Preview: {chunk_content[:80]}...")
            
            # sort chunks by RRF score (highest first)
        self.sorted_chunks= sorted([(all_unique_chunks[chunk_content],score)for chunk_content, score in rrf_scores.items()],
                                key=lambda x:x[1],
                                reverse=True
                                )
        if verbose:
                print(f" RRF Complete Processed {len(self.sorted_chunks)} unique chunks from {len(chunk_lists)} queries")
        return self.sorted_chunks
        
        # Apply RRF to our retrieval results
    def reranker_chunks(self):
        # Initialize Cohere reranker
        reranker= CrossEncoderReranker(model=self.cross_encoder,top_n=self.reranker_k_no)
        documents_only = [item[0] for item in self.sorted_chunks]

        # Rerank the retrived documents

        self.reranked_docs= reranker.compress_documents(documents_only,self.original_query)
        # show reranked results
        for i, doc in enumerate(self.reranked_docs,1):
            print(f"{i:2d}.{doc.page_content}")
        return self.reranked_docs
    def generate_final_answer(self, chunks, query):
        """Generate final answer using multimodal content and local Qwen-VL model"""
        if self.reranked_docs:
            chunks=self.reranked_docs
        if  self.original_query:
            query=self.original_query
        try:
            # Build the base prompt
            prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
            # Append retrieved documents, tables and text
            for i, chunk in enumerate(chunks):
                prompt_text += f"--- Document {i+1} ---\n"
                
                if "original_content" in chunk.metadata:
                    original_data = json.loads(chunk.metadata["original_content"])
                    
                    # Add raw text
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
                    original_data = json.loads(chunk.metadata["original_content"])
                    images_base64 = original_data.get("images_base64", [])
                    
                    for image_base64 in images_base64:
                        message_content[0]["text"] = "<image>\n" + message_content[0]["text"]
                        message_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        })
            
            message = HumanMessage(content=message_content)
            response = self.llm.invoke([message])
            return response.content
        except json.JSONDecodeError as e :
            print("not sucessful{e}")

if __name__=="__main__":
    pipeline = retrival_pipeline()
    all_retrieval_results = pipeline.multiquery_RRM("what is infra")    
    fused_results = pipeline.reciprocal_rank_fusion(all_retrieval_results, k=60, verbose=True)
    reranked_docs_c =pipeline.reranker_chunks()
    print
    response=pipeline.generate_final_answer(chunks=reranked_docs_c[:2],query="what is infra")