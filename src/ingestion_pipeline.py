import os
import json
from typing import List
from dotenv import load_dotenv
import time
from models import get_embedding_model,build_llms,build_structured_llm

# LangChain and Unstructured imports
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres import PGVector 
# Load environment variables from .env file
load_dotenv()

import streamlit as st
from langsmith import traceable


# Configuration path


class MultiModalRAG:
    def __init__(
        self,
        export_json_path: str = "chunks_export.json",
        llm_api_base: str = None,
        llm_api_key: str = None,
    ):
        print(f" Initializing Embeddings:")
        self.embeddings = get_embedding_model()
        self.export_json_path = export_json_path

        # Load API details from env if not provided
        
        # Initialize vision-capable models API with fallbacks
        self.llm_grok,self.llm_open_router,self.local_llm_lazy=build_llms()
        self.llm = self.llm_grok.with_fallbacks([self.llm_open_router, self.local_llm_lazy])
      

    @traceable(name="partition_documents",project_name="Injestion_pipeline")
    def partition_documents(self, file_path: str = None):
        path = file_path
        if os.path.isfile(path):
            files = [path]
            print(files)
        elif os.path.isdir(path):
            files = []
            print("path")
            for root, dirs, filenames in os.walk(path):
                for f in filenames:
                    files.append(os.path.join(root, f))
        # elif not os.path.isdir(path):
        #     print(f"❌ Error: {path} not found")
        #     return []   # return empty list, not None

        all_elements = []
        target_extensions = (".docx", ".xlsx", ".csv", ".pdf", ".pptx")
        for file in files:
            print(f"file exists:{file}")
            if file.lower().endswith(target_extensions):
                print(f"file process try:{file}")
                try:
                    print(f"📄 Partitioning document: {file}")
                    elements = partition(
                        filename=file,
                        strategy="hi_res",
                        infer_table_structure=True,
                        extract_image_block_types=["Image"],
                        extract_image_block_to_payload=True,
                    )
                    all_elements.extend(elements)
                except Exception as e:
                    print(f"⚠️ Failed to partition {file}: {e}")
            else:
                print(f"file format is not supported yet: {file}")

        return all_elements  # always a list

    @traceable(name="create_chunks_by_title",project_name="Injestion_pipeline")
    def create_chunks_by_title(self, elements):
        """Create intelligent chunks using title-based strategy"""
        print("🔨 Creating smart chunks...")
        chunks = chunk_by_title(
            elements,
            max_characters=3000,
            new_after_n_chars=2400,
            combine_text_under_n_chars=500,
        )
        print(f"✅ Created {len(chunks)} chunks")
        return chunks

    @traceable(name="separate_content_types",project_name="Injestion_pipeline")
    def separate_content_types(self, chunk):
        """Analyze what types of content are in a chunk"""
        content_data = {
            "text": chunk.text,
            "tables": [],
            "images": [],
            "types": ["text"],
        }

        if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
            for element in chunk.metadata.orig_elements:
                element_type = type(element).__name__

                # Handle Tables
                if element_type == "Table":
                    content_data["types"].append("table")
                    table_html = getattr(element.metadata, "text_as_html", element.text)
                    content_data["tables"].append(table_html)

                # Handle Images
                elif element_type == "Image":
                    if hasattr(element, "metadata") and hasattr(
                        element.metadata, "image_base64"
                    ):
                        content_data["types"].append("image")
                        content_data["images"].append(element.metadata.image_base64)

        content_data["types"] = list(set(content_data["types"]))
        return content_data

    @traceable(name="create_ai_enhanced_summary",project_name="Injestion_pipeline")
    def create_ai_enhanced_summary(
        self, text: str, tables: List[str], images: List[str]
    ) -> str:
        """Create AI-enhanced summary for mixed content using local Qwen-VL model"""
        try:
            prompt_text = f"""You are creating a searchable description for document content retrieval.

                CONTENT TO ANALYZE:
                TEXT CONTENT:
                {text}
                """

            # Add tables if present
            if tables:
                prompt_text += "\nTABLES:\n"
                for i, table in enumerate(tables):
                    prompt_text += f"Table {i+1}:\n{table}\n\n"

                prompt_text += """
            YOUR TASK:
            Generate a comprehensive, searchable description that covers:
            1. Key facts, numbers, and data points from text and tables
            2. Main topics and concepts discussed
            3. Questions this content could answer
            4. Visual content analysis (charts, diagrams, patterns in images)
            5. Alternative search terms users might use

            Make it detailed and searchable - prioritize findability over brevity.

            SEARCHABLE DESCRIPTION:"""

            # Fixed schema bug: type must be the literal 'text'
            message_content = [{"type": "text", "text": prompt_text}]

            # Add images to the message payload and prepend placeholders
            for image_base64 in images:
                message_content[0]["text"] = "<image>\n" + message_content[0]["text"]
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    }
                )

            message = HumanMessage(content=message_content)
            response = self.llm.invoke([message])
            return response.content

        except Exception as e:
            print(f"     ❌ AI summary failed: {e}")
            summary = f"{text[:300]}..."
            if tables:
                summary += f" [Contains {len(tables)} table(s)]"
            if images:
                summary += f" [Contains {len(images)} image(s)]"
            return summary

    @traceable(name="summarise_chunks",project_name="Injestion_pipeline")
    def summarise_chunks(self, chunks,collection="DemoRAG"):
        """Process all chunks with AI Summaries"""
        print("🧠 Processing chunks with AI Summaries...")
        langchain_documents = []
        total_chunks = len(chunks)
        print(total_chunks)
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            current_chunk = i + 1
            print(f"   Processing chunk {current_chunk}/{total_chunks}")

            progress_bar.progress(current_chunk / total_chunks)

            # Analyze chunk content
            content_data = self.separate_content_types(chunk)

            print(f"     Types found: {content_data['types']}")
            print(
                f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}"
            )

            # Create AI-enhanced Summary if chunk has tables or images
            if content_data["tables"] or content_data["images"]:
                print("     → Creating AI summary for mixed content...")
                try:
                    enhanced_content = self.create_ai_enhanced_summary(
                        content_data["text"],
                        content_data["tables"],
                        content_data["images"],
                    )
                    print("     → AI summary completed")
                    print(
                        f"     → Enhanced content preview: {enhanced_content[:200]}..."
                    )
                except Exception as e:
                    print(
                        f"     ❌ AI summary failed with error: {e}, falling back to raw text"
                    )
                    enhanced_content = content_data["text"]
            else:
                print("     → Using raw text (no tables/images)")
                enhanced_content = content_data["text"]

            doc = Document(
                page_content=enhanced_content,
                metadata={
                    "original_content": json.dumps(
                        {
                            "raw_text": content_data["text"],
                            "tables_html": content_data["tables"],
                            "images_base64": content_data["images"],
                        }
                    )
                },
            )
            langchain_documents.append(doc)
        progress_bar.empty()
        print(langchain_documents)
        
        database=self.create_vector_store(collection)
        database.add_documents(langchain_documents)
        print(f"✅ Processed {len(langchain_documents)} chunks")
        return langchain_documents

# Example: reshaping a chunk into a cleaner Document

    

    def export_chunks_to_json(self, chunks, filename=None):
        """Export processed chunks to clean JSON format"""
        path = filename or self.export_json_path
        export_data = []
        for i, doc in enumerate(chunks):
            chunk_data = {
                "chunk_id": i + 1,
                "enhanced_content": doc.page_content,
                "metadata": {
                    "original_content": json.loads(
                        doc.metadata.get("original_content", "{}")
                    )
                },
            }
            export_data.append(chunk_data)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Exported {len(export_data)} chunks to {path}")
        return export_data

    @traceable(name="create_vector_store",project_name="Injestion_pipeline")
    def create_vector_store(self,collection):
        """Create and persist PG vector store using Supabase"""
        

        self.db = PGVector(
            embeddings=self.embeddings,
            collection_name= collection,
            connection=os.environ["DATABASE_URL"],
            use_jsonb = True
        )
        
        
        return self.db

  

#     def retrieve_documents(self, query: str, k: int = 2):
#         """Retrieve top k relevant documents for a given query"""

#         retriever = self.db.as_retriever(search_kwargs={"k": k})
#         return retriever.invoke(query)

#     def generate_final_answer(self, chunks, query):
#         """Generate final answer using multimodal content and local Qwen-VL model"""
#         try:
#             # Build the base prompt
#             prompt_text = f"""Based on the following documents, please answer this question: {query}

# CONTENT TO ANALYZE:
# """
#             # Append retrieved documents, tables and text
#             for i, chunk in enumerate(chunks):
#                 prompt_text += f"--- Document {i+1} ---\n"

#                 if "original_content" in chunk.metadata:
#                     original_data = json.loads(chunk.metadata["original_content"])

#                     # Add raw text
#                     raw_text = original_data.get("raw_text", "")
#                     if raw_text:
#                         prompt_text += f"TEXT:\n{raw_text}\n\n"

#                     # Add tables as HTML
#                     tables_html = original_data.get("tables_html", [])
#                     if tables_html:
#                         prompt_text += "TABLES:\n"
#                         for j, table in enumerate(tables_html):
#                             prompt_text += f"Table {j+1}:\n{table}\n\n"

#                 prompt_text += "\n"

#             prompt_text += """
# Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

# ANSWER:"""

#             # Build message content starting with text
#             message_content = [{"type": "text", "text": prompt_text}]

#             # Add all images from all chunks and prepend placeholders
#             for chunk in chunks:
#                 if "original_content" in chunk.metadata:
#                     original_data = json.loads(chunk.metadata["original_content"])
#                     images_base64 = original_data.get("images_base64", [])

#                     for image_base64 in images_base64:
#                         message_content[0]["text"] = (
#                             "<image>\n" + message_content[0]["text"]
#                         )
#                         message_content.append(
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/jpeg;base64,{image_base64}"
#                                 },
#                             }
#                         )

#             message = HumanMessage(content=message_content)
#             response = self.llm.invoke([message])
#             return response.content

#         except Exception as e:
#             print(f"❌ Answer generation failed: {e}")
#             return "Sorry, I encountered an error while generating the final answer."

#     def query(self, query: str, k: int = 2) -> str:
#         """End-to-end retrieval and answer generation for a given query"""
#         retrieved_docs = self.retrieve_documents(query, k)

#         print("\n--- Retrieved Documents ---")
#         for idx, doc in enumerate(retrieved_docs):
#             print(f"\n[Document {idx+1}]")
#             print(doc.page_content[:300] + "...")

#         print("\n--- Generating Final Answer ---")
#         answer = self.generate_final_answer(retrieved_docs, query)
#         return answer


def ingestion_pipeline(file_path="test_documents",collection="DemoRAG"):
    """Run the full ingestion pipeline: partition, chunk, summarise, export, and create vector store"""
    start = time.perf_counter()
    self = MultiModalRAG()

    def elapsed():
        return f"{time.perf_counter()-start:.1f}s"

    placeholder = st.empty()
    placeholder.write(f"document processing... {elapsed()}")
    elements = self.partition_documents(file_path)
    placeholder.write(f"creating chunks... {elapsed()}")
    chunks = self.create_chunks_by_title(elements)

    placeholder.write(f"generating AI summary of chunks...")
    langchain_docs = self.summarise_chunks(chunks,collection)
    placeholder.write(f"Your Documents Process : Ask any query")
    end = time.perf_counter()
    time_taken = end - start
    placeholder
    return langchain_docs, time_taken


if __name__ == "__main__":
    ingestion_pipeline()
