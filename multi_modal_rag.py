import os
import json
from typing import List
from dotenv import load_dotenv

# LangChain and Unstructured imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()

# Configuration paths
PDF_PATH = "/run/media/pranshu/52FB-5ED4/Development Pratice/RAG Industry Level/1744172411069_ready-mix-concrete-catalogue.pdf"
CHROMA_DB_DIR = "db/chroma_db"
EXPORT_JSON_PATH = "chunks_export.json"

def partition_documents(file_path: str):
    """Extract elements from PDF using unstructured library"""
    print(f"📄 Partitioning document: {file_path}")
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )
    print(f"✅ Extracted {len(elements)} elements")
    return elements

def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print("🔨 Creating smart chunks...")
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }

    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # Handle Tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle Images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
                    
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content using local Qwen-VL model"""
    try:
        # Initialize vision-capable local LLM
        llm = init_chat_model(
            model="Qwen/Qwen2-VL-7B-Instruct-AWQ",
            openai_api_base="http://localhost:8005/v1",
            openai_api_key="pranshu123",
            model_provider="openai",
            temperature=0.0,
        )
        
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
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary

def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    print("🧠 Processing chunks with AI Summaries...")
    langchain_documents = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")

        # Analyze chunk content
        content_data = separate_content_types(chunk)

        print(f"     Types found: {content_data['types']}")
        print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

        # Create AI-enhanced Summary if chunk has tables or images
        if content_data['tables'] or content_data['images']:
            print("     → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'],
                    content_data['images'],
                )
                print("     → AI summary completed")
                print(f"     → Enhanced content preview: {enhanced_content[:200]}...")
            except Exception as e:
                print(f"     ❌ AI summary failed with error: {e}, falling back to raw text")
                enhanced_content = content_data['text']
        else:
            print("     → Using raw text (no tables/images)")
            enhanced_content = content_data['text']
        
        doc = Document(
            page_content=enhanced_content, 
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )
        langchain_documents.append(doc)
    
    print(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents

def export_chunks_to_json(chunks, filename=EXPORT_JSON_PATH):
    """Export processed chunks to clean JSON format"""
    export_data = []
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk_data)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Exported {len(export_data)} chunks to {filename}")
    return export_data

def create_vector_store(documents, persist_directory=CHROMA_DB_DIR):
    """Create and persist ChromaDB vector store using local nomic embeddings"""
    print("🔮 Creating embeddings and storing in ChromaDB...")
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"✅ Vector store created and saved to {persist_directory}")
    return vectorstore

def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content and local Qwen-VL model"""
    try:
        # Initialize vision-capable local LLM
        llm = init_chat_model(
            model="Qwen/Qwen2-VL-7B-Instruct-AWQ",
            openai_api_base="http://localhost:8005/v1",
            openai_api_key="pranshu123",
            model_provider="openai",
            temperature=0.0,
        )
        
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
        response = llm.invoke([message])
        return response.content
        
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the final answer."

def main():
    # 1. Partition PDF
    elements = partition_documents(PDF_PATH)
    
    # 2. Chunk elements
    chunks = create_chunks_by_title(elements)
    
    # 3. Summarise chunks
    langchain_docs = summarise_chunks(chunks)
    
    # 4. Export to JSON
    export_chunks_to_json(langchain_docs)
    
    # 5. Create Vector Store
    db = create_vector_store(langchain_docs)
    
    # 6. Test Retrieval and Generation
    query = "What is the ready mix concrete strength or applications?"
    print(f"\n🔍 Testing retrieval and generation with query: '{query}'")
    retriever = db.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(query)
    
    print("\n--- Retrieved Documents ---")
    for idx, doc in enumerate(retrieved_docs):
        print(f"\n[Document {idx+1}]")
        print(doc.page_content[:300] + "...")
        
    print("\n--- Generating Final Answer ---")
    answer = generate_final_answer(retrieved_docs, query)
    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    main()