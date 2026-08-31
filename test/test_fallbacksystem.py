from src.Config import build_llms

build_llms()

def test_primary_llm_responds(mock_groq_llm):
    result = mock_groq_llm.invoke("What is RAG?")
    assert result.content == "mocked groq response"

def test_fallback_triggers_openrouter(mock_fallback_llm):
    result = mock_fallback_llm.invoke("What is RAG?")
    assert result.content == "response from fallback chain"

def test_retriever_returns_docs(mock_vectorstore, sample_docs):
    results = mock_vectorstore.similarity_search("RAG pipeline")
    assert len(results) == 2
    assert "RAG" in results[0].page_content

# Only runs with INTEGRATION=true
def test_real_llm_call(real_llms):
    result = real_llms.invoke("Say hello")
    assert result is not None