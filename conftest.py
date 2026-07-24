# conftest.py  (project root)
import pytest
import os
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────
# 1. ENVIRONMENT SETUP
#    Mirrors what GitHub Actions injects via secrets
# ─────────────────────────────────────────────────────
def pytest_configure(config):
    """Set safe dummy values so tests never hit real APIs."""
    os.environ.setdefault("GROQ_API_KEY",        "test-groq-key")
    os.environ.setdefault("OPENROUTER_API_KEY",  "test-openrouter-key")
    os.environ.setdefault("DATABASE_URL",         os.environ.setdefault("DATABASE_URL", "postgresql://user:password@localhost:5432/testdb"))  # local DB for tests
    os.environ.setdefault("ENV",                  "test")


# ─────────────────────────────────────────────────────
# 2. GROQ LLM FIXTURE (primary LLM in your stack)
# ─────────────────────────────────────────────────────
@pytest.fixture
def mock_groq_llm():
    """Mocked Groq LLM — no real API call, no cost."""
    llm = MagicMock()
    llm.invoke.return_value          = MagicMock(content="mocked groq response")
    llm.ainvoke.return_value         = MagicMock(content="mocked groq async response")
    llm.model_name                   = "llama3-8b-8192"
    return llm


# ─────────────────────────────────────────────────────
# 3. OPENROUTER LLM FIXTURE (fallback LLM)
# ─────────────────────────────────────────────────────
@pytest.fixture
def mock_openrouter_llm():
    """Mocked OpenRouter LLM — used as fallback in your fallback system."""
    llm = MagicMock()
    llm.invoke.return_value          = MagicMock(content="mocked openrouter response")
    llm.ainvoke.return_value         = MagicMock(content="mocked openrouter async response")
    llm.model_name                   = "openai/gpt-3.5-turbo"
    return llm


# ─────────────────────────────────────────────────────
# 4. FALLBACK SYSTEM FIXTURE
#    Primary = Groq, Fallback = OpenRouter
#    This directly maps to your test_fallbacksystem.py
# ─────────────────────────────────────────────────────
@pytest.fixture
def mock_fallback_llm(mock_groq_llm, mock_openrouter_llm):
    """
    Simulates your build_llms() fallback chain.
    Primary: Groq  →  Fallback: OpenRouter
    """
    from unittest.mock import MagicMock
    fallback = MagicMock()
    fallback.primary  = mock_groq_llm
    fallback.fallback = mock_openrouter_llm
    fallback.invoke.return_value = MagicMock(content="response from fallback chain")
    return fallback


# ─────────────────────────────────────────────────────
# 5. EMBEDDINGS FIXTURE
# ─────────────────────────────────────────────────────
@pytest.fixture
def mock_embeddings():
    embeddings = MagicMock()
    embeddings.embed_query.return_value     = [0.1, 0.2, 0.3, 0.4, 0.5]
    embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    return embeddings


# ─────────────────────────────────────────────────────
# 6. VECTOR STORE FIXTURE
# ─────────────────────────────────────────────────────
@pytest.fixture
def mock_vectorstore():
    from langchain_core.documents import Document
    vs = MagicMock()
    vs.similarity_search.return_value = [
        Document(page_content="RAG stands for Retrieval Augmented Generation.",
                 metadata={"source": "test.pdf", "page": 1}),
        Document(page_content="Groq provides fast LLM inference.",
                 metadata={"source": "test.pdf", "page": 2}),
    ]
    vs.as_retriever.return_value = MagicMock()
    vs.as_retriever.return_value.invoke.return_value = vs.similarity_search.return_value
    return vs


# ─────────────────────────────────────────────────────
# 7. DATABASE FIXTURE
#    Uses SQLite locally instead of your real DATABASE_URL
# ─────────────────────────────────────────────────────
@pytest.fixture
def mock_db():
    """Mock DB connection — avoids hitting your real database."""
    db = MagicMock()
    db.query.return_value  = [{"id": 1, "content": "test record"}]
    db.insert.return_value = True
    return db


# ─────────────────────────────────────────────────────
# 8. SAMPLE DOCUMENTS
# ─────────────────────────────────────────────────────
@pytest.fixture
def sample_docs():
    from langchain_core.documents import Document
    return [
        Document(page_content="LLMs are large language models used in AI.",
                 metadata={"source": "intro.pdf", "page": 1}),
        Document(page_content="RAG retrieves relevant documents before generating answers.",
                 metadata={"source": "rag_guide.pdf", "page": 1}),
        Document(page_content="Groq API offers ultra-fast inference for open source models.",
                 metadata={"source": "groq_docs.pdf", "page": 1}),
    ]


# ─────────────────────────────────────────────────────
# 9. INTEGRATION TEST GATE
#    Only runs when INTEGRATION=true is set
#    Use in CI for a separate integration job
# ─────────────────────────────────────────────────────
@pytest.fixture
def real_llms():
    """
    Real build_llms() call — only runs if INTEGRATION=true.
    
    Locally:  INTEGRATION=true uv run pytest
    In CI:    add `INTEGRATION: true` under env: in workflow
    """
    if os.getenv("INTEGRATION") != "true":
        pytest.skip("Skipping real API test. Set INTEGRATION=true to run.")
    from src.models import build_llms
    return build_llms()

# conftest.py
@pytest.fixture
def mock_db_connection():
    """Mock psycopg2 connection — no real PostgreSQL needed."""
    with patch("psycopg2.connect") as mock_connect:
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()

        # cursor().execute().fetchone() chain
        mock_cursor.fetchone.return_value   = (1,)
        mock_cursor.fetchall.return_value   = [(1, "test")]
        mock_conn.cursor.return_value       = mock_cursor
        mock_connect.return_value           = mock_conn

        yield mock_conn