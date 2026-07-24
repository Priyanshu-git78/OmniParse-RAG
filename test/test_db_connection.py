# test/test_db_connection.py
import psycopg2
import os
import pytest
from unittest.mock import patch, MagicMock


# ✅ Unit test — uses mock, no real DB
def test_db_connection_mock(mock_db_connection):
    cursor = mock_db_connection.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    assert result == (1,)


def test_db_cursor_works(mock_db_connection):
    cursor = mock_db_connection.cursor()
    assert cursor is not None


# ✅ Integration test — only runs when real DB available
@pytest.mark.integration
def test_real_db_connection():
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    
    try:
        conn = psycopg2.connect(url)
        assert conn is not None
        conn.close()
    except psycopg2.OperationalError as e:
        pytest.skip(f"DB not reachable: {e}")