"""Test server-side binding optimization for psycopg3."""

import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel
from prompt_improver.database.psycopg_client import TypeSafePsycopgClient
from prompt_improver.database.config import DatabaseConfig


class TestDataModel(BaseModel):
    """Test model for server-side binding tests."""
    id: int
    name: str
    value: float


class TestServerSideBinding:
    """Test server-side binding functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock database config."""
        with patch.dict(os.environ, {
            'POSTGRES_USERNAME': 'test_user',
            'POSTGRES_PASSWORD': 'test_pass',
            'POSTGRES_HOST': 'localhost',
            'POSTGRES_PORT': '5432',
            'POSTGRES_DATABASE': 'test_db'
        }):
            config = DatabaseConfig()
            return config

    @pytest.fixture
    def mock_client(self, mock_config):
        """Create a mock client for testing."""
        client = TypeSafePsycopgClient.__new__(TypeSafePsycopgClient)
        client.config = mock_config
        client.metrics = MagicMock()
        client.metrics.record_query = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_fetch_models_server_side_prepared(self, mock_client):
        """Test fetch_models_server_side with prepared statements."""
        # Mock the connection and cursor
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = [
            {"id": 1, "name": "test1", "value": 10.5},
            {"id": 2, "name": "test2", "value": 20.7}
        ]
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        
        # Mock the connection context manager properly
        mock_client.connection = AsyncMock()
        mock_client.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_client.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Execute the method
        query = "SELECT id, name, value FROM test_table WHERE id = %(id)s"
        params = {"id": 1}
        
        result = await mock_client.fetch_models_server_side(
            TestDataModel, query, params, prepared=True
        )
        
        # Verify the result
        assert len(result) == 2
        assert isinstance(result[0], TestDataModel)
        assert result[0].id == 1
        assert result[0].name == "test1"
        assert result[0].value == 10.5
        
        # Verify prepared statement was used
        mock_cursor.execute.assert_called_once_with(query, params, prepare=True)
        
        # Verify metrics were recorded
        mock_client.metrics.record_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_models_server_side_not_prepared(self, mock_client):
        """Test fetch_models_server_side without prepared statements."""
        # Mock the connection and cursor
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = [
            {"id": 1, "name": "test1", "value": 10.5}
        ]
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        
        # Mock the connection context manager properly
        mock_client.connection = AsyncMock()
        mock_client.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_client.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Execute the method
        query = "SELECT id, name, value FROM test_table WHERE id = %(id)s"
        params = {"id": 1}
        
        result = await mock_client.fetch_models_server_side(
            TestDataModel, query, params, prepared=False
        )
        
        # Verify the result
        assert len(result) == 1
        assert isinstance(result[0], TestDataModel)
        
        # Verify regular execute was used (no prepare=True)
        mock_cursor.execute.assert_called_once_with(query, params)

    @pytest.mark.asyncio
    async def test_execute_batch_server_side(self, mock_client):
        """Test execute_batch_server_side functionality."""
        # Mock the connection and cursor
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 3
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        
        # Mock the connection context manager properly
        mock_client.connection = AsyncMock()
        mock_client.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_client.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Execute the method
        query = "INSERT INTO test_table (name, value) VALUES (%(name)s, %(value)s)"
        params_list = [
            {"name": "test1", "value": 10.5},
            {"name": "test2", "value": 20.7},
            {"name": "test3", "value": 30.9}
        ]
        
        result = await mock_client.execute_batch_server_side(
            query, params_list, prepared=True
        )
        
        # Verify the result
        assert result == 3
        
        # Verify executemany was called with prepare=True
        mock_cursor.executemany.assert_called_once_with(query, params_list, prepare=True)
        
        # Verify metrics were recorded
        mock_client.metrics.record_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_one_server_side(self, mock_client):
        """Test fetch_one_server_side functionality."""
        # Mock the connection and cursor
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone.return_value = {"id": 1, "name": "test1", "value": 10.5}
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        
        # Mock the connection context manager properly
        mock_client.connection = AsyncMock()
        mock_client.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_client.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Execute the method
        query = "SELECT id, name, value FROM test_table WHERE id = %(id)s"
        params = {"id": 1}
        
        result = await mock_client.fetch_one_server_side(
            TestDataModel, query, params, prepared=True
        )
        
        # Verify the result
        assert isinstance(result, TestDataModel)
        assert result.id == 1
        assert result.name == "test1"
        assert result.value == 10.5
        
        # Verify prepared statement was used
        mock_cursor.execute.assert_called_once_with(query, params, prepare=True)

    @pytest.mark.asyncio
    async def test_execute_server_side(self, mock_client):
        """Test execute_server_side functionality."""
        # Mock the connection and cursor
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 1
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        
        # Mock the connection context manager properly
        mock_client.connection = AsyncMock()
        mock_client.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_client.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Execute the method
        query = "UPDATE test_table SET value = %(value)s WHERE id = %(id)s"
        params = {"id": 1, "value": 15.0}
        
        result = await mock_client.execute_server_side(
            query, params, prepared=True
        )
        
        # Verify the result
        assert result == 1
        
        # Verify prepared statement was used
        mock_cursor.execute.assert_called_once_with(query, params, prepare=True)

    @pytest.mark.asyncio
    async def test_server_side_methods_performance_tracking(self, mock_client):
        """Test that server-side methods track performance correctly."""
        # Mock the connection and cursor
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = [{"id": 1, "name": "test", "value": 10.5}]
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        
        # Mock the connection context manager properly
        mock_client.connection = AsyncMock()
        mock_client.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_client.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Execute the method
        query = "SELECT id, name, value FROM test_table"
        
        await mock_client.fetch_models_server_side(TestDataModel, query, prepared=True)
        
        # Verify metrics were recorded with [SERVER-SIDE] prefix
        mock_client.metrics.record_query.assert_called_once()
        call_args = mock_client.metrics.record_query.call_args
        assert call_args[0][0].startswith("[SERVER-SIDE]")
        assert query in call_args[0][0] 