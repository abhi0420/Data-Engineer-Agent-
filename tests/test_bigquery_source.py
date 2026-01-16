import pytest
from unittest.mock import Mock, MagicMock, patch
from google.cloud import bigquery

class TestBigQuerySource:
    """Tests for BigQuerySource class"""

    @pytest.fixture
    def mock_credentials(self):
        """Mock GCP credentials - must be applied before BigQuerySource init"""
        with patch('bigquery_source.service_account.Credentials.from_service_account_file') as mock_creds:
            mock_creds.return_value = Mock()
            yield mock_creds

    @pytest.fixture
    def mock_bq_client(self):
        """Mock BigQuery client"""
        with patch('bigquery_source.bigquery.Client') as mock_client:
            yield mock_client.return_value  # Return the instance, not the class

    @pytest.fixture
    def bq_source(self, mock_credentials, mock_bq_client):
        """Create BigQuerySource with mocked dependencies - order matters!"""
        from bigquery_source import BigQuerySource
        return BigQuerySource(project_id="test_project")

    # ========== CREATE DATASET TESTS ==========
    
    def test_create_dataset_success(self, bq_source, mock_bq_client):
        """Test successful dataset creation"""
        mock_bq_client.create_dataset.return_value = None
        
        result = bq_source.create_dataset("test_dataset", "US")
        
        mock_bq_client.create_dataset.assert_called_once()
        assert result == "Dataset test_dataset created."

    def test_create_dataset_failure(self, bq_source, mock_bq_client):
        """Test dataset creation failure"""
        mock_bq_client.create_dataset.side_effect = Exception("Already exists")
        
        result = bq_source.create_dataset("test_dataset", "US")
        
        assert "ERROR" in result
        assert "test_dataset" in result

    # ========== CREATE TABLE TESTS ==========

    def test_create_table_success(self, bq_source, mock_bq_client):
        """Test successful table creation"""
        mock_bq_client.create_table.return_value = None
        schema = [bigquery.SchemaField("col1", "STRING")]

        result = bq_source.create_table("test_dataset", "test_table", schema)
        
        mock_bq_client.create_table.assert_called_once()
        assert result == "Table test_table created in dataset test_dataset."

    def test_create_table_failure(self, bq_source, mock_bq_client):
        """Test table creation failure"""
        mock_bq_client.create_table.side_effect = Exception("Permission denied")
        schema = [bigquery.SchemaField("col1", "STRING")]

        result = bq_source.create_table("test_dataset", "test_table", schema)
        
        assert "ERROR" in result
        assert "test_table" in result

    # ========== QUERY TESTS ==========

    def test_query_success(self, bq_source, mock_bq_client):
        """Test successful query execution"""
        import pandas as pd
        
        # Mock the query job and result
        mock_df = pd.DataFrame({"col1": ["a", "b"]})
        mock_result = Mock()
        mock_result.to_dataframe.return_value = mock_df
        mock_job = Mock()
        mock_job.result.return_value = mock_result
        mock_bq_client.query.return_value = mock_job

        result = bq_source.query("SELECT * FROM test_table")
        
        mock_bq_client.query.assert_called_once_with("SELECT * FROM test_table")
        assert len(result) == 2

    # ========== SCHEMA INFERENCE TESTS ==========

    def test_get_schema_of_df(self, bq_source):
        """Test schema inference from DataFrame - no mocking needed"""
        import pandas as pd
        
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'age': [30, 25],
            'salary': [50000.0, 60000.0]
        })
        
        schema = bq_source.get_schema_of_df(df)
        
        assert len(schema) == 3
        schema_dict = {field.name: field.field_type for field in schema}
        assert schema_dict['name'] == 'STRING'
        assert schema_dict['age'] == 'INTEGER'
        assert schema_dict['salary'] == 'FLOAT'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


