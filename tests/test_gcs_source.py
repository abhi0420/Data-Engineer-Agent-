import pytest
from unittest.mock import patch, Mock
from google.cloud import storage

class TestGCPSource:
    """Tests for GCPSource class"""

    @pytest.fixture
    def mock_credentials(self):
        """Mock GCP credentials - must be applied before GCPSource init"""
        with patch('gcs_source.service_account.Credentials.from_service_account_file') as mock_creds:
            mock_creds.return_value = Mock()
            yield mock_creds

    @pytest.fixture
    def mock_storage_client(self):
        """Mock GCP Storage client"""
        with patch('gcs_source.storage.Client') as mock_client:
            yield mock_client.return_value  # Return the instance, not the class

    @pytest.fixture
    def gcp_source(self, mock_credentials, mock_storage_client):
        """Create GCPSource with mocked dependencies - order matters!"""
        from gcs_source import GCPSource
        return GCPSource(project_id="test_project")

    # ========== BUCKET EXISTS TESTS ==========

    def test_bucket_exists_true(self, gcp_source, mock_storage_client):
        """Test bucket exists returns True"""
        mock_storage_client.lookup_bucket.return_value = Mock()
        
        exists = gcp_source.bucket_exists()
        
        mock_storage_client.lookup_bucket.assert_called_once_with(gcp_source.bucket_name)
        assert exists is True

    def test_bucket_exists_false(self, gcp_source, mock_storage_client):
        """Test bucket exists returns False"""
        mock_storage_client.lookup_bucket.return_value = None
        
        exists = gcp_source.bucket_exists()
        
        mock_storage_client.lookup_bucket.assert_called_once_with(gcp_source.bucket_name)
        assert exists is False