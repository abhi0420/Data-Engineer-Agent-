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
        return GCPSource(project_id="test_project", bucket_name="test_bucket")

    # ========== BUCKET EXISTS TESTS ==========

    def test_bucket_exists_true(self, gcp_source, mock_storage_client):
        """Test bucket exists returns True"""
        # gcp_source.bucket is mock_storage_client.bucket() return value
        mock_bucket = mock_storage_client.bucket.return_value
        mock_bucket.exists.return_value = True
        
        exists = gcp_source.bucket_exists()
        
        mock_bucket.exists.assert_called_once()
        assert exists is True

    def test_bucket_exists_false(self, gcp_source, mock_storage_client):
        """Test bucket exists returns False"""
        mock_bucket = mock_storage_client.bucket.return_value
        mock_bucket.exists.return_value = False
        
        exists = gcp_source.bucket_exists()
        
        mock_bucket.exists.assert_called_once()
        assert exists is False

    # ========== UPLOAD FILE TESTS ==========

    def test_upload_file_success(self, gcp_source, mock_storage_client):
        """Test successful file upload"""
        mock_bucket = mock_storage_client.bucket.return_value
        mock_blob = mock_bucket.blob.return_value

        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            result = gcp_source.upload_file("local_path.txt", "dest_blob.txt")
            
            mock_bucket.blob.assert_called_once_with("dest_blob.txt")
            mock_blob.upload_from_filename.assert_called_once_with("local_path.txt")
            assert "gs://" in result
            assert "test_bucket" in result

    def test_upload_file_failure(self, gcp_source):
        """Test file upload failure due to missing source file"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            result = gcp_source.upload_file("nonexistent_path.txt", "dest_blob.txt")
            
            assert "ERROR" in result

    # ========== DOWNLOAD FILE TESTS ==========

    def test_download_file_success(self, gcp_source, mock_storage_client):
        """Test successful file download"""
        mock_bucket = mock_storage_client.bucket.return_value
        mock_blob = mock_bucket.blob.return_value
        mock_blob.exists.return_value = True

        with patch('os.makedirs'):
            result = gcp_source.download_file("test.csv")
            
            mock_bucket.blob.assert_called_with("test.csv")
            mock_blob.download_to_filename.assert_called_once()
            assert "test.csv" in result

    def test_download_file_not_found(self, gcp_source, mock_storage_client):
        """Test download when file doesn't exist"""
        mock_bucket = mock_storage_client.bucket.return_value
        mock_blob = mock_bucket.blob.return_value
        mock_blob.exists.return_value = False

        result = gcp_source.download_file("nonexistent.csv")
        
        assert "ERROR" in result
        assert "not found" in result

    def test_list_blobs(self, gcp_source, mock_storage_client):
        """Test List blobs in bucket"""
        mock_bucket = mock_storage_client.bucket.return_value
        mock_blob1 = Mock()
        mock_blob1.name = "file1.txt"
        mock_blob2 = Mock()
        mock_blob2.name = "file2.txt"
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

        blobs = gcp_source.list_blobs()
        gcp_source.bucket.list_blobs.assert_called_once()
        assert blobs == ["file1.txt", "file2.txt"]
        


if __name__ == "__main__":
    pytest.main()
    
    