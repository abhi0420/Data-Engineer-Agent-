from google.oauth2 import service_account
from google.cloud import storage
import os

path_to_gcs_service_account = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

class GCPSource:
    def __init__(self, project_id : str, bucket_name : str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.creds = service_account.Credentials.from_service_account_file(path_to_gcs_service_account)
        self.client = storage.Client(credentials=self.creds, project=self.project_id)
        self.bucket = self.client.bucket(self.bucket_name)

    def list_blobs(self, prefix: str = "") -> list[str]:
        blobs = self.bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]
    
    def download_file(self, filename: str, dest_path: str = None) -> str:
        blob = self.bucket.blob(filename)
        if not blob.exists():
            return f"File '{filename}' not found in bucket"

        if dest_path is None:
            dest_path = f"./data/{filename}"
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)
        return dest_path

    def upload_file(self, source_file_path: str, dest_blob_name: str) -> str:
        if not os.path.exists(source_file_path):
            return f"Source file '{source_file_path}' not found"

        blob = self.bucket.blob(dest_blob_name)
        blob.upload_from_filename(source_file_path)
        return f"gs://{self.bucket_name}/{dest_blob_name}"
    
    def file_exists(self, filename : str) -> bool:
        blob = self.bucket.blob(filename)
        return blob.exists()
    
    def bucket_exists(self) -> bool:
        return self.bucket.exists()

