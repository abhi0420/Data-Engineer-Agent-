from google.cloud import bigquery
from typing import List, Dict, Optional, Any
from google.oauth2 import service_account
from dotenv import load_dotenv
import os 

load_dotenv()

path_to_bq_service_account = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


class BigQuerySource:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.creds = service_account.Credentials.from_service_account_file(path_to_bq_service_account)
        self.client = bigquery.Client(credentials=self.creds, project=self.project_id)
        

    def create_dataset(self, dataset_id: str, location: str = "US") -> str:
        dataset_ref = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
        dataset_ref.location = location
        try:
            self.client.create_dataset(dataset_ref)
        except Exception as e:
            return f"ERROR : Failed to create dataset {dataset_id} . Exception: {str(e)}"
        return f"Dataset {dataset_id} created."
    
    def list_datasets(self) -> List[str]:
        datasets = self.client.list_datasets()
        return [dataset.dataset_id for dataset in datasets] 
    
    def create_table(self, dataset_id: str, table_id: str, schema: List[bigquery.SchemaField]) -> str:
        table_ref = self.client.dataset(dataset_id).table(table_id)
        table = bigquery.Table(table_ref, schema=schema)
        try:
            self.client.create_table(table)
        except Exception as e:
            return f"ERROR : Failed to create table {table_id} in dataset {dataset_id} . Exception: {str(e)}"
        return f"Table {table_id} created in dataset {dataset_id}."
    
    def create_partitioned_table(self, dataset_id: str, table_id: str, schema: List[bigquery.SchemaField], partition_field: str) -> str:
        table_ref = self.client.dataset(dataset_id).table(table_id)
        table = bigquery.Table(table_ref, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_field
        )
        try:
            self.client.create_table(table)
        except Exception as e:
            return f"ERROR : Failed to create partitioned table {table_id} in dataset {dataset_id} . Exception: {str(e)}"
        return f"Partitioned Table {table_id} created in dataset {dataset_id}."