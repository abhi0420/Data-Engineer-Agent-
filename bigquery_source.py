from google.cloud import bigquery
from typing import List, Dict, Optional, Any
from google.oauth2 import service_account
from dotenv import load_dotenv
import os 
import pandas as pd

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
    
    def insert_rows(self, dataset_id: str, table_id: str, rows: List[Dict[str, Any]]) -> str:
        table_ref = self.client.dataset(dataset_id).table(table_id)
        try:
            errors = self.client.insert_rows_json(table_ref, rows)
            if errors:
                return f"ERROR : Failed to insert rows into table {table_id} . Errors: {errors}"
        except Exception as e:
            return f"ERROR : Failed to insert rows into table {table_id} . Exception: {str(e)}"
        return f"Rows inserted into table {table_id}."
    
    def query(self, query: str) -> pd.DataFrame:
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            df = results.to_dataframe()
            return df
        except Exception as e:
            print(f"ERROR : Failed to execute query. Exception: {str(e)}")
            return pd.DataFrame()   
        
    def delete_table(self, dataset_id: str, table_id: str) -> str:
        table_ref = self.client.dataset(dataset_id).table(table_id)
        try:
            self.client.delete_table(table_ref)
        except Exception as e:
            return f"ERROR : Failed to delete table {table_id} in dataset {dataset_id} . Exception: {str(e)}"
        return f"Table {table_id} deleted from dataset {dataset_id}."
    
    def get_schema_of_df(self, df : pd.DataFrame) -> List[bigquery.SchemaField]:
        schema = []
        mapping = {
            'object': 'STRING', 'int64': 'INTEGER', 'float64': 'FLOAT', 'bool': 'BOOLEAN', 'datetime64[ns]': 'TIMESTAMP', 'datetime' : 'TIMESTAMP', 'json' : 'JSON'}
        for col in df.columns:
            col_type = str(df[col].dtype).lower()
            bq_type = mapping.get(col_type, 'STRING')
            schema.append(bigquery.SchemaField(name=col, field_type=bq_type, mode='NULLABLE'))
        return schema
    
    def load_data_from_gcs(self, dataset_id: str, table_id: str, gcs_uri: str, file_format: str = "CSV", skip_leading_rows: int = 0, schema : List[bigquery.SchemaField] = None, field_delimiter: str = ",") -> str:

        table_ref = self.client.dataset(dataset_id).table(table_id)

        job_config = bigquery.LoadJobConfig(
                    schema=schema,
                    skip_leading_rows=skip_leading_rows,
                    source_format=bigquery.SourceFormat.CSV,
                    )
        
        if file_format.upper() == "CSV":
            job_config.source_format = bigquery.SourceFormat.CSV
            job_config.skip_leading_rows = skip_leading_rows
            job_config.field_delimiter = field_delimiter
        elif file_format.upper() == "JSON":
            job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        else:
            return f"ERROR : Unsupported file format {file_format}."
        
        
        load_job = self.client.load_table_from_uri(
            gcs_uri,
            table_ref,
            job_config=job_config
        )
        
        try:
            load_job.result()
        except Exception as e:
            return f"ERROR : Failed to load data into table {table_id} from {gcs_uri} . Exception: {str(e)}"
        
        return f"Data loaded into table {table_id} from {gcs_uri}."
    

if __name__ == "__main__":
    bq_source = BigQuerySource(project_id="data-engineering-476308")
    print(bq_source.list_datasets())
    #print(bq_source.create_dataset("test_dataset"))
    #print(bq_source.create_table(
    #    dataset_id="test_dataset",
    #    table_id="test_table",
    #    schema=[bigquery.SchemaField("name", "STRING"), bigquery.SchemaField("age", "INTEGER")]   ))
    #print(bq_source.insert_rows(
    #    dataset_id="test_dataset",
    #    table_id="test_table",
    #    rows=[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    #    ))
    #print(bq_source.query("SELECT * FROM `data-engineering-476308.test_dataset.test_table` LIMIT 10"))
    #print(bq_source.load_data_from_gcs(
    #     dataset_id="test_dataset",
    #     table_id="merged_wb",
    #     gcs_uri="gs://merged_data_storage_1146/merged_wb.csv",
    #     file_format="CSV",
    #     skip_leading_rows=1,
    #     schema=[bigquery.SchemaField("E_id", "STRING"), bigquery.SchemaField("E_name", "STRING"), bigquery.SchemaField("Salary", "INTEGER")],
    #     field_delimiter=","
    # ))
    #print(bq_source.query("SELECT * FROM `data-engineering-476308.test_dataset.merged_wb` LIMIT 10"))
    print(bq_source.delete_table(
        dataset_id="test_dataset",
        table_id="merged_wb"
    ))