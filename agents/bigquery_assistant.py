from config.model_config import get_llm
from langchain.agents import create_agent
from langchain.tools import tool
from bigquery_source import BigQuerySource
import time
from dotenv import load_dotenv
import os
import ast
import pandas as pd
from google.cloud import bigquery
import warnings
from langchain_community.callbacks import get_openai_callback

warnings.filterwarnings("ignore")

load_dotenv()

model = get_llm()




@tool
def infer_schema_from_file(file_path: str) -> str:
    """Infers the BigQuery schema from a local or GCS file (CSV, JSON, Parquet) by reading its headers and dtypes.
    Use this tool before creating a table or loading data when the schema is not explicitly provided.

    Required parameters:
        - file_path: Local path (e.g., ./data/merged_wb.csv) or GCS URI (e.g., gs://bucket/file.csv)
    """
    DTYPE_MAP = {
        "int64": "INTEGER",
        "int32": "INTEGER",
        "float64": "FLOAT",
        "float32": "FLOAT",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
        "object": "STRING",
    }

    temp_file = None

    try:
        if file_path.startswith("gs://"):
            from google.cloud import storage
            from google.oauth2 import service_account
            import tempfile

            # Parse gs://bucket/blob
            path_without_scheme = file_path[5:]
            bucket_name, blob_name = path_without_scheme.split("/", 1)
            ext = os.path.splitext(blob_name)[1].lower()

            creds = service_account.Credentials.from_service_account_file(
                os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            )
            client = storage.Client(credentials=creds)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                return f"ERROR: GCS file '{file_path}' not found."

            temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            blob.download_to_filename(temp_file.name)
            local_path = temp_file.name
        else:
            if not os.path.exists(file_path):
                return f"ERROR: File '{file_path}' not found."
            ext = os.path.splitext(file_path)[1].lower()
            local_path = file_path

        if ext == ".csv":
            df = pd.read_csv(local_path, nrows=1)
        elif ext == ".json":
            df = pd.read_json(local_path, lines=True, nrows=1)
        elif ext == ".parquet":
            df = pd.read_parquet(local_path).head(1)
        else:
            return f"ERROR: Unsupported file format '{ext}'. Supported: csv, json, parquet."

        schema = [
            {"name": col, "type": DTYPE_MAP.get(str(dtype), "STRING")}
            for col, dtype in df.dtypes.items()
        ]

    except Exception as e:
        return f"ERROR: Could not read file - {str(e)}"
    finally:
        if temp_file:
            os.unlink(temp_file.name)

    return f"""✅ Schema Inferred Successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 File: {file_path}
📋 Columns: {len(schema)}
📝 Schema:
{schema}
"""


@tool
def execute_bigquery_query(project_id: str, query: str) -> str: 
    """Executes a BigQuery SQL query and returns the results as a string.
    
    Required parameters:
        - project_id: The GCP project ID
        - query: The SQL query to execute
    """
    
    bq_obj = BigQuerySource(project_id)
    print("BigQuery Client Initialized.")
    
    try:
        df = bq_obj.query(query)
        result_str = df.to_string(index=False)
        row_count = len(df)
    except Exception as e:
        return f"""ERROR: Query Execution Failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ {str(e)}
📝 Query: {query[:100]}{'...' if len(query) > 100 else ''}
"""
    
    time.sleep(2)  
    return f"""✅ Query Executed Successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Rows returned: {row_count}
📋 Results:
{result_str}
"""

@tool
def create_bigquery_dataset(project_id: str, dataset_id: str, location: str = "US") -> str:
    """Creates a BigQuery dataset in the specified project.
    
    Required parameters:
        - project_id: The GCP project ID
        - dataset_id: The name of the dataset to create
    
    Optional parameters:
        - location: Dataset location (default: "US")
    """
    
    bq_obj = BigQuerySource(project_id)
    print("BigQuery Client Initialized.")
    
    result = bq_obj.create_dataset(dataset_id, location)
    time.sleep(2)
    
    if "ERROR" in result.upper() or "error" in result.lower():
        return f"""ERROR: Dataset Creation Failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ {result}
"""
    
    return f"""✅ Dataset Created Successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 Dataset: {dataset_id}
🌍 Location: {location}
🔗 Project: {project_id}
"""

@tool
def create_view(project_id: str, dataset_id: str, view_id: str, query: str) -> str:
    """
    Creates a BigQuery view in the specified dataset.
    """
    bq_obj = BigQuerySource(project_id)
    print("BigQuery Client Initialized.")
    
    result = bq_obj.create_view(dataset_id, view_id, query)
    time.sleep(2)
    
    if "ERROR" in str(result).upper():
        return f"""ERROR: View Creation Failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ {result}
"""
    
    return f"""✅ View Created Successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👁️ View: {view_id}
📁 Dataset: {dataset_id}
🔗 Project: {project_id}
"""

@tool
def create_partitioned_table(project_id: str, dataset_id: str, table_id: str, schema: str, partition_field: str) -> str:
    """Creates a partitioned BigQuery table in the specified dataset with the given schema.
    
    Required parameters:
        - project_id: The GCP project ID
        - dataset_id: The dataset containing the table
        - table_id: The name of the table to create
        - schema: List of dicts with 'name', 'type', and optional 'mode' (default: NULLABLE)
                  Example: [{'name': 'col1', 'type': 'STRING'}, {'name': 'col2', 'type': 'INTEGER'}]
        - partition_field: The field to partition the table on
    """
    
    bq_obj = BigQuerySource(project_id)
    print("BigQuery Client Initialized.")
    
    # Convert schema string to list of bigquery.SchemaField
    schema_fields = []
    try:
        schema_list = ast.literal_eval(schema) if isinstance(schema, str) else schema
        for field in schema_list:
            schema_fields.append(bigquery.SchemaField(name=field['name'], field_type=field['type'], mode=field.get('mode', 'NULLABLE')))
    except (ValueError, SyntaxError) as e:
        return f"ERROR : Invalid schema format. Expected list of dicts. Exception: {str(e)}"
    
    result = bq_obj.create_partitioned_table(dataset_id, table_id, schema_fields, partition_field)
    time.sleep(2)
    
    if "ERROR" in str(result).upper():
        return f"""ERROR: Partitioned Table Creation Failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ {result}
"""
    
    return f"""✅ Partitioned Table Created Successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Table: {table_id}
📁 Dataset: {dataset_id}
📅 Partition Field: {partition_field}
🔗 Project: {project_id}
"""



@tool
def create_bigquery_table(project_id: str, dataset_id: str, table_id: str, schema: str) -> str:
    """Creates a BigQuery table in the specified dataset with the given schema.
    
    Required parameters:
        - project_id: The GCP project ID
        - dataset_id: The dataset containing the table
        - table_id: The name of the table to create
        - schema: List of dicts with 'name', 'type', and optional 'mode' (default: NULLABLE)
                  Example: [{'name': 'col1', 'type': 'STRING'}, {'name': 'col2', 'type': 'INTEGER'}]
    """
    
    bq_obj = BigQuerySource(project_id)
    print("BigQuery Client Initialized.")
    
    # Convert schema string to list of bigquery.SchemaField
    schema_fields = []
    try:
        schema_list = ast.literal_eval(schema) if isinstance(schema, str) else schema
        for field in schema_list:
            schema_fields.append(bigquery.SchemaField(name=field['name'], field_type=field['type'], mode=field.get('mode', 'NULLABLE')))
    except (ValueError, SyntaxError) as e:
        return f"ERROR : Invalid schema format. Expected list of dicts. Exception: {str(e)}"
    
    result = bq_obj.create_table(dataset_id, table_id, schema_fields)
    time.sleep(2)
    
    if "ERROR" in str(result).upper():
        return f"""ERROR: Table Creation Failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ {result}
"""
    
    return f"""✅ Table Created Successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Table: {table_id}
📁 Dataset: {dataset_id}
🔗 Project: {project_id}
📋 Schema: {len(schema_fields)} columns
"""

@tool
def load_table_from_gcs(project_id: str, dataset_id: str, table_id: str, source_uri: str, schema: str = None, file_format: str = "CSV") -> str:
    """Loads data into a BigQuery table from a GCS URI.
    
    Required parameters:
        - project_id: The GCP project ID
        - dataset_id: The dataset containing the table
        - table_id: The target table name
        - source_uri: The GCS URI of the source file (e.g., gs://bucket_name/file.csv)

    Optional parameters:
        - schema: List of dicts with 'name', 'type', and optional 'mode'. If omitted, BigQuery will auto-detect the schema.
                  Example: [{"name": "ID", "type": "STRING"}, {"name": "Name", "type": "STRING"}]
        - file_format: The format of the source file (e.g., CSV, JSON, PARQUET). Default is CSV.
    """
    
    bq_obj = BigQuerySource(project_id)
    print("BigQuery Client Initialized.")
    
    # Convert schema string to list of bigquery.SchemaField, or None for autodetect
    schema_fields = None
    if schema is not None:
        schema_fields = []
        try:
            schema_list = ast.literal_eval(schema) if isinstance(schema, str) else schema
            for field in schema_list:
                schema_fields.append(bigquery.SchemaField(name=field['name'], field_type=field['type'], mode=field.get('mode', 'NULLABLE')))
        except (ValueError, SyntaxError) as e:
            return f"ERROR : Invalid schema format. Expected list of dicts. Exception: {str(e)}"
    
    result = bq_obj.load_data_from_gcs(dataset_id, table_id, source_uri, file_format, 1 if schema_fields is not None else 0, schema_fields)
    time.sleep(2)
    
    if "ERROR" in str(result).upper():
        return f"""ERROR: Data Load Failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ {result}
"""
    
    return f"""✅ Data Loaded Successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Table: {dataset_id}.{table_id}
📦 Source: {source_uri}
📄 Format: {file_format}
🔗 Project: {project_id}
"""

@tool
def insert_rows_into_bigquery(project_id: str, dataset_id: str, table_id: str, rows: str) -> str:
    """Inserts rows into a BigQuery table.
    
    Required parameters:
        - project_id: The GCP project ID
        - dataset_id: The dataset containing the table
        - table_id: The target table name
        - rows: List of dicts representing rows to insert
                Example: [{'col1': 'value1', 'col2': 123}]
    """
    
    bq_obj = BigQuerySource(project_id)
    print("BigQuery Client Initialized.")
    
    # Convert rows string to list of dicts
    try:
        rows_list = ast.literal_eval(rows) if isinstance(rows, str) else rows
    except (ValueError, SyntaxError) as e:
        return f"ERROR : Invalid rows format. Expected list of dicts. Exception: {str(e)}"
    
    result = bq_obj.insert_rows(dataset_id, table_id, rows_list)
    time.sleep(2)
    
    if "ERROR" in str(result).upper():
        return f"""ERROR: Row Insertion Failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ {result}
"""
    
    return f"""✅ Rows Inserted Successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Table: {dataset_id}.{table_id}
📝 Rows inserted: {len(rows_list)}
🔗 Project: {project_id}
"""
@tool
def delete_bigquery_table(project_id: str, dataset_id: str, table_id: str) -> str:
    """Deletes a BigQuery table.
    
    Required parameters:
        - project_id: The GCP project ID
        - dataset_id: The dataset containing the table
        - table_id: The name of the table to delete
    """
    
    bq_obj = BigQuerySource(project_id)
    print("BigQuery Client Initialized.")
    
    result = bq_obj.delete_table(dataset_id, table_id)
    time.sleep(2)
    
    if "ERROR" in str(result).upper():
        return f"""ERROR: Table Deletion Failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ {result}
"""
    
    return f"""✅ Table Deleted Successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Table: {dataset_id}.{table_id}
🔗 Project: {project_id}
"""

bigquery_agent = create_agent(
        model=model,
        system_prompt="""You are a BigQuery assistant agent. You can help handle tasks related to GCP Bigquery. 

Examine the task given by the user, understand what needs to be done and select the appropriate tool.

Go through the Tool Parameters. Optional parameters have defaults - DO NOT ask for them if not provided.
If REQUIRED parameters are missing and cannot be inferred, respond with:

"ERROR: The task requires the parameters - [list of all parameters]. Cannot proceed.". Mentioning ALL parameters is important so conlfict can be resolved in one go.

Use the appropriate tools to complete the operation. Report any tool errors back as ERROR messages.
Once complete, provide clear status with relevant details.
       """,
        tools=[execute_bigquery_query, create_bigquery_dataset, create_bigquery_table, insert_rows_into_bigquery, load_table_from_gcs, delete_bigquery_table, create_view, create_partitioned_table, infer_schema_from_file]
    )

if __name__ == "__main__":
    with get_openai_callback() as cb:
        result = bigquery_agent.invoke(
        {"messages":
        [{"role": "user",
        "content": "Create a table 'sales-01' in 'sales_data' dataset with schema [{'name': 'product', 'type': 'STRING'}, {'name': 'quantity', 'type': 'INTEGER'}] in project 'data-engineering-476308' located in 'US'."}]
        }
        )
        print(result)
        print("Total Tokens Used in BigQuery Agent: ", cb.total_tokens)
        print("Total Cost (USD): $", cb.total_cost)