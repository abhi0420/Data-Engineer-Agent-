from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from bigquery_source import BigQuerySource
import time
from dotenv import load_dotenv
import os
from google.cloud import bigquery
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1000)

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
    except Exception as e:
        return f"ERROR : Failed to execute query. Exception: {str(e)}"
    
    time.sleep(2)  
    return f"Query executed successfully. Results:\n{result_str}"

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
    return result

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
        schema_list = eval(schema)  # Expecting schema to be a string representation of a list of dicts
        for field in schema_list:
            schema_fields.append(bigquery.SchemaField(name=field['name'], field_type=field['type'], mode=field.get('mode', 'NULLABLE')))
    except Exception as e:
        return f"ERROR : Invalid schema format. Exception: {str(e)}"
    
    result = bq_obj.create_table(dataset_id, table_id, schema_fields)
    time.sleep(2)  
    return result

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
        rows_list = eval(rows)  # Expecting rows to be a string representation of a list of dicts
    except Exception as e:
        return f"ERROR : Invalid rows format. Exception: {str(e)}"
    
    result = bq_obj.insert_rows(dataset_id, table_id, rows_list)
    time.sleep(2)  
    return result


bigquery_agent = create_agent(
        model=model,
        system_prompt="""You are a BigQuery assistant agent. You can help handle tasks related to GCP Bigquery. 

Examine the task given by the user, understand what needs to be done and select the appropriate tool.

Go through the Tool Parameters. Optional parameters have defaults - DO NOT ask for them if not provided.
If REQUIRED parameters are missing and cannot be inferred, respond with:

"ERROR: Missing required information - [list missing parameters]. Cannot proceed."

Use the appropriate tools to complete the operation. Report any tool errors back as ERROR messages.
Once complete, provide clear status with relevant details.
       """,
        tools=[execute_bigquery_query, create_bigquery_dataset, create_bigquery_table, insert_rows_into_bigquery]
    )

if __name__ == "__main__":
    result = bigquery_agent.invoke(
    {"messages":
    [{"role": "user",
    "content": "Create a table 'sales-01' in 'sales_data' dataset with schema [{'name': 'product', 'type': 'STRING'}, {'name': 'quantity', 'type': 'INTEGER'}] in project 'data-engineering-476308' located in 'US'."}]
    }
    )
    print(result)