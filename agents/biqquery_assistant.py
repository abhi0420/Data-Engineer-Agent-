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
    """Executes a BigQuery SQL query and returns the results as a string."""
    
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
    """Creates a BigQuery dataset in the specified project."""
    
    bq_obj = BigQuerySource(project_id)
    print("BigQuery Client Initialized.")
    
    result = bq_obj.create_dataset(dataset_id, location)
    time.sleep(2)  
    return result

@tool
def create_bigquery_table(project_id: str, dataset_id: str, table_id: str, schema: str) -> str:
    """Creates a BigQuery table in the specified dataset with the given schema."""
    
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
def insert_rows_into_bigquery(project_id: str, dataset_id: str, table_id:
    str, rows: str) -> str:
        """Inserts rows into a BigQuery table."""
        
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

        Examine the task given by the user, understand what needs to be done and what tools are required & what parameters are required for the tool.

        In case of critical information is missing, do not make assumptions, respond with:

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
    "content": "Create a table in  'sales_data' in project 'data-engineering-476308' located in 'US'."}]
    }
    )
    print(result)