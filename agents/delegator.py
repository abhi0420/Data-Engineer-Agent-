from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from agents.data_transformer import smart_transformer_agent
from agents.connector import connector_agent

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1000)

def call_smart_transformer_agent(task: str) -> str:
    """Calls the smart transformer agent to handle data transformation tasks."""
    print("Calling Smart Transformer Agent to handle the task : ", task)
    response = smart_transformer_agent.invoke(
        {"messages": [{"role": "user", "content": task}]}
    )

    print("Smart Transformer Agent Response : ", response['messages'][-1].content)
    return response['messages'][-1].content

def call_connector_agent(task: str) -> str:
    """Calls the connector agent to handle data connection tasks."""
    print("Calling Connector Agent to handle the task : ", task)
    response = connector_agent.invoke(
        {"messages": [{"role": "user", "content": task}]}
    )
    return response['messages'][-1].content

delegator_agent = create_agent(

    model = model,
    system_prompt= """
In a Data Engineering team, you are the Delegator. There are different agents working in your team. Your main task is to understand the users request and call the right agent/agents & assign the appropriate tasks to them.

In case the user request involves multiple steps or requires collaboration between agents, you should break down the request into smaller tasks and assign them to the relevant agents accordingly. You should also manage the flow of information between agents to ensure that the overall task is completed efficiently and accurately. 

Here are the tools provided to you::

1. call_connector_agent Tool : This tool calls the connector Agent. This Agent can connect to GCS source & perform data operations which include:

- Downloading files from GCS into local storage
- Uploading files to GCS from local storage
- Deleting files from GCS
- Creating new GCS buckets
- Listing files in GCS buckets

Example task : "Extract the file submissions.csv from the GCP bucket data_storage_1146 in the project data-engineering-476308 and save it locally"


2. call_smart_transformer_agent Tool : This tool calls the smart transformer Agent. This Agent can perform data transformations using pandas based on user instructions. Its capabilities include:
- Previewing data files (CSV, Excel, JSON, Parquet)
- Understanding user instructions for data transformations
- Generating pandas code to perform the transformations
- Saving the transformed data to specified file formats (CSV, Excel, JSON, Parquet)
- Executing data transformations (e.g., filtering, aggregating, joining)

Example task : "Read the file data.csv and add a new column 'Date' with today's date, then save the result to output.csv"

IMPORTANT: Always run a single tool at a time & wait for its completion before calling another tool.


""",
tools=[call_connector_agent, call_smart_transformer_agent]
)

if __name__ == "__main__":
    
    result = delegator_agent.invoke({
        "messages": [{
            "role": "user",
            "content": """Extract the file submissions.csv from the GCP bucket data_storage_1146 in the project data-engineering-476308 and save it locally in data folder. Then read the file and add a new column 'Status' with values 'Success' when Target is 1 & 'Failure' when Target is 0, then save the result to updated_submissions.csv. Finally, upload the updated_submissions.csv file to to a new GCP bucket processed_data_storage_1146 in the same project."""
        }]
    })