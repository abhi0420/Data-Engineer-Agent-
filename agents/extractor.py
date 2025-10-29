from langchain.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from sources import GCPSource

@tool
def extract_data_from_gcp(project_id : str, bucket_name : str, filename : str) -> str:
    """Extracts data from the given GCP bucket to a local file."""

    gcp_obj = GCPSource(project_id, bucket_name)
    print("GCP Storage Client Initialized.")

    bucket_exists = gcp_obj.bucket_exists()
    if not bucket_exists:
        return f"Bucket {bucket_name} does not exist in project {project_id}."

    file_present = gcp_obj.file_exists(filename)

    if file_present:
        data_path = gcp_obj.download_file(filename)
    else:
        return f"File {filename} does not exist in bucket {bucket_name}."

    return f"Data extracted and saved to {data_path}"


if __name__ == "__main__":
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1000)

    extractor_agent = create_agent(
        model=model,
        system_prompt="You are a data extraction agent. Your job is to extract the required files to the local filesystem from the mentioned GCP location.",
        tools=[extract_data_from_gcp]
         )

    result = extractor_agent.invoke(
    {"messages":
    [{"role": "user", 
    "content": "Extract the file datas.csv from the GCP bucket data_stora_16 in the project data-engineering-43333 and save it locally."}]}
)
    
    ai_response = result['messages'][-1].content
    print("AI Response:", ai_response)