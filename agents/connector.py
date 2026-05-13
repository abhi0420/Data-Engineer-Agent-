from langchain.messages import SystemMessage
from config.model_config import get_llm
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.callbacks import get_openai_callback
from gcs_source import GCPSource
import time
import os
from dotenv import load_dotenv

load_dotenv()
model = get_llm()


@tool
def extract_data_from_gcp(project_id : str, bucket_name : str, filename : str) -> str:
    """Extracts data from the given GCP bucket to a local file."""

    gcp_obj = GCPSource(project_id, bucket_name)
    print("GCP Storage Client Initialized.")

    bucket_exists = gcp_obj.bucket_exists()
    if not bucket_exists:
        return f"""ERROR: Bucket Not Found
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Bucket '{bucket_name}' does not exist in project '{project_id}'.
"""

    file_present = gcp_obj.file_exists(filename)

    if file_present:
        data_path = gcp_obj.download_file(filename)
    else:
        return f"""ERROR: File Not Found
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ File '{filename}' does not exist in bucket '{bucket_name}'.
"""
    time.sleep(2)  
    return f"""✅ Download Complete
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 File: {filename}
📦 Source: gs://{bucket_name}/{filename}
💾 Saved to: {data_path}
"""



@tool
def create_gcs_bucket(project_id: str, bucket_name: str, location: str = "US") -> str:
    """Creates a new GCS bucket in the given project."""
    try:
        gcp_obj = GCPSource.create_bucket(project_id, bucket_name, location=location)
        if not gcp_obj:
            return f"""ERROR: Bucket Creation Failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Failed to create bucket '{bucket_name}' in project '{project_id}'.
"""
        return f"""✅ Bucket Created
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Bucket: {bucket_name}
📍 Project: {project_id}
🌍 Location: {location}
"""
    except Exception as e:
        return f"ERROR: Bucket Creation Failed - {str(e)}"


@tool
def load_data_to_gcp(project_id: str, bucket_name: str, source_file_path: str, dest_blob_name: str) -> str:
    """Uploads a local file to an existing GCS bucket. Use create_gcs_bucket first if the bucket does not exist."""

    gcp_obj = GCPSource(project_id, bucket_name)
    if not os.path.exists(source_file_path):
        print("Path does not exist locally, prepending ./data/")
        source_file_path = "./data/" + source_file_path

    if not gcp_obj.bucket_exists():
        return f"""ERROR: Bucket Not Found
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Bucket '{bucket_name}' does not exist in project '{project_id}'.
💡 Tip: Use the create_gcs_bucket tool to create it first.
"""

    print("Connected to existing GCP bucket.")
    print(f"Proceeding to upload file {source_file_path} as {dest_blob_name}")
    upload_status = gcp_obj.upload_file(source_file_path, dest_blob_name)
    time.sleep(2)
    if "ERROR" in upload_status:
        return f"""ERROR: Upload Failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ {upload_status}
"""
    print("Upload operation completed.")
    return f"""✅ Upload Complete
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 File: {dest_blob_name}
📦 Destination: gs://{bucket_name}/{dest_blob_name}
📍 Source: {source_file_path}
"""



connector_agent = create_agent(
        model=model,
        system_prompt="""You are a connector agent for GCS (Google Cloud Storage) operations.

Your job is to extract parameters from the task description and call the appropriate tool. Each tool has a docstring that describes its required and optional parameters - read them carefully.

If critical information is missing and cannot be inferred from the task, respond starting with "ERROR:" and clearly state what the error was & what information is needed.

Report any tool errors back starting with ERROR and include the error message in your response.
""",
        tools=[extract_data_from_gcp, load_data_to_gcp, create_gcs_bucket]
         )

if __name__ == "__main__":
    

    with get_openai_callback() as cb:
        result = connector_agent.invoke(
        {"messages":
        [{"role": "user", 
        "content": "Extract the file submissions.csv from the GCP bucket data_storage_1146 in the project data-engineering-476308 and save it locally.  "}]}
    )
        
        ai_response = result['messages'][-1].content
        print("AI Response:", ai_response)
        print("\nToken Usage:", cb.total_tokens)
        print("Total Cost (USD): $", cb.total_cost)
    

