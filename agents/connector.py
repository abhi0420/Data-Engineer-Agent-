from langchain.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.callbacks import get_openai_callback
from gcs_source import GCPSource
import time
import os
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1000)


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
def load_data_to_gcp(project_id: str, bucket_name: str,  source_file_path: str, dest_blob_name: str, create_new_bucket: bool = False) -> str:
    """Uploads data to the given GCP bucket from a local file."""
    
    # First, try to connect to existing bucket
    gcp_obj = GCPSource(project_id, bucket_name)
    bucket_exists = gcp_obj.bucket_exists()
    if not os.path.exists(source_file_path):
        print("Path does not exist locally, prepending ./data/")
        source_file_path = "./data/" + source_file_path

    if not bucket_exists:
        if create_new_bucket:
            # Create bucket and get new instance
            print(f"Bucket {bucket_name} does not exist. Creating new bucket in project {project_id}.")
            gcp_obj = GCPSource.create_bucket(project_id, bucket_name)
            if not gcp_obj:
                return f"""ERROR: Bucket Creation Failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Failed to create bucket '{bucket_name}' in project '{project_id}'.
"""
            print(f"Bucket {bucket_name} created in project {project_id}.")
        else:
            return f"""ERROR: Bucket Not Found
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Bucket '{bucket_name}' does not exist in project '{project_id}'.
💡 Tip: Set create_new_bucket=True to create it.
"""
    else:
        print("Connected to existing GCP bucket.")
    print(f"Proceeding to upload file {source_file_path} as {dest_blob_name}")
    # Upload file (works for both new and existing buckets)
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

If critical information is missing and cannot be inferred from the task, respond starting with "ERROR:" and clearly state what information is needed.

Report any tool errors back starting with ERROR.""",
        tools=[extract_data_from_gcp, load_data_to_gcp]
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
    

