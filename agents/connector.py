from langchain.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
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
        return f"ERROR : Bucket {bucket_name} does not exist in project {project_id}."

    file_present = gcp_obj.file_exists(filename)

    if file_present:
        data_path = gcp_obj.download_file(filename)
    else:
        return f"ERROR : File {filename} does not exist in bucket {bucket_name}."
    time.sleep(2)  
    return f"Data extracted and saved to {data_path}"



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
                return f"ERROR :Failed to create bucket {bucket_name} in project {project_id}."
            print(f"Bucket {bucket_name} created in project {project_id}.")
        else:
            return f"ERROR : Bucket {bucket_name} does not exist in project {project_id}."
    else:
        print("Connected to existing GCP bucket.")
    print(f"Proceeding to upload file {source_file_path} as {dest_blob_name}")
    # Upload file (works for both new and existing buckets)
    upload_status = gcp_obj.upload_file(source_file_path, dest_blob_name)
    time.sleep(2)  
    if "ERROR" in upload_status:
        return "ERROR : Upload failed : " + upload_status
    print("Upload operation completed.")
    return f"Data uploaded to {upload_status}"



connector_agent = create_agent(
        model=model,
        system_prompt="""You are a connector agent. You can connect to GCS source to perform data operations.

Examine the user's request carefully to understand what needs to be done and extract the required parameters.

Use your judgment to infer reasonable values for optional parameters from the task description and context. 

However, if critical information is not mentioned and cannot be determined from context, respond with:
"ERROR: Missing required information - [list missing parameters]. Cannot proceed."

Use the appropriate tools to complete the operation. Report any tool errors back as ERROR messages.
Once complete, provide clear status with relevant details.""",
        tools=[extract_data_from_gcp, load_data_to_gcp]
         )

if __name__ == "__main__":
    


    result = connector_agent.invoke(
    {"messages":
    [{"role": "user", 
    "content": "Extract the file submissions.csv from the GCP bucket data_storage_1146 in the project data-engineering-476308 and save it locally.  "}]}
)
    
    ai_response = result['messages'][-1].content
    print("AI Response:", ai_response)