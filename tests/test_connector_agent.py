from agentevals.trajectory.match import create_trajectory_match_evaluator
from agents.connector import connector_agent
from langchain.messages import HumanMessage,AIMessage,ToolMessage, ToolCall



strict_evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")
basic_evaluator = create_trajectory_match_evaluator(trajectory_match_mode="unordered")

def test_extract_data_from_gcp_tool_call():
    result = connector_agent.invoke(
    {"messages":
    [{"role": "user", 
    "content": "Extract the file submissions.csv from the GCP bucket data_storage_1146 in the project data-engineering-476308 and save it locally.  "}]}

)
    print(result['messages'][-1].content)
    reference_trajectory = [
        HumanMessage(content="Extract the file submissions.csv from the GCP bucket data_storage_1146 in the project data-engineering-476308 and save it locally."),
        AIMessage(content="", tool_calls=[{"id": "call_1", "name": "extract_file_from_gcp", "args": {"project_id": "data-engineering-476308", "bucket_name": "data_storage_1146", "file_name": "submissions.csv"}}]),
        ToolMessage(content="Data extracted and saved locally at ./data/submissions.csv", tool_call_id="call_1"),
        AIMessage(content="Data extraction completed successfully. The file submissions.csv has been saved locally at ./data/submissions.csv.")
    ]
    evaluation = basic_evaluator(outputs=result['messages'], reference_outputs=reference_trajectory)
    print(evaluation)


if __name__ == "__main__":
    test_extract_data_from_gcp_tool_call()