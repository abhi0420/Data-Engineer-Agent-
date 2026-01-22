from agentevals.trajectory.llm import create_trajectory_llm_as_judge
from agents.bigquery_assistant import bigquery_agent
from langchain.messages import HumanMessage,AIMessage,ToolMessage, ToolCall


llm_evaluator = create_trajectory_llm_as_judge(model="openai:gpt-4o-mini")

def run_workflow_and_get_trajectory(request):
    """Run your workflow and extract the trajectory"""
    
    result = bigquery_agent.invoke(
        {"messages": [{"role": "user", "content": request}]}
    )
    
    # The result['messages'] already contains the full trajectory
    # Convert LangChain messages to OpenAI format for the evaluator
    trajectory = []
    for msg in result['messages']:
        if hasattr(msg, 'type'):  # LangChain message object
            if msg.type.lower() == 'human':
                trajectory.append({"role": "user", "content": msg.content})
            elif msg.type.lower() == 'ai':
                entry = {"role": "assistant", "content": msg.content}
                # Include tool calls if present
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    entry["tool_calls"] = [
                        {"name": tc["name"], "args": tc["args"]} 
                        for tc in msg.tool_calls
                    ]
                trajectory.append(entry)
            elif msg.type == 'tool':
                trajectory.append({
                    "role": "tool", 
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id
                })
        else:  # Already a dict
            trajectory.append(msg)
    
    return trajectory

def test_create_bigquery_dataset():
    message = "Create a BigQuery dataset named test_data in the project data-engineering-476308."
    trajectory = run_workflow_and_get_trajectory(message)
    test_result = llm_evaluator(outputs=trajectory)
    # print(f"Score: {test_result['score']}")
    # print(f"Reasoning: {test_result['comment']}")
    return test_result

def test_run_bigquery_query():
    message = "Run a BigQuery SQL query to select all records from the table submissions in the dataset test_data and return the first 5 rows."
    trajectory = run_workflow_and_get_trajectory(message)
    test_result = llm_evaluator(outputs=trajectory)
    # print(f"Score: {test_result['score']}")
    # print(f"Reasoning: {test_result['comment']}")
    return test_result

def test_create_bigquery_table():
    message = "Create a BigQuery table named submissions in the dataset test_data with columns id (INTEGER), name (STRING), and score (FLOAT)."
    trajectory = run_workflow_and_get_trajectory(message)
    test_result = llm_evaluator(outputs=trajectory)
    # print(f"Score: {test_result['score']}")
    # print(f"Reasoning: {test_result['comment']}")
    return test_result

def insert_data_into_bigquery_table():
    message = "Insert the following data into the BigQuery table submissions in the dataset test_data: (1, 'Alice', 95.5), (2, 'Bob', 89.0), (3, 'Charlie', 92.0)."
    trajectory = run_workflow_and_get_trajectory(message)
    test_result = llm_evaluator(outputs=trajectory)
    # print(f"Score: {test_result['score']}")
    # print(f"Reasoning: {test_result['comment']}")
    return test_result


def perform_all_tests():
    test_results = []
    
    print("Running test: Create BigQuery Dataset")
    test_results.append(test_create_bigquery_dataset())
    
    print("Running test: Run BigQuery Query")
    test_results.append(test_run_bigquery_query())

    print("Running test: Create BigQuery Table")
    test_results.append(test_create_bigquery_table())

    print("Running test: Insert Data into BigQuery Table")
    test_results.append(insert_data_into_bigquery_table())

    print("\nSummary of Test Results:")
    for i, result in enumerate(test_results, 1):
        print(f"------- Test {i} -------- \n Score = {result['score']}, Reasoning = {result['comment']}, \n Result : {'Pass' if result['score'] is True else 'Fail'}")


if __name__ == "__main__":
    perform_all_tests()