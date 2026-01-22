from agentevals.trajectory.llm import create_trajectory_llm_as_judge
from agents.data_transformer import smart_transformer_agent
from langchain.messages import HumanMessage,AIMessage,ToolMessage, ToolCall


llm_evaluator = create_trajectory_llm_as_judge(model="openai:gpt-4o-mini")

def run_workflow_and_get_trajectory(request):
    """Run your workflow and extract the trajectory"""
    
    result = smart_transformer_agent.invoke(
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

def test_data_transformation():
    message = "Preview ./data/submissions.csv, add a new column 'status' with value 'complete' for all rows, and save to ./data/submissions_transformed.csv"
    trajectory = run_workflow_and_get_trajectory(message)
    test_result = llm_evaluator(outputs=trajectory)
    # print(f"Score: {test_result['score']}")
    # print(f"Reasoning: {test_result['comment']}")
    return test_result


def test_data_preview():
    message = "Preview the first 10 rows of ./data/submissions.csv"
    trajectory = run_workflow_and_get_trajectory(message)
    test_result = llm_evaluator(outputs=trajectory)
    # print(f"Score: {test_result['score']}")
    # print(f"Reasoning: {test_result['comment']}")
    return test_result

def test_unsupported_file_format():
    message = "Preview the file ./data/submissions.xml"
    trajectory = run_workflow_and_get_trajectory(message)
    test_result = llm_evaluator(outputs=trajectory)
    # print(f"Score: {test_result['score']}")
    # print(f"Reasoning: {test_result['comment']}")
    return test_result

def perform_all_tests():
    test_results = []

    print("Testing Data Transformation Workflow:")
    test_results.append(test_data_transformation())
    print("\nTesting Data Preview Workflow:")
    test_results.append(test_data_preview())
    print("\nTesting Unsupported File Format Workflow:")
    test_results.append(test_unsupported_file_format())

    print("\nSummary of Test Results:")
    for i, result in enumerate(test_results, 1):
        print(f"Test {i}: Score: {result['score']},\n Reasoning: {result['comment']}, \n Result : {'Pass' if result['score'] is True else 'Fail'}")
        
if __name__ == "__main__":  
    perform_all_tests()
