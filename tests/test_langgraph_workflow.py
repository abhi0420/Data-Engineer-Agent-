from langchain.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator
#from agents.connector import connector_agent
from langgraph_workflow import execute_workflow



# evals/run_evals.py
import json
from agentevals.trajectory.llm import create_trajectory_llm_as_judge

def run_workflow_and_get_trajectory(request):
    """Run your workflow and extract the trajectory"""
    from langgraph_workflow import execute_workflow
    
    result = execute_workflow(request)
    
    # Convert your state to OpenAI message format
    trajectory = []
    trajectory.append({"role": "user", "content": request})
    
    for task in result['tasks_done']:
        if task:  # Skip empty dicts
            agent = list(task.keys())[0]
            action = task[agent]
            trajectory.append({
                "role": "assistant",
                "content": f"Agent: {agent}, Action: {action}"
            })
    
    return trajectory

# Core scenarios to test
SCENARIOS = [
    "Download file.csv from bucket data_storage_1146",
    "Merge ./data/file1.csv and ./data/file2.csv on column id",
    "Create BigQuery dataset test_data in project my-project",
]

# Run evaluations
evaluator = create_trajectory_llm_as_judge(model="openai:gpt-4o-mini")

for request in SCENARIOS:
    print(f"\n{'='*50}")
    print(f"Testing: {request}")
    print('='*50)
    
    trajectory = run_workflow_and_get_trajectory(request)
    result = evaluator(outputs=trajectory)
    
    print(f"Score: {result['score']}")
    print(f"Reasoning: {result['comment']}")