from typing_extensions import TypedDict, List
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
import json
from agents.connector import connector_agent
from agents.data_transformer import smart_transformer_agent
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1000)

class State(TypedDict):
    user_request : str
    tasks_done : List[dict]
    model_responses : List[str]
    next_agent : str
    next_task : str
    has_error : bool

def semantic_search(query: str, model_responses: List[dict], top_k: int = 4):
    task_texts = [json.dumps(task) for task in model_responses]
    vectorizer = TfidfVectorizer()
    task_vectors = vectorizer.fit_transform(task_texts)  # ❌ Fails if empty
    query_vector = vectorizer.transform([query])
    
    # Find most similar
    similarities = cosine_similarity(query_vector, task_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return [model_responses[i] for i in top_indices]


def conflict_resolver(state: State) -> State:
    """Resolves conflicts in the state if any."""
    # Placeholder for conflict resolution logic
    latest_model_response = state['model_responses'][-1] if state['model_responses'] else ""
    if latest_model_response:
        print("Resolving conflict : {}".format(latest_model_response))
        relevant_tasks = semantic_search(latest_model_response, state['model_responses'])
        print("Relevant tasks for conflict resolution : ", relevant_tasks)
        prompt = f"""
        In a Data Engineering team, you are the Conflict Resolver. Your teammates are :

        1. Connector Agent : This agent can connect to GCS source & perform data operations 
        2. Smart Transformer Agent :  This agent can perform data transformations using pandas based on user instructions.

        You will be called when either of these agents encounter conflicts in their operations. You will be given the relevant previous responses from these agents. Your task is to analyze these responses, identify the conflicts, and provide a resolution for the issue. 

        Given the user task that caused the conflict, you will use the analyzed information to rewrite the task in a way that helps the agents avoid the conflict and successfully complete the task.

        User request that caused the conflict: {state['user_request']}
        Relevant responses from agents to help you resolve the conflict: {json.dumps(relevant_tasks, indent=2)}


        Your response needs to be a dictionary only in the following format:

        {{"Agent_Name" : "task"}}
        
        NOTE : Do not provide any explanations or additional text.
        """
        # Implement conflict resolution based on relevant tasks
        # For now, just log the relevant tasks
        response = model.invoke(prompt)
        print("Conflict Resolver Response : ", response.content)
        response_content = response.content
        if isinstance(response_content, str):
            response_content = ast.literal_eval(response_content)
        next_agent = list(response_content.keys())[0]
        next_task = response_content[next_agent]
        state['next_task'] = next_task
        state['next_agent'] = next_agent
        state['has_error'] = False  # Reset error flag after resolution
    return state

def delegator_logic(state: State) -> State:
    user_request = state['user_request']
    tasks_done = state.get('tasks_done', [])
    if state.get('has_error', False):
        state['next_agent'] = 'conflict_resolver'
        state['next_task'] = 'Resolve the error from previous task'
        state['has_error'] = False  # Reset flag
        return state
    # Taking only top 5 requests and responses to avoid overload

    prompt = f"""
        In a Data Engineering team, you are the Delegator. There are different agents working in your team to help with data engineering tasks. Your main task is to understand the users request and call the right agent/agents & assign the appropriate tasks to them.

        In case the user request involves multiple steps, you should break down the request into smaller tasks and assign them to the relevant agents accordingly. You should also manage the flow of information between agents to ensure that the overall task is completed.

        These are the tools provided to you:

        1. call_connector_agent Tool : This tool can connect to GCS source & perform data operations which include:

        - Downloading files from GCS into local storage
        - Uploading files to GCS from local storage
        - Deleting files from GCS
        - Creating new GCS buckets
        - Listing files in GCS buckets

        2. call_smart_transformer_agent Tool :  This tool can perform data transformations using pandas based on user instructions. Its capabilities include:
        - Previewing data files (CSV, Excel, JSON, Parquet)
        - Understanding user instructions for data transformations
        - Generating pandas code to perform the transformations
        - Saving the transformed data to specified file formats (CSV, Excel, JSON, Parquet)
        - Executing data transformations (e.g., filtering, aggregating, joining)

        3. conflict_resolver Tool : This tool helps resolve any conflicts that arise during the execution of tasks by other agents. Call this tool when you identify conflicting instructions or errors in task execution. 


        This is the user's request: {user_request}

        For this request, the progress so far has been:
        Recent Tasks Completed: {json.dumps(tasks_done[-5:], indent=2)}

        Most Recent Agent Responses (showing what actually happened):
        {json.dumps(state.get('model_responses', [])[-2:], indent=2)}

        CRITICAL: Read the agent responses. If a step is already done successfully, DO NOT assign it again.
        If all parts of the user request are complete, respond with END.

        
        Based on the user request, decide which agent to call next.

        Your response should only be a dictionary with the agent name and the task to be assigned to that agent in the following format:
        {{
            "next_agent" : "task description"
        }}

        Example : 

        {{
            "call_connector_agent" : "Extract the file submissions.csv from the GCP bucket data_storage_1146 in the project data-engineering-476308 and save it locally in data folder."
        }}

        If the user's request has been completed, respond with:

        {{
            "END" : "The task has been completed."
        }}

        Do not provide any explanations or additional text.
    """

    response = model.invoke(prompt)
    print("Delegator Response : ", response.content)
    response_content = response.content

    if isinstance(response_content, str):
        response_content = ast.literal_eval(response_content)
    next_agent = list(response_content.keys())[0]
    next_task = response_content[next_agent]
    state['next_task'] = next_task
    state['next_agent'] = next_agent
    print("Updated State : ", state)
    return state


def call_smart_transformer_agent(state : State) -> State:
    """Calls the smart transformer agent to handle data transformation tasks."""
    task = state['next_task']
    print("Calling Smart Transformer Agent to handle the task : ", task)
    response = smart_transformer_agent.invoke(
        {"messages": [{"role": "user", "content": task}]}
    )

    print("Smart Transformer Agent Response : ", response['messages'][-1].content)
    if "ERROR" in response['messages'][-1].content.upper():
        state['has_error'] = True
    state["tasks_done"].append({"call_smart_transformer_agent": state['next_task']})
    state['model_responses'].append("smart_transformer_agent" + response['messages'][-1].content)
    return state

def call_connector_agent(state : State) -> State:
    """Calls the connector agent to handle data connection tasks."""
    task = state['next_task']
    print("Calling Connector Agent to handle the task : ", task)
    response = connector_agent.invoke(
        {"messages": [{"role": "user", "content": task}]}
    )
    if "ERROR" in response['messages'][-1].content.upper():
        state['has_error'] = True
    state["tasks_done"].append({"call_connector_agent": state['next_task']})
    state['model_responses'].append("connector_agent:" + response['messages'][-1].content)
    return state



def execute_workflow():
    workflow = StateGraph(State)

    workflow.add_node("delegator_logic", delegator_logic)
    workflow.add_node("call_connector_agent", call_connector_agent)
    workflow.add_node("call_smart_transformer_agent", call_smart_transformer_agent)
    workflow.add_node("conflict_resolver", conflict_resolver) 
    workflow.add_edge(START, "delegator_logic")

    workflow.add_conditional_edges("delegator_logic", lambda state: state['next_agent'], {
        "call_connector_agent": "call_connector_agent",
        "call_smart_transformer_agent": "call_smart_transformer_agent",
        "conflict_resolver": "conflict_resolver",
        "END": END
    })
    workflow.add_edge("call_connector_agent", "delegator_logic")
    workflow.add_edge("call_smart_transformer_agent", "delegator_logic")
    workflow.add_edge("conflict_resolver", "delegator_logic")
    workflow.add_edge("delegator_logic", END)

    chain = workflow.compile()

    state = chain.invoke({"user_request" : "Extract the file submissions.csv from the GCP bucket data_storage_1146 in the project data-engineering-476308 and save it locally in data folder. Then read the file and add a new column 'Status' with values 'Success' when Target is 1 & 'Failure' when Target is 0, then save the result to updated_submissions.csv. Finally, upload the updated_submissions.csv file to the same bucket in the same project.", "tasks_done": [{}], "next_agent": "", "next_task": "", "model_responses": [], "has_error": False})

    print(state)

if __name__ == "__main__":
    execute_workflow()



