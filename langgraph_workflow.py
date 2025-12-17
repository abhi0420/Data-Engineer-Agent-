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

def semantic_search(query: str, model_responses: List[dict], top_k: int = 2):
    print("Performing semantic search for resolving conflict : ", query)
    task_texts = [json.dumps(task) for task in model_responses]
    vectorizer = TfidfVectorizer()
    task_vectors = vectorizer.fit_transform(task_texts)  # ❌ Fails if empty
    query_vector = vectorizer.transform([query])
    
    # Find most similar
    similarities = cosine_similarity(query_vector, task_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return [model_responses[i] for i in top_indices]

def resolution_flow(state: State) -> str:
    "checks if workflow should end"
    if state['next_agent'] == 'END':
        return "END"
    elif "connector" in state['next_agent'].lower():
        return "call_connector_agent"
    elif "transformer" in state['next_agent'].lower():
        return "call_smart_transformer_agent"
    else:
        return "delegator_logic"

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

        You will be called when either of these agents encounter conflicts in their operations. You will be given the relevant previous responses from these agents.

        Given the Main task, sub task that caused the conflict & the conflict, you will use the analyzed information to rewrite the Sub task where the conflict arose in a way that helps the agents avoid the conflict and successfully complete the task. In case the provided info is insufficient to resolve conflict, you will end the workflow.

        Main task: {state['user_request']}
        Sub Task where conflict arose: {state['next_task']}
        Conflict : {latest_model_response}

        Relevant responses from agents to help you: {json.dumps(relevant_tasks, indent=2)}


        Your response needs to be a dictionary only in the following format:

        {{"Agent_Name" : "task"}}

        IMPORTANT : In case you find the user provided info is actually incorrect or insufficient to resolve the conflict, respond with :

        {{"END" : "End the workflow as the conflict could not be resolved"}}

        NOTE : Do not provide any explanations or additional text.
        """
        # Implement conflict resolution based on relevant tasks
        # For now, just log the relevant tasks
        try:    
            response = model.invoke(prompt)
            print("Conflict Resolver Response : ", response.content)
            response_content = response.content
            if isinstance(response_content, str):
                response_content = ast.literal_eval(response_content)
            next_agent = list(response_content.keys())[0]
            next_task = response_content[next_agent]
            print("Conflict Resolver recommends ", next_agent, " Task : ", next_task)
            state['next_task'] = next_task
            state['next_agent'] = next_agent
            state['has_error'] = False
        except Exception as e:
            print("Conflict Resolver encountered an error: ", str(e))
            state['next_agent'] = 'END'
            state['next_task'] = 'End the workflow as the conflict could not be resolved'
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

        In case the user request involves multiple steps, you should break down the request into smaller tasks and assign them to the relevant agents accordingly. 

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
    workflow.add_conditional_edges("conflict_resolver", resolution_flow, {
        "delegator_logic": "delegator_logic",
        "call_connector_agent": "call_connector_agent",
        "call_smart_transformer_agent": "call_smart_transformer_agent",
        "END": END
    })
    workflow.add_edge("call_connector_agent", "delegator_logic")
    workflow.add_edge("call_smart_transformer_agent", "delegator_logic")
    workflow.add_edge("delegator_logic", END)

    chain = workflow.compile()

    state = chain.invoke({"user_request" : "Read the files wb1.csv & wb2.csv from the bucket data_storage_1146 in project data-engineering-476308, merge them on the common column. Then save the result in a new file. Finally, upload this new file to a new bucket merged_data_storage_1146 with the same filename.", "tasks_done": [{}], "next_agent": "", "next_task": "", "model_responses": [], "has_error": False})

    print(state)

if __name__ == "__main__":
    execute_workflow()



