from typing_extensions import TypedDict, List
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
import json
from agents.connector import connector_agent
from agents.data_transformer import smart_transformer_agent
from agents.bigquery_assistant import bigquery_agent
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1000)

class State(TypedDict):
    user_request : str
    tasks_done : List[dict]
    model_responses : List[str]
    next_agent : str
    next_task : str
    has_error : bool
    error_details : str
    current_error: str

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
        return "connector_agent"
    elif "transformer" in state['next_agent'].lower():
        return "smart_transformer_agent"
    elif "bigquery" in state['next_agent'].lower():
        return "bigquery_agent"
    else:
        return "delegator_logic"

def conflict_resolver(state: State) -> State:
    """Resolves conflicts in the state if any."""
    # Placeholder for conflict resolution logic
    latest_model_response = state['model_responses'][-1] if state['model_responses'] else ""
    if latest_model_response:
        print("Resolving conflict : {}".format(state['error_details']))
        relevant_tasks = semantic_search(state['error_details'], state['model_responses'])
        print("Relevant tasks for conflict resolution : ", relevant_tasks)
        prompt = f"""
        In a Data Engineering team, you are the Conflict Resolver. Your teammates are :

        1. Connector Agent : Can connect to GCS source & perform data operations 
        2. Smart Transformer Agent : Can perform data transformations using pandas based on user instructions.
        3. BigQuery Agent : Can perform operations on BigQuery based on user instructions.

        You will be called when either of these agents encounter conflicts in their operations. 

        You are given the main task user requested  : {state['user_request']},
        Sub_tasks completed successfully : {json.dumps(state['tasks_done'], indent=2)},
        The Sub_task where the conflict arose : {state['next_task']},
        Conflict details : {state['error_details']},
        and relevant previous conversations from these agents : {json.dumps(relevant_tasks, indent=2)}.

        Based on this information, rewrite only the conflictd Sub_task with the necessary corrections that would resolve the conflict. Then call the appropriate agent to handle this corrected task.

        Your response needs to be a dictionary only in the following format:

                {{
            "agent" : "agent_name",
            "action" : "task description",
            "parameters" : {{"param1" : "value1", "param2" : "value2",...}}
        }}

        FORMATTING RULES:
        - Use Python boolean syntax: True/False
        - Use double quotes for strings
        - No trailing commas
        
        IMPORTANT : In case you find the user provided info is actually incorrect or insufficient to resolve the conflict, respond with :

        {{"agent" : "END", "action" : "End the workflow as the conflict could not be resolved", "parameters" : {{}}}}

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
            
            # Parse structured format
            state['next_agent'] = response_content.get('agent', 'END')
            state['next_task'] = response_content.get('action', '') + " with parameters " + str(response_content.get('parameters', {}))
            print(f"Conflict Resolver recommends {state['next_agent']} - Action: {response_content.get('action')}")
            state['has_error'] = False
            state['error_details'] = ""
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
        state['next_task'] = 'Resolve the error : ' + state.get('error_details', '')
        state['has_error'] = False  # Reset flag
        return state
    # Taking only top 5 requests and responses to avoid overload

    prompt = f"""
        In a Data Engineering team, you are the Delegator. There are different agents working in your team to help with data engineering tasks. Your main task is to understand users request & call the right agent/agents & assign the appropriate tasks to them.

        CRITICAL: Break down the user request into SINGLE, ATOMIC tasks. Each task should handle ONE file operation at a time.

        These are the tools provided to you:

        1. call_connector_agent Tool : This tool can connect to GCS source & perform data operations which include:

        - Downloading ONE file at a time from GCS into local storage
        - Uploading ONE file at a time to GCS from local storage
        - Deleting files from GCS
        - Creating new GCS buckets
        - Listing files in GCS buckets
        
        IMPORTANT: The connector_agent handles ONE file per call. If multiple files need processing, create separate tasks for each file.

        2. call_smart_transformer_agent Tool :  Can perform data transformations using pandas based on user instructions. Capabilities are :
        - Previewing data files (CSV, Excel, JSON, Parquet)
        - Understanding user instructions for data transformations
        - Generating pandas code to perform transformations
        - Saving the transformed data to specified file formats (CSV, Excel, JSON, Parquet)
        - Executing data transformations (e.g., aggregating, joining)

        3. call_bigquery_agent Tool : Can perform operations on BigQuery based on user instructions. Capabilities include:
        - Creating datasets & tables
        - Inserting rows into tables
        - Querying data
        - Loading data from GCS into BigQuery tables

        3. conflict_resolver Tool : Helps resolve any conflicts that arise during the execution of tasks by other agents. Call this tool when you identify conflicting instructions or errors in task execution. 


        This is the user's request: {user_request}

        For this request, the progress so far has been:
        Tasks Completed: {json.dumps(tasks_done[-5:], indent=2)}

        Recent Agent Responses (showing what actually happened):
        {json.dumps(state.get('model_responses', [])[-2:], indent=2)}

        Based on the user request, decide which agent to call next.
        
        REMEMBER: 
        - Process files ONE AT A TIME - if the user asks for multiple files, handle the first one now
        - Each connector_agent call should specify only ONE file using "filename" parameter (not "source_file_paths")

        Your response should only be a dictionary with agent name and task to be assigned to that agent, params required for the task in the format:
        {{
            "agent" : "agent_name",
            "action" : "task description",
            "parameters" : {{"param1" : "value1", "param2" : "value2"}}
        }}

        Example : 

        {{
            "agent" : "connector_agent",
             
              "action" : "Extract the file submissions.csv from the GCP bucket data_storage_1146 in the project data-engineering-476308 and save it locally in data folder.",
              "parameters" : {{"project_id" : "data-engineering-476308", "bucket_name" : "data_storage_1146", "filename" : "submissions.csv"}}

        }}

        Note: Use "filename" parameter (singular) for connector_agent, not "source_file_paths".

        If the user's request has been completed, respond with:

        {{
            "agent" : "END",
            "action" : "The task has been completed.",
            "parameters" : {{}}
        }}

        Do not provide any explanations or additional text.
    """

    response = model.invoke(prompt)
    print("Delegator Response : ", response.content)
    response_content = response.content

    if isinstance(response_content, str):
        response_content = ast.literal_eval(response_content)
    next_agent = response_content['agent']
    next_task = response_content['action']
    params = response_content.get('parameters', {})
    state['next_task'] = next_task + " with parameters " + str(params)
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
        state["error_details"] = response['messages'][-1].content
    else:
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
    response_text = response['messages'][-1].content
    print("Connector Agent Response : ", response_text)
    if "ERROR" in response_text.upper():
        state['has_error'] = True
        state['error_details'] = response_text  # Use error_details consistently
        # Don't update tasks_done or model_responses
    else:
        state['error_details'] = ""  # Clear error
        state["tasks_done"].append({"call_connector_agent": state['next_task']})
        state['model_responses'].append("connector_agent:" + response['messages'][-1].content)
    return state

def call_bigquery_agent(state : State) -> State:
    """Calls the BigQuery agent to handle BigQuery tasks."""
    task = state['next_task']
    print("Calling BigQuery Agent to handle the task : ", task)
    response = bigquery_agent.invoke(
        {"messages": [{"role": "user", "content": task}]}
    )
    response_text = response['messages'][-1].content
    print("BigQuery Agent Response : ", response_text)
    if "ERROR" in response_text.upper():
        state['has_error'] = True
        state['error_details'] = response_text  # Use error_details consistently
        # Don't update tasks_done or model_responses
    else:
        state['error_details'] = ""  # Clear error
        state["tasks_done"].append({"bigquery_agent": state['next_task']})
        state['model_responses'].append("bigquery_agent:" + response['messages'][-1].content)
    return state

def execute_workflow():
    workflow = StateGraph(State)

    workflow.add_node("delegator_logic", delegator_logic)
    workflow.add_node("call_connector_agent", call_connector_agent)
    workflow.add_node("call_smart_transformer_agent", call_smart_transformer_agent)
    workflow.add_node("conflict_resolver", conflict_resolver) 
    workflow.add_node("call_bigquery_agent", call_bigquery_agent)   
    workflow.add_edge(START, "delegator_logic")

    workflow.add_conditional_edges("delegator_logic", lambda state: state['next_agent'], {
        "connector_agent": "call_connector_agent",
        "smart_transformer_agent": "call_smart_transformer_agent",
        "bigquery_agent": "call_bigquery_agent",
        "conflict_resolver": "conflict_resolver",
        "END": END
    })
    workflow.add_conditional_edges("conflict_resolver", resolution_flow, {
        "delegator_logic": "delegator_logic",
        "connector_agent": "call_connector_agent",
        "bigquery_agent": "call_bigquery_agent",
        "smart_transformer_agent": "call_smart_transformer_agent",
        "END": END
    })
    workflow.add_edge("call_connector_agent", "delegator_logic")
    workflow.add_edge("call_smart_transformer_agent", "delegator_logic")
    workflow.add_edge("call_bigquery_agent", "delegator_logic")
    workflow.add_edge("delegator_logic", END)

    chain = workflow.compile()

    state = chain.invoke({"user_request" : "Read the files wb1.csv & wb2.csv from the bucket data_storage_1146 in project data-engineering-476308, merge them on the common column. Then save the result in a new file. Upload this new file to a new bucket merged_data_storage_1146 with the same filename. Once done, create a big query dataset 'emp_data' in project data-engineering-476308, next create a table 'employee' in this dataset and finally load the fiile in gcs bucket into this table.", "tasks_done": [{}], "next_agent": "", "next_task": "", "model_responses": [], "has_error": False})

    print(state)

if __name__ == "__main__":
    execute_workflow()



