from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
import pandas as pd
from datetime import datetime
import os
import shutil
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1000)

@tool
def generate_pandas_logic(instructions: str, input_filename: str, output_filename: str) -> str:
    """
    Generates and executes Pandas transformation code.
    
    Args:
        instructions: What transformations to apply (e.g., "Add column 'Date' with today's date")
        input_filename: Path to the input file to read
        output_filename: Path where transformed data should be saved
    """
    
    try:
        if not os.path.exists(input_filename):
            return f"ERROR : File {input_filename} does not exist."
        
        print(f"Input: {input_filename}")
        print(f"Output: {output_filename}")
        print(f"Instructions: {instructions}")
        
        file_ext = os.path.splitext(input_filename)[1].lower()
        output_ext = os.path.splitext(output_filename)[1].lower()
    
        prompt = f""" 
You are a pandas expert. Generate Python code to:

1. Read the input file: {input_filename} (format: {file_ext})
2. Apply transformations: {instructions}
3. Save result to: {output_filename} (format: {output_ext})

Rules:
1. Only code, no explanations
2. Don't import libraries (already imported)
3. Only pandas operations, no other I/O
4. Don't include print statements
5. Do not execute any code  

Example:
df = pd.read_csv("{input_filename}")
df['column'] = 'value'
df.to_csv("{output_filename}", index=False)

Generate the code:
"""

        response = model.invoke(prompt)
        generated_code = response.content.strip()

        # Clean markdown
        if generated_code.startswith("```python"):
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif generated_code.startswith("```"):
            generated_code = generated_code.split("```")[1].split("```")[0].strip()

        print("*" * 50)
        print("Generated Code:")
        print(generated_code)
        print("*" * 50)

        # Security check
        forbidden_keywords = ['import', 'exec', 'eval', 'open', '__', 'os.', 'sys.', 'subprocess', 'compile']
        code_lower = generated_code.lower()
        for keyword in forbidden_keywords:
            if keyword in code_lower:
                return f"Error: Forbidden operation: '{keyword}'"
        
        # Execute
        namespace = {'df': None, 'pd': pd, 'datetime': datetime}
        try:
            exec(generated_code, namespace)
        except Exception as e:
            return f"Error executing code: {str(e)}\n\nCode:\n{generated_code}"

        # Backup original
        backup_dir = "./backups"
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"{timestamp}_{os.path.basename(input_filename)}")
        shutil.copy2(input_filename, backup_path)

        return f"Success! Transformed data saved to: {output_filename}\nOriginal backed up to: {backup_path}"
    
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def preview_data(filename: str, num_rows: int = 5) -> str:
    """Preview the first few rows of a data file."""
    try:
        if not os.path.exists(filename):
            return f"ERROR : File {filename} does not exist."
        
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext == ".csv":  
            df = pd.read_csv(filename)
        elif file_ext in [".xls", ".xlsx"]:
            df = pd.read_excel(filename)
        elif file_ext == ".json":
            df = pd.read_json(filename)
        else:
            return f"ERROR : Unsupported file format: {file_ext}"

        preview = df.head(num_rows).to_string()

        return f"""
File: {filename}
Rows: {len(df)}
Columns: {list(df.columns)}

Preview (first {num_rows} rows):
{preview}
"""
    except Exception as e:
        return f"ERROR: {str(e)}"
smart_transformer_agent = create_agent(
        model=model,
        system_prompt="""You are a smart data transformer agent.

Steps to follow:

1. Preview the input file to understand its structure

2. Identify from user's request:
   - Input filename
   - Transformation operations
   - Output filename (if not specified, generate one with '_transformed' suffix)

3. Call generate_pandas_logic with THREE parameters:
   - instructions: what transformations to apply
   - input_filename: the source file
   - output_filename: where to save results

   In case you encounter any errors during code execution, inform the same to the user with relevant error message. DO NOT rerun the code in case of errors.
IMPORTANT: Always extract the output filename from user's request. If not mentioned, create one.

NOTE : In case you don't find the file, check in Data folder.
Example:
User: "Preview data.csv, add column X, save to output.csv"
You call: generate_pandas_logic(
    instructions="Add column X with value Y",
    input_filename="./data/data.csv",
    output_filename="./data/output.csv"
)
""",
        tools=[preview_data, generate_pandas_logic]
    )
    

if __name__ == "__main__":


    result = smart_transformer_agent.invoke({
        "messages": [{
            "role": "user",
            "content": """Preview ./data/submissions.csv, 
                        Save the data in this file as a properly formatted json file"""
        }]
    })

    print("\nAI Response:", result['messages'][-1].content)