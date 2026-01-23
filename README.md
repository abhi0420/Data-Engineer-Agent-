# Data Engineer Agent

A multi-agent AI system that automates data engineering tasks using LangGraph orchestration with GCP integrations (BigQuery & Cloud Storage).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Request                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DELEGATOR LOGIC                                   │
│           • Parses user request into atomic tasks                           │
│           • Routes to appropriate agent                                     │
│           • Tracks completed tasks                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
        ┌───────────────────┐ ┌───────────────┐ ┌───────────────────┐
        │  CONNECTOR AGENT  │ │ TRANSFORMER   │ │  BIGQUERY AGENT   │
        │                   │ │    AGENT      │ │                   │
        │ • Download files  │ │ • Preview data│ │ • Create datasets │
        │ • Upload files    │ │ • Transform   │ │ • Create tables   │
        │ • Create buckets  │ │   with pandas │ │ • Run queries     │
        │                   │ │ • Save results│ │ • Load from GCS   │
        └───────────────────┘ └───────────────┘ └───────────────────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
                              (on ERROR)
                                      ▼
                    ┌─────────────────────────────────────┐
                    │        CONFLICT RESOLVER            │
                    │                                     │
                    │ • Uses TF-IDF semantic search       │
                    │ • Finds relevant past responses     │
                    │ • Rewrites failed task with fixes   │
                    │ • Routes back to appropriate agent  │
                    └─────────────────────────────────────┘
```

## Workflow State

The system maintains state throughout execution:

```python
State = {
    "user_request": str,      # Original user request
    "tasks_done": List[dict], # Completed tasks log
    "model_responses": List,  # Agent responses for context
    "next_agent": str,        # Next agent to call
    "next_task": str,         # Task description with params
    "has_error": bool,        # Error flag for routing
    "error_details": str      # Error message for resolver
}
```

## Agents & Tools

### 1. Connector Agent
Handles Google Cloud Storage operations.

|           Tool          |          Description            | Parameters |
|-------------------------|---------------------------------|------------|
| `extract_data_from_gcp` | Download file from GCS to local | `project_id`, `bucket_name`, `filename` `bucket_name`, `filename` |
| `load_data_to_gcp`      | Upload local file to GCS        | `project_id`, `bucket_name`, `source_file_path`, `dest_blob_name`, `create_new_bucket` |

### 2. Smart Transformer Agent
Performs pandas-based data transformations with AI-generated code.

|          Tool           |           Description          | Parameters |
|-------------------------|--------------------------------|------------|
| `preview_data`          | Preview first N rows of a file | `filename`, `num_rows` |
| `generate_pandas_logic` | Generate & execute pandas transformations | `instructions`, `input_filename`, `output_filename` |

**Security Features:**
- Forbidden keywords check (`import`, `exec`, `eval`, `os.`, `subprocess`, etc.)
- Automatic backup of original files before transformation
- Sandboxed execution namespace

### 3. BigQuery Agent
Manages BigQuery datasets, tables, and queries.

|             Tool          |      Description     | Parameters |
|---------------------------|----------------------|------------|
| `create_bigquery_dataset` | Create a new dataset | `project_id`, `dataset_id`, `location` |
| `create_bigquery_table`   | Create table with schema | `project_id`, `dataset_id`, `table_id`, `schema` |
| `create_partitioned_table`| Create partitioned table | `project_id`, `dataset_id`, `table_id`, `schema`, `partition_field` |
| `execute_bigquery_query` | Run SQL query | `project_id`, `query` |
| `load_table_from_gcs`    | Load data from GCS URI | `project_id`, `dataset_id`, `table_id`, `source_uri`, `schema`, `file_format` |
| `insert_rows_into_bigquery`| Insert rows into table | `project_id`, `dataset_id`, `table_id`, `rows` |
| `create_view`              | Create a view          | `project_id`, `dataset_id`, `view_id`, `query` |

### 4. Conflict Resolver
Automatically handles errors by:
1. Searching past agent responses using TF-IDF similarity
2. Analyzing error context
3. Rewriting the failed task with corrections
4. Routing back to the appropriate agent

**Key Principle:** Only uses information explicitly provided in context - never fabricates parameter values.

## Installation

### Prerequisites
- Python 3.10+
- GCP Project with BigQuery and Cloud Storage APIs enabled
- Service account JSON key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Data-Engineer-Agent.git
cd Data-Engineer-Agent
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
# Create .env file
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

## Usage

### Basic Example

```python
from langgraph_workflow import execute_workflow

request = """
Download sales.csv from bucket my-data-bucket in project my-gcp-project,
add a 'processed_date' column with today's date,
then upload to bucket processed-data-bucket
"""

result = execute_workflow(request)
```

### Complex Multi-Step Example

```python
request = """
1. Download wb1.csv and wb2.csv from bucket data_storage in project my-project
2. Merge them on the common column 'id'
3. Upload merged file to bucket merged_data_storage
4. Create BigQuery dataset 'analytics' 
5. Load the merged data into a new table
"""

result = execute_workflow(request)
```

### Running Individual Agents

```python
# Connector Agent
from agents.connector import connector_agent

result = connector_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Download file.csv from bucket my-bucket in project my-project"
    }]
})

# Transformer Agent  
from agents.data_transformer import smart_transformer_agent

result = smart_transformer_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Preview ./data/sales.csv, filter rows where amount > 100, save to filtered.csv"
    }]
})

# BigQuery Agent
from agents.bigquery_assistant import bigquery_agent

result = bigquery_agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "Create dataset 'sales_data' in project my-project"
    }]
})
```

## Project Structure

```
Data-Engineer-Agent/
├── langgraph_workflow.py    # Main orchestration workflow
├── agents/
│   ├── connector.py         # GCS operations agent
│   ├── data_transformer.py  # Pandas transformation agent
│   ├── bigquery_assistant.py# BigQuery operations agent
│   └── delegator.py         # (Legacy) delegator logic
├── gcs_source.py            # GCPSource class for GCS operations
├── bigquery_source.py       # BigQuerySource class for BQ operations
├── tests/
│   ├── test_bigquery_source.py
│   ├── test_gcs_source.py
│   └── conftest.py
├── data/                    # Local data directory
├── backups/                 # Auto-generated transformation backups
└── .env                     # Environment variables
```

## Error Handling

The system uses a structured error recovery flow:

1. **Agent encounters error** → Sets `has_error=True` and `error_details`
2. **Delegator detects error** → Routes to `conflict_resolver`
3. **Conflict Resolver** → Uses semantic search to find relevant context
4. **Resolution attempt** → Rewrites task or returns END if unresolvable

Error messages follow the pattern `ERROR: <description>` for consistent detection.

## Configuration

### Model Settings
```python
# In each agent file
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,      # Low temperature for consistency
    max_tokens=1000
)
```

### GCS Settings
- Default download location: `./data/`
- Backup location: `./backups/`

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

## Limitations

- Single-threaded execution (no parallel agent calls)
- File paths are relative to project root
- Maximum context window limitations for very long workflows

## Future Improvements

- [ ] Add retry logic with exponential backoffs
- [ ] Web UI for workflow visualization
- [ ] Support for additional data sources (S3, Azure Blob)
- [ ] Integrating an Airflow Agent into workflow to provide DAG related features

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request
