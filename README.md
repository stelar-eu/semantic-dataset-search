# Semantic Table Search

A FastAPI-based service that enables semantic search capabilities for tabular datasets. The service supports multiple LLM and embedding providers, including Ollama, Groq, and SentenceTransformers, with ChromaDB for vector storage.

## Features

- Add, update, and delete datasets with descriptions and metadata
- Semantic search across datasets using natural language queries
- Vector-based similarity search using embeddings
- Support for both basic and expanded search results
- Real-time streaming search responses for better user experience
- Multi-collection storage (descriptions, use cases, domains)
- LLM-powered dataset description extraction and classification
- Authorization scope filtering for dataset access control
- Flexible LLM and embedding provider configuration

## Prerequisites

- Python 3.12.8 or higher
- At least one of the following:
  - Ollama installed and running (for local LLM and embeddings)
  - Groq API access (for cloud-based LLM)
  - SentenceTransformers (automatic fallback for embeddings)
- ChromaDB (installed automatically via pip)

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# ChromaDB Configuration
CHROMA_DIR=./chroma_storage

# LLM Provider Configuration
LLM_OPTION=ollama  # Options: "ollama", "groq"

# Ollama Configuration (if using LLM_OPTION=ollama)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Groq Configuration (if using LLM_OPTION=groq)
GROQ_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama3-8b-8192

# Embedding Provider Configuration
EMBEDDING_OPTION=ollama  # Options: "ollama", "sentencetransformers"

# Ollama Embedding Configuration (if using EMBEDDING_OPTION=ollama)
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Note: If EMBEDDING_OPTION is not "ollama", the service will automatically use SentenceTransformers with "all-MiniLM-L6-v2"
```

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd semantic-table-search
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Running Locally

1. Ensure Ollama is running:
```bash
ollama serve
```

2. Start the FastAPI server:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Docker Installation

1. Build the Docker image:
```bash
docker build -t semantic-table-search .
```

2. Run the container:
```bash
docker run -p 8000:8000 \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  -e OLLAMA_EMBEDDING_MODEL=nomic-embed-text \
  -e OLLAMA_MODEL=llama2 \
  -e CHROMA_DIR=/app/chroma_storage \
  semantic-table-search
```

## API Endpoints

### Add Dataset
```bash
POST /add_dataset
{
    "dataset_id": "unique_id",
    "dataset_official_description": "Official description of the dataset",
    "dataset_profile_description": "Profile description from the tool",
    "dataset_metadata": {
        "key": "value"
    }
}
```

### Update Dataset
```bash
PUT /update_dataset
{
    "dataset_id": "unique_id",
    "dataset_official_description": "Updated official description",
    "dataset_profile_description": "Updated profile description",
    "dataset_domain": "domain",
    "dataset_metadata": {
        "key": "value"
    }
}
```

### Update Dataset Metadata
```bash
PUT /update_dataset_metadata
{
    "dataset_id": "unique_id",
    "dataset_metadata": {
        "key": "value"
    }
}
```

### Delete Dataset
```bash
DELETE /delete_dataset
{
    "dataset_id": "unique_id"
}
```

### Search Datasets
```bash
POST /search_datasets
{
    "query": "Your search query",
    "n_results": 5,
    "auth_scope": ["scope1", "scope2"]  # Optional: filter by authorization scopes
}
```

### Search Datasets (Streaming)
```bash
POST /search_datasets_streaming
{
    "query": "Your search query",
    "n_results": 5,
    "auth_scope": ["scope1", "scope2"]  # Optional: filter by authorization scopes
}
```

Returns a streaming response with real-time progress updates:
- `text/event-stream` format
- Progress messages during query analysis and search
- Final results with dataset IDs and relevance scores

### Search Datasets (Expanded)
```bash
POST /search_datasets_expanded
{
    "query": "Your search query",
    "n_results": 5,
    "auth_scope": ["scope1", "scope2"]  # Optional: filter by authorization scopes
}
```

Returns complete dataset information including descriptions, use cases, and domains. Also available as a streaming endpoint with real-time progress updates.

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Development

The project structure:
```
├── server.py           # Main FastAPI application
├── models.py           # Pydantic models
├── prompts.py          # LLM prompt templates
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── .env              # Environment variables (create this)
├── .gitignore        # Git ignore rules
├── LICENSE           # Project license
└── notebooks/        # Test and development notebooks
```

## Architecture

The service uses three ChromaDB collections for different aspects of dataset information:
- **dataset_descriptions**: General descriptions of datasets
- **dataset_use_cases**: Purpose and use case information
- **dataset_domains**: Domain classification

Each dataset is processed through LLM chains to extract structured information from raw descriptions.

### Streaming Responses

The service provides streaming endpoints for real-time search feedback:
- **Progress Updates**: Real-time status messages during processing
- **Component Analysis**: Shows how the query is broken down into description, purpose, and domain
- **Search Progress**: Updates on candidate retrieval and scoring
- **Final Results**: Complete search results with relevance scores

## Notes

- The service supports multiple LLM and embedding providers for flexibility
- When using Ollama, ensure it's running and accessible
- Groq provides faster cloud-based LLM processing
- SentenceTransformers provides reliable local embedding generation
- ChromaDB data is persisted in the directory specified by `CHROMA_DIR`
- When running with Docker, ensure external services (Ollama/Groq) are accessible
- The default port is 8000, but can be changed by modifying the uvicorn command
- The service automatically creates ChromaDB collections on startup
- Streaming endpoints provide better user experience for long-running searches
- Authorization scopes enable fine-grained access control over datasets

## Configuration Options

### LLM Providers

The service supports multiple LLM providers:

1. **Ollama** (local): Set `LLM_OPTION=ollama`
   - Requires Ollama server running locally
   - Configure `OLLAMA_URL` and `OLLAMA_MODEL`

2. **Groq** (cloud): Set `LLM_OPTION=groq`
   - Requires Groq API access
   - Configure `GROQ_API_KEY` and `GROQ_MODEL`

### Embedding Providers

The service supports multiple embedding providers:

1. **Ollama** (local): Set `EMBEDDING_OPTION=ollama`
   - Requires Ollama server with embedding model
   - Configure `OLLAMA_EMBEDDING_MODEL`

2. **SentenceTransformers** (default): Any other value or unset
   - Uses local SentenceTransformers model
   - Automatically downloads "all-MiniLM-L6-v2" model

### Authorization Scopes

The service supports filtering datasets by authorization scopes:
- Add `auth_scope` metadata when creating/updating datasets
- Use `auth_scope` parameter in search requests to filter results
- Only datasets matching the provided scopes will be returned 