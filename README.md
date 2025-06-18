# Semantic Table Search

A FastAPI-based service that enables semantic search capabilities for tabular datasets. The service uses Ollama for embeddings and LLM operations, and ChromaDB for vector storage.

## Features

- Add datasets with descriptions and metadata
- Semantic search across datasets using natural language queries
- Vector-based similarity search using embeddings
- Support for both basic and expanded search results

## Prerequisites

- Python 3.12.8 or higher
- Ollama installed and running (for embeddings and LLM operations)
- ChromaDB (installed automatically via pip)

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_MODEL=llama2
CHROMA_DIR=./chroma_storage
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
pip install fastapi uvicorn langchain-ollama chromadb python-dotenv pydantic
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

### Search Datasets
```bash
POST /search_datasets
{
    "query": "Your search query",
    "n_results": 5
}
```

### Search Datasets (Expanded)
```bash
POST /search_datasets_expanded
{
    "query": "Your search query",
    "n_results": 5
}
```

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
└── notebooks/          # Test and development notebooks
```

## Notes

- The service requires Ollama to be running and accessible
- ChromaDB data is persisted in the directory specified by `CHROMA_DIR`
- When running with Docker, make sure Ollama is accessible from the container
- The default port is 8000, but can be changed by modifying the uvicorn command 