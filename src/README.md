# Semantic Dataset Search API

This directory contains the core implementation of the Semantic Dataset Search API, a FastAPI-based service that provides intelligent dataset discovery using vector embeddings and LLM-powered analysis.

## Overview

The API enables users to:
- Add datasets with descriptions and metadata
- Search for datasets using natural language queries
- Update and delete existing datasets
- Filter results based on authentication scopes

## Architecture

The system uses a multi-collection approach with ChromaDB for vector storage:
- **Description Collection**: Stores general dataset descriptions
- **Use Case Collection**: Stores dataset purposes and use cases  
- **Domain Collection**: Stores dataset domain classifications

## Files

### `server.py`
The main FastAPI application with the following endpoints:

#### Core Endpoints
- `GET /` - Health check endpoint
- `POST /add_dataset` - Add a new dataset (async processing)
- `DELETE /delete_dataset` - Remove a dataset from all collections
- `PUT /update_dataset` - Update dataset content and metadata
- `PUT /update_dataset_metadata` - Update only dataset metadata

#### Search Endpoints
- `POST /search_datasets` - Basic semantic search returning dataset IDs and distances
- `POST /search_datasets_streaming` - Streaming version of basic search
- `POST /search_datasets_expanded` - Search with full dataset information
- `POST /search_datasets_expanded_streaming` - Streaming version of expanded search

#### Key Features
- **LLM Integration**: Supports both Ollama and Groq LLM providers
- **Embedding Options**: Ollama or SentenceTransformer embeddings
- **Background Processing**: Dataset ingestion runs asynchronously
- **Streaming Responses**: Real-time search progress updates
- **Authentication Scoping**: Filter results based on user permissions

### `models.py`
Pydantic models defining the API request/response schemas:

- `DatasetDescription` - Structured dataset information (description, purpose, domain)
- `AddDatasetRequest` - Schema for adding new datasets
- `DeleteDatasetRequest` - Schema for dataset deletion
- `UpdateDatasetRequest` - Schema for updating datasets
- `UpdateDatasetMetadataRequest` - Schema for metadata-only updates
- `SearchDatasetsRequest` - Schema for search queries with optional auth scoping

### `prompts.py`
LLM prompt templates for dataset analysis:

- `DATASET_DESCRIPTION_PROMPT_TEMPLATE` - Extracts structured information from dataset descriptions
- `CANDIDATE_DATASET_DESCRIPTION_INFERENCE_PROMPT_TEMPLATE` - Analyzes user queries to infer search intent

### `utils.py`
Utility functions:

- `flatten_auth_scope()` - Converts authentication scopes into ChromaDB query filters

## Search Algorithm

The search process works in three phases:

1. **Query Analysis**: LLM analyzes the user query to extract:
   - General description requirements
   - Use case requirements  
   - Domain requirements

2. **Vector Search**: Performs semantic search across three collections:
   - Description collection for general matches
   - Use case collection for purpose matches
   - Domain collection for domain matches

3. **Result Ranking**: Combines distances from all three searches to rank results by relevance

## Configuration

The API supports multiple configuration options via environment variables:

- `LLM_OPTION`: Choose between "ollama" or "groq"
- `EMBEDDING_OPTION`: Choose between "ollama" or "sentence_transformer"
- `CHROMA_DIR`: Path to ChromaDB storage directory
- Various API keys and model configurations

## Usage

The API is designed to be deployed as a containerized service. See the main project README for deployment instructions.
