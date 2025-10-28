import json
import os
from contextlib import asynccontextmanager

import chromadb
import uvicorn
from chromadb.utils.embedding_functions import (
    OllamaEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

from .models import (
    AddDatasetRequest,
    DatasetDescription,
    DeleteDatasetRequest,
    DatasetReranking,
    SearchDatasetsRequest,
    UpdateDatasetMetadataRequest,
    UpdateDatasetRequest,
)
from .prompts import (
    CANDIDATE_DATASET_DESCRIPTION_INFERENCE_PROMPT_TEMPLATE,
    DATASET_DESCRIPTION_PROMPT_TEMPLATE,
    DATASET_RERANKING_PROMPT_TEMPLATE,
)
from .utils import flatten_auth_scope
from .processing import compose_dataset_description_for_reranking

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set logger to handle info, warnings, and errors
logger.setLevel(logging.INFO)

# Only load environment variables from .env file outside of production mode
# Otherwise the environment variables should be set during the deployment
if os.getenv("ENVIRONMENT_MODE", "dev") != "production":
    load_dotenv()
else:
    print("[INFO] Running in production mode, not loading .env file")
    # Validate that all required environment variables are set
    # else exit gracefully.
    print("[INFO] Validating environment variables")
    required_env_vars = [
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "GROQ_URL",
        "LLM_OPTION",
        "CHROMA_DIR",
    ]
    for var in required_env_vars:
        if not os.getenv(var):
            raise EnvironmentError(
                f"[ERROR] Required environment variable '{var}' is not set."
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the ChromaDB client, embedding function, collections and llm chains.
    """
    # Initialize ChromaDB
    app.state.client = chromadb.PersistentClient(path=os.getenv("CHROMA_DIR"))

    if os.getenv("EMBEDDING_OPTION") == "ollama":
        print("Using Ollama embedding function")
        app.state.ef = OllamaEmbeddingFunction(
            model_name=os.getenv("OLLAMA_EMBEDDING_MODEL"), url=os.getenv("OLLAMA_URL")
        )
    else:
        app.state.ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    app.state.description_collection = app.state.client.get_or_create_collection(
        name="dataset_descriptions", embedding_function=app.state.ef
    )
    app.state.use_case_collection = app.state.client.get_or_create_collection(
        name="dataset_use_cases", embedding_function=app.state.ef
    )
    app.state.domain_collection = app.state.client.get_or_create_collection(
        name="dataset_domains", embedding_function=app.state.ef
    )
    # Initialize LLM 
    app.state.llm = init_chat_model(
        model=os.getenv("LLM_MODEL"),
        model_provider=os.getenv("LLM_PROVIDER"),
        temperature=0,
        max_tokens=None
    )
    
    dataset_description_prompt = ChatPromptTemplate.from_template(
        DATASET_DESCRIPTION_PROMPT_TEMPLATE
    )
    app.state.dataset_description_chain = (
        dataset_description_prompt
        | app.state.llm.with_structured_output(DatasetDescription)
    )

    candidate_dataset_description_inference_prompt = ChatPromptTemplate.from_template(
        CANDIDATE_DATASET_DESCRIPTION_INFERENCE_PROMPT_TEMPLATE
    )
    app.state.candidate_dataset_description_chain = (
        candidate_dataset_description_inference_prompt
        | app.state.llm.with_structured_output(DatasetDescription)
    )

    dataset_reranking_prompt = ChatPromptTemplate.from_template(
        DATASET_RERANKING_PROMPT_TEMPLATE
    )
    app.state.dataset_reranking_chain = (
        dataset_reranking_prompt
        | app.state.llm.with_structured_output(DatasetReranking)
    )

    try:
        # Yield control to the application
        yield
    finally:
        pass


app = FastAPI(title="Semantic Dataset Search API", lifespan=lifespan)

@app.get("/")
def get_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"status": "running", "message": "Semantic Dataset Search API is running"}


def _ingest_dataset(request: AddDatasetRequest) -> None:
    dataset_id = request.dataset_id
    try:
        if request.dataset_official_description is None:
            official_description = ""
        else:
            official_description = str(request.dataset_official_description)

        if request.dataset_profile_description is None:
            profile_description = ""
        else:
            profile_description = str(request.dataset_profile_description)

        total_description = (
            "Official Description: "
            + official_description
            + "\n"
            + "Profile Description extracted from the tool: "
            + profile_description
        )

        dataset_description = app.state.dataset_description_chain.invoke(
            {"column_descriptions": total_description}
        )
        general_description = dataset_description.general_description
        purpose = dataset_description.purpose
        domain = dataset_description.domain

        metadata = {
            "dataset_id": dataset_id,
            "official_description": official_description,
            "profile_description": profile_description,
        }
        if request.dataset_metadata:
            processed_metadata = {}
            for key, value in request.dataset_metadata.items():
                if key == "auth_scope" and isinstance(value, list):
                    for scope in value:
                        processed_metadata[f"auth_scope_{scope}"] = True
                elif isinstance(value, list):
                    processed_metadata[key] = json.dumps(value)
                else:
                    processed_metadata[key] = value
            metadata.update(processed_metadata)

        app.state.description_collection.add(
            documents=[general_description],
            metadatas=[metadata],
            ids=[dataset_id],
        )
        app.state.use_case_collection.add(
            documents=[purpose],
            metadatas=[metadata],
            ids=[dataset_id],
        )
        app.state.domain_collection.add(
            documents=[domain],
            metadatas=[metadata],
            ids=[dataset_id],
        )
        logger.info(f"Dataset {dataset_id} ingested successfully")

    except Exception as exc:
        logger.exception("Dataset %s ingestion failed: %s", dataset_id, exc)

@app.post("/add_dataset", status_code=status.HTTP_202_ACCEPTED)
async def add_dataset(
    request: AddDatasetRequest,
    background_tasks: BackgroundTasks,
):
    # enqueue the ingestion to run *after* the response is sent
    background_tasks.add_task(_ingest_dataset, request)

    # 202 Accepted means “request received, processing later”
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "accepted",
            "message": "Dataset ingestion scheduled; check logs for completion.",
        },
    )

@app.delete("/delete_dataset")
def delete_dataset(request: DeleteDatasetRequest):
    """
    Delete a dataset from the ChromaDB database collections.
    """
    dataset_id = request.dataset_id
    try:
        app.state.description_collection.delete(ids=[dataset_id])
        app.state.use_case_collection.delete(ids=[dataset_id])
        app.state.domain_collection.delete(ids=[dataset_id])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success", "message": "Dataset deleted successfully"}

@app.put("/update_dataset")
def update_dataset(request: UpdateDatasetRequest):
    """
    Update a dataset in the ChromaDB database collections.
    """
    dataset_id = request.dataset_id

    try:
        if request.dataset_official_description is None:
            official_description = ""
        else:
            official_description = str(request.dataset_official_description)

        if request.dataset_profile_description is None:
            profile_description = ""
        else:
            profile_description = str(request.dataset_profile_description)

        total_description = str(
            "Official Description: "
            + str(official_description)
            + "\n"
            + "Profile Description extracted from the tool: "
            + str(profile_description)
        )

        # Extract full descriptions
        dataset_description = app.state.dataset_description_chain.invoke(
            {"column_descriptions": total_description}
        )

        general_description = dataset_description.general_description
        purpose = dataset_description.purpose
        domain = dataset_description.domain

        # Ingest descriptions
        if request.dataset_metadata is not None:
            # Convert any list values to JSON strings for ChromaDB compatibility
            processed_metadata = {}
            for key, value in request.dataset_metadata.items():
                if isinstance(value, list):
                    processed_metadata[key] = json.dumps(value)
                else:
                    processed_metadata[key] = value

            app.state.description_collection.update(
                documents=[general_description],
                metadatas=[processed_metadata],
                ids=[dataset_id],
            )
            app.state.use_case_collection.update(
                documents=[purpose], metadatas=[processed_metadata], ids=[dataset_id]
            )
            app.state.domain_collection.update(
                documents=[domain], metadatas=[processed_metadata], ids=[dataset_id]
            )
        else:
            app.state.description_collection.update(
                documents=[general_description], ids=[dataset_id]
            )
            app.state.use_case_collection.update(documents=[purpose], ids=[dataset_id])
            app.state.domain_collection.update(documents=[domain], ids=[dataset_id])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "success", "message": "Dataset updated successfully"}


@app.put("/update_dataset_metadata")
def update_dataset_metadata(request: UpdateDatasetMetadataRequest):
    """
    Update the metadata of a dataset in the ChromaDB database collections.
    """
    try:
        dataset_id = request.dataset_id
        existing_metadata = app.state.description_collection.get(ids=[dataset_id])[
            "metadatas"
        ][0]
        new_metadata = {**existing_metadata, **request.dataset_metadata}
        app.state.description_collection.update(
            ids=[dataset_id], metadatas=[new_metadata]
        )
        app.state.use_case_collection.update(ids=[dataset_id], metadatas=[new_metadata])
        app.state.domain_collection.update(ids=[dataset_id], metadatas=[new_metadata])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success", "message": "Dataset metadata updated successfully"}


@app.post("/search_datasets")
def search_datasets(request: SearchDatasetsRequest):
    """
    Search for datasets in the ChromaDB database.

    Args:
        query: The query to search for.
        n_results: The number of results to return.

    Returns:
        A list of tuples, each containing a dataset ID and a distance.
    """

    try:
        candidate_dataset_description = (
            app.state.candidate_dataset_description_chain.invoke(
                {"query": request.query}
            )
        )
        logger.info(f"Candidate dataset description: {candidate_dataset_description}")
        general_description = candidate_dataset_description.general_description
        purpose = candidate_dataset_description.purpose
        domain = candidate_dataset_description.domain

        general_description_results = None
        purpose_results = None
        domain_results = None
        general_description_results_dict = {}
        purpose_results_dict = {}
        domain_results_dict = {}
        max_general_description_distance = 0
        max_purpose_distance = 0
        max_domain_distance = 0

        where_scope = flatten_auth_scope(request.auth_scope)

        if general_description:
            general_description_results = app.state.description_collection.query(
                query_texts=[general_description],
                n_results=2 * request.n_results,
                where=where_scope,
            )
            if general_description_results["ids"][0]:  # Check if results are not empty
                general_description_results_dict = {
                    k: v
                    for k, v in zip(
                        general_description_results["ids"][0],
                        general_description_results["distances"][0],
                    )
                }
                max_general_description_distance = max(
                    general_description_results["distances"][0]
                )
        # Search for the purpose
        if purpose:
            purpose_results = app.state.use_case_collection.query(
                query_texts=[purpose],
                n_results=2 * request.n_results,
                where=where_scope,
            )
            if purpose_results["ids"][0]:  # Check if results are not empty
                purpose_results_dict = {
                    k: v
                    for k, v in zip(
                        purpose_results["ids"][0], purpose_results["distances"][0]
                    )
                }
                max_purpose_distance = max(purpose_results["distances"][0])

        # Search for the domain
        if domain:
            domain_results = app.state.domain_collection.query(
                query_texts=[domain], n_results=2 * request.n_results, where=where_scope
            )
            if domain_results["ids"][0]:  # Check if results are not empty
                domain_results_dict = {
                    k: v
                    for k, v in zip(
                        domain_results["ids"][0], domain_results["distances"][0]
                    )
                }
                max_domain_distance = max(domain_results["distances"][0])

        # Collect all candidate datasets
        candidate_datasets = []
        if general_description_results and general_description_results["ids"][0]:
            candidate_datasets.extend(general_description_results["ids"][0])
        if purpose_results and purpose_results["ids"][0]:
            candidate_datasets.extend(purpose_results["ids"][0])
        if domain_results and domain_results["ids"][0]:
            candidate_datasets.extend(domain_results["ids"][0])

        if not candidate_datasets:
            return []

        candidate_datasets_distances = {k: 0 for k in candidate_datasets}
        for dataset in candidate_datasets:
            if general_description:
                if dataset in general_description_results_dict:
                    candidate_datasets_distances[
                        dataset
                    ] += general_description_results_dict[dataset]
                else:
                    candidate_datasets_distances[
                        dataset
                    ] += max_general_description_distance
            if purpose:
                if dataset in purpose_results_dict:
                    candidate_datasets_distances[dataset] += purpose_results_dict[
                        dataset
                    ]
                else:
                    candidate_datasets_distances[dataset] += max_purpose_distance
            if domain:
                if dataset in domain_results_dict:
                    candidate_datasets_distances[dataset] += domain_results_dict[
                        dataset
                    ]
                else:
                    candidate_datasets_distances[dataset] += max_domain_distance

        # sort the candidate datasets by the distances
        sorted_candidate_datasets_distances = sorted(
            candidate_datasets_distances.items(), key=lambda x: x[1]
        )

        # return the top candidate datasets
        return sorted_candidate_datasets_distances[: request.n_results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_datasets_streaming")
async def search_datasets_streaming(request: SearchDatasetsRequest):
    """
    Search for datasets in the ChromaDB database with streaming response.

    Args:
        query: The query to search for.
        n_results: The number of results to return.

    Returns:
        A streaming response of dataset search results.
    """

    async def generate_results():
        try:

            where_scope = flatten_auth_scope(request.auth_scope)

            # Stream the initial processing message
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Analyzing query...'})}\n\n"

            candidate_dataset_description = (
                app.state.candidate_dataset_description_chain.invoke(
                    {"query": request.query}
                )
            )
            general_description = candidate_dataset_description.general_description
            purpose = candidate_dataset_description.purpose
            domain = candidate_dataset_description.domain

            yield f"data: {json.dumps({'status': 'analysis_complete', 'message': 'Query analysis complete', 'components': {'general_description': general_description, 'purpose': purpose, 'domain': domain}})}\n\n"

            general_description_results = None
            purpose_results = None
            domain_results = None
            general_description_results_dict = {}
            purpose_results_dict = {}
            domain_results_dict = {}
            max_general_description_distance = 0
            max_purpose_distance = 0
            max_domain_distance = 0

            yield f"data: {json.dumps({'status': 'processing', 'message': 'Searching for candidates...'})}\n\n"
            # Search for general description
            if general_description:
                general_description_results = app.state.description_collection.query(
                    query_texts=[general_description],
                    n_results=2 * request.n_results,
                    where=where_scope,
                )
                if general_description_results["ids"][0]:
                    general_description_results_dict = {
                        k: v
                        for k, v in zip(
                            general_description_results["ids"][0],
                            general_description_results["distances"][0],
                        )
                    }
                    max_general_description_distance = max(
                        general_description_results["distances"][0]
                    )

            # Search for the purpose
            if purpose:
                purpose_results = app.state.use_case_collection.query(
                    query_texts=[purpose],
                    n_results=2 * request.n_results,
                    where=where_scope,
                )
                if purpose_results["ids"][0]:
                    purpose_results_dict = {
                        k: v
                        for k, v in zip(
                            purpose_results["ids"][0], purpose_results["distances"][0]
                        )
                    }
                    max_purpose_distance = max(purpose_results["distances"][0])

            # Search for the domain
            if domain:
                domain_results = app.state.domain_collection.query(
                    query_texts=[domain],
                    n_results=2 * request.n_results,
                    where=where_scope,
                )
                if domain_results["ids"][0]:
                    domain_results_dict = {
                        k: v
                        for k, v in zip(
                            domain_results["ids"][0], domain_results["distances"][0]
                        )
                    }
                    max_domain_distance = max(domain_results["distances"][0])

            # Collect all candidate datasets
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Combining search results...'})}\n\n"
            candidate_datasets = []
            if general_description_results and general_description_results["ids"][0]:
                candidate_datasets.extend(general_description_results["ids"][0])
            if purpose_results and purpose_results["ids"][0]:
                candidate_datasets.extend(purpose_results["ids"][0])
            if domain_results and domain_results["ids"][0]:
                candidate_datasets.extend(domain_results["ids"][0])

            if not candidate_datasets:
                yield f"data: {json.dumps({'status': 'complete', 'results': []})}\n\n"
                return

            # Calculate distances
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Calculating relevance scores...'})}\n\n"
            candidate_datasets_distances = {k: 0 for k in candidate_datasets}
            for dataset in candidate_datasets:
                if general_description:
                    if dataset in general_description_results_dict:
                        candidate_datasets_distances[
                            dataset
                        ] += general_description_results_dict[dataset]
                    else:
                        candidate_datasets_distances[
                            dataset
                        ] += max_general_description_distance
                if purpose:
                    if dataset in purpose_results_dict:
                        candidate_datasets_distances[dataset] += purpose_results_dict[
                            dataset
                        ]
                    else:
                        candidate_datasets_distances[dataset] += max_purpose_distance
                if domain:
                    if dataset in domain_results_dict:
                        candidate_datasets_distances[dataset] += domain_results_dict[
                            dataset
                        ]
                    else:
                        candidate_datasets_distances[dataset] += max_domain_distance

            # Sort and return results
            sorted_candidate_datasets_distances = sorted(
                candidate_datasets_distances.items(), key=lambda x: x[1]
            )
            final_results = sorted_candidate_datasets_distances[: request.n_results]

            yield f"data: {json.dumps({'status': 'complete', 'results': final_results})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_results(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/search_datasets_expanded")
def search_datasets_expanded(request: SearchDatasetsRequest):
    """
    Search for datasets in the ChromaDB database and return the complete dataset entries.

    Args:
        query: The query to search for.
        n_results: The number of results to return.

    Returns:
        A list of datasets, each containing the dataset ID, the distance, the official description, the profile description and the domain.
    """

    try:
        candidate_dataset_description = (
            app.state.candidate_dataset_description_chain.invoke(
                {"query": request.query}
            )
        )
        general_description = candidate_dataset_description.general_description
        purpose = candidate_dataset_description.purpose
        domain = candidate_dataset_description.domain

        general_description_results = None
        purpose_results = None
        domain_results = None
        general_description_results_dict = {}
        purpose_results_dict = {}
        domain_results_dict = {}
        max_general_description_distance = 0
        max_purpose_distance = 0
        max_domain_distance = 0

        where_scope = flatten_auth_scope(request.auth_scope)

        if general_description:
            general_description_results = app.state.description_collection.query(
                query_texts=[general_description],
                n_results=2 * request.n_results,
                where=where_scope,
            )
            if general_description_results["ids"][0]:  # Check if results are not empty
                general_description_results_dict = {
                    k: v
                    for k, v in zip(
                        general_description_results["ids"][0],
                        general_description_results["distances"][0],
                    )
                }
                max_general_description_distance = max(
                    general_description_results["distances"][0]
                )

        # Search for the purpose
        if purpose:
            purpose_results = app.state.use_case_collection.query(
                query_texts=[purpose],
                n_results=2 * request.n_results,
                where=where_scope,
            )
            if purpose_results["ids"][0]:  # Check if results are not empty
                purpose_results_dict = {
                    k: v
                    for k, v in zip(
                        purpose_results["ids"][0], purpose_results["distances"][0]
                    )
                }
                max_purpose_distance = max(purpose_results["distances"][0])

        # Search for the domain
        if domain:
            domain_results = app.state.domain_collection.query(
                query_texts=[domain], n_results=2 * request.n_results, where=where_scope
            )
            if domain_results["ids"][0]:  # Check if results are not empty
                domain_results_dict = {
                    k: v
                    for k, v in zip(
                        domain_results["ids"][0], domain_results["distances"][0]
                    )
                }
                max_domain_distance = max(domain_results["distances"][0])

        candidate_datasets = (
            general_description_results["ids"][0]
            + purpose_results["ids"][0]
            + domain_results["ids"][0]
        )
        candidate_datasets_distances = {k: 0 for k in candidate_datasets}
        for dataset in candidate_datasets:
            if general_description:
                if dataset in general_description_results_dict:
                    candidate_datasets_distances[
                        dataset
                    ] += general_description_results_dict[dataset]
                else:
                    candidate_datasets_distances[
                        dataset
                    ] += max_general_description_distance
            if purpose:
                if dataset in purpose_results_dict:
                    candidate_datasets_distances[dataset] += purpose_results_dict[
                        dataset
                    ]
                else:
                    candidate_datasets_distances[dataset] += max_purpose_distance
            if domain:
                if dataset in domain_results_dict:
                    candidate_datasets_distances[dataset] += domain_results_dict[
                        dataset
                    ]
                else:
                    candidate_datasets_distances[dataset] += max_domain_distance

        # sort the candidate datasets by the distances
        sorted_candidate_datasets_distances = sorted(
            candidate_datasets_distances.items(), key=lambda x: x[1]
        )

        results = []

        for dataset_id, _ in sorted_candidate_datasets_distances:
            dataset_info = {
                "dataset_description": app.state.description_collection.get(
                    ids=[dataset_id]
                )["documents"][0],
                "use_case": app.state.use_case_collection.get(ids=[dataset_id])[
                    "documents"
                ][0],
                "domain": app.state.domain_collection.get(ids=[dataset_id])[
                    "documents"
                ][0],
            }
            results.append((dataset_id, dataset_info))

        return results[: request.n_results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_datasets_expanded_streaming")
async def search_datasets_expanded_streaming(request: SearchDatasetsRequest):
    """
    Search for datasets in the ChromaDB database and return the complete dataset entries.

    Args:
        query: The query to search for.
        n_results: The number of results to return.

    Returns:
        A stream of datasets, each containing the dataset ID, the distance, the official description, the profile description and the domain.
    """

    async def generate_results():
        try:

            where_scope = flatten_auth_scope(request.auth_scope)

            yield f"data: {json.dumps({'status': 'processing', 'message': 'Analyzing query...'})}\n\n"
            candidate_dataset_description = (
                app.state.candidate_dataset_description_chain.invoke(
                    {"query": request.query}
                )
            )
            general_description = candidate_dataset_description.general_description
            purpose = candidate_dataset_description.purpose
            domain = candidate_dataset_description.domain

            yield f"data: {json.dumps({'status': 'analysis_complete', 'message': 'Query analysis complete', 'components': {'general_description': general_description, 'purpose': purpose, 'domain': domain}})}\n\n"

            general_description_results = None
            purpose_results = None
            domain_results = None
            general_description_results_dict = {}
            purpose_results_dict = {}
            domain_results_dict = {}
            max_general_description_distance = 0
            max_purpose_distance = 0
            max_domain_distance = 0

            yield f"data: {json.dumps({'status': 'processing', 'message': 'Searching for candidates...'})}\n\n"
            if general_description:
                general_description_results = app.state.description_collection.query(
                    query_texts=[general_description],
                    n_results=2 * request.n_results,
                    where=where_scope,
                )
                if general_description_results["ids"][
                    0
                ]:  # Check if results are not empty
                    general_description_results_dict = {
                        k: v
                        for k, v in zip(
                            general_description_results["ids"][0],
                            general_description_results["distances"][0],
                        )
                    }
                    max_general_description_distance = max(
                        general_description_results["distances"][0]
                    )

            # Search for the purpose
            if purpose:
                purpose_results = app.state.use_case_collection.query(
                    query_texts=[purpose],
                    n_results=2 * request.n_results,
                    where=where_scope,
                )
                if purpose_results["ids"][0]:  # Check if results are not empty
                    purpose_results_dict = {
                        k: v
                        for k, v in zip(
                            purpose_results["ids"][0], purpose_results["distances"][0]
                        )
                    }
                    max_purpose_distance = max(purpose_results["distances"][0])

            # Search for the domain
            if domain:
                domain_results = app.state.domain_collection.query(
                    query_texts=[domain],
                    n_results=2 * request.n_results,
                    where=where_scope,
                )
                if domain_results["ids"][0]:  # Check if results are not empty
                    domain_results_dict = {
                        k: v
                        for k, v in zip(
                            domain_results["ids"][0], domain_results["distances"][0]
                        )
                    }
                    max_domain_distance = max(domain_results["distances"][0])

            candidate_datasets = (
                general_description_results["ids"][0]
                + purpose_results["ids"][0]
                + domain_results["ids"][0]
            )
            candidate_datasets_distances = {k: 0 for k in candidate_datasets}
            for dataset in candidate_datasets:
                if general_description:
                    if dataset in general_description_results_dict:
                        candidate_datasets_distances[
                            dataset
                        ] += general_description_results_dict[dataset]
                    else:
                        candidate_datasets_distances[
                            dataset
                        ] += max_general_description_distance
                if purpose:
                    if dataset in purpose_results_dict:
                        candidate_datasets_distances[dataset] += purpose_results_dict[
                            dataset
                        ]
                    else:
                        candidate_datasets_distances[dataset] += max_purpose_distance
                if domain:
                    if dataset in domain_results_dict:
                        candidate_datasets_distances[dataset] += domain_results_dict[
                            dataset
                        ]
                    else:
                        candidate_datasets_distances[dataset] += max_domain_distance

            yield f"data: {json.dumps({'status': 'processing', 'message': 'Combining search results...'})}\n\n"

            # sort the candidate datasets by the distances
            sorted_candidate_datasets_distances = sorted(
                candidate_datasets_distances.items(), key=lambda x: x[1]
            )

            results = []
            results_count = 0

            for dataset_id, _ in sorted_candidate_datasets_distances:
                if results_count >= request.n_results:
                    break

                dataset_info = {
                    "dataset_description": app.state.description_collection.get(
                        ids=[dataset_id]
                    )["documents"][0],
                    "use_case": app.state.use_case_collection.get(ids=[dataset_id])[
                        "documents"
                    ][0],
                    "domain": app.state.domain_collection.get(ids=[dataset_id])[
                        "documents"
                    ][0],
                }

                results.append({"dataset_id": dataset_id, "dataset_info": dataset_info})

                results_count += 1

            yield f"data: {json.dumps({'status': 'complete', 'message': 'Search complete', 'results': results})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_results(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/search_datasets_explainable")
def search_datasets_explainable(request: SearchDatasetsRequest):
    """
    Search for datasets in the ChromaDB database and return the complete dataset entries with explainability.

    Args:
        query: The query to search for.
        n_results: The number of results to return.    

    Returns:
        A list of datasets, each containing the dataset ID, the distance, the official description, the profile description and the domain with explainability.
    """

    try:
        candidate_dataset_description = (
            app.state.candidate_dataset_description_chain.invoke(
                {"query": request.query}
            )
        )
        general_description = candidate_dataset_description.general_description
        purpose = candidate_dataset_description.purpose
        domain = candidate_dataset_description.domain

        general_description_results = None
        purpose_results = None
        domain_results = None
        general_description_results_dict = {}
        purpose_results_dict = {}
        domain_results_dict = {}
        max_general_description_distance = 0
        max_purpose_distance = 0
        max_domain_distance = 0


        if general_description:
            general_description_results = app.state.description_collection.query(
                query_texts=[general_description],
                n_results=2 * request.n_results,
            )
            if general_description_results["ids"][0]:  # Check if results are not empty
                general_description_results_dict = {
                    k: v
                    for k, v in zip(
                        general_description_results["ids"][0],
                        general_description_results["distances"][0],
                    )
                }
                max_general_description_distance = max(
                    general_description_results["distances"][0]
                )

        # Search for the purpose
        if purpose:
            purpose_results = app.state.use_case_collection.query(
                query_texts=[purpose],
                n_results=2 * request.n_results,
            )
            if purpose_results["ids"][0]:  # Check if results are not empty
                purpose_results_dict = {
                    k: v
                    for k, v in zip(
                        purpose_results["ids"][0], purpose_results["distances"][0]
                    )
                }
                max_purpose_distance = max(purpose_results["distances"][0])

        # Search for the domain
        if domain:
            domain_results = app.state.domain_collection.query(
                query_texts=[domain], 
                n_results=2 * request.n_results
            )
            if domain_results["ids"][0]:  # Check if results are not empty
                domain_results_dict = {
                    k: v
                    for k, v in zip(
                        domain_results["ids"][0], domain_results["distances"][0]
                    )
                }
                max_domain_distance = max(domain_results["distances"][0])

        candidate_datasets = (
            general_description_results["ids"][0]
            + purpose_results["ids"][0]
            + domain_results["ids"][0]
        )
        candidate_datasets_distances = {k: 0 for k in candidate_datasets}
        for dataset in candidate_datasets:
            if general_description:
                if dataset in general_description_results_dict:
                    candidate_datasets_distances[
                        dataset
                    ] += general_description_results_dict[dataset]
                else:
                    candidate_datasets_distances[
                        dataset
                    ] += max_general_description_distance
            if purpose:
                if dataset in purpose_results_dict:
                    candidate_datasets_distances[dataset] += purpose_results_dict[
                        dataset
                    ]
                else:
                    candidate_datasets_distances[dataset] += max_purpose_distance
            if domain:
                if dataset in domain_results_dict:
                    candidate_datasets_distances[dataset] += domain_results_dict[
                        dataset
                    ]
                else:
                    candidate_datasets_distances[dataset] += max_domain_distance

        # sort the candidate datasets by the distances
        sorted_candidate_datasets_distances = sorted(
            candidate_datasets_distances.items(), key=lambda x: x[1]
        )

        results = []

        for dataset_id, _ in sorted_candidate_datasets_distances:
            dataset_info = {
                "dataset_description": app.state.description_collection.get(
                    ids=[dataset_id]
                )["documents"][0],
                "use_case": app.state.use_case_collection.get(ids=[dataset_id])[
                    "documents"
                ][0],
                "domain": app.state.domain_collection.get(ids=[dataset_id])[
                    "documents"
                ][0],
                "dataset_official_description": app.state.description_collection.get(ids=[dataset_id])["metadatas"][0]["official_description"],
                "dataset_title": app.state.description_collection.get(ids=[dataset_id])["metadatas"][0].get("title", ""),
            }
            results.append((dataset_id, dataset_info))
            logger.info(f"Results before reranking: {results}")
        try:
            combined_dataset_descriptions = [f"{i+1}. {compose_dataset_description_for_reranking(dataset_info)}" for i, (dataset_id, dataset_info) in enumerate(results)]
            reranked_indexes = app.state.dataset_reranking_chain.invoke(
                {"query": request.query, "dataset_results": combined_dataset_descriptions}
            )
            logger.info(f"Reranked indexes: {reranked_indexes.reranked_indexes}")
            reranked_results = [results[i-1] for i in reranked_indexes.reranked_indexes]
        except Exception as e:
            logger.exception("Error reranking datasets: %s", e)
            reranked_results = results

        return {
            "results": reranked_results[: request.n_results],
            "query_analysis": {
                "general_description": general_description,
                "purpose": purpose,
                "domain": domain,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_all_datasets")
def get_all_datasets():
    """
    Get all datasets from the ChromaDB database collections without any scope checks.
    """
    try:
        # Get all datasets from the description collection
        all_datasets = app.state.description_collection.get()
        
        results = []
        
        for i, dataset_id in enumerate(all_datasets["ids"]):
            dataset_info = {
                "dataset_id": dataset_id,
                "dataset_description": all_datasets["documents"][i],
                "use_case": app.state.use_case_collection.get(ids=[dataset_id])["documents"][0],
                "domain": app.state.domain_collection.get(ids=[dataset_id])["documents"][0],
                "metadata": all_datasets["metadatas"][i]
            }
            results.append(dataset_info)
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
