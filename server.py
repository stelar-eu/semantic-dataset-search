from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction, SentenceTransformerEmbeddingFunction

from prompts import DATASET_DESCRIPTION_PROMPT_TEMPLATE, CANDIDATE_DATASET_DESCRIPTION_INFERENCE_PROMPT_TEMPLATE
from models import DatasetDescription

import pandas as pd
import json
import uvicorn
from dotenv import load_dotenv
import os
load_dotenv()

app = FastAPI(title="Semantic Table Search API")

@app.on_event("startup")
def startup_event():
    """
    Initialize the ChromaDB client, embedding function, collections and llm chains.
    """
    # Initialize ChromaDB
    app.state.client = chromadb.PersistentClient(path=os.getenv("CHROMA_DIR"))

    if os.getenv("EMBEDDING_OPTION") == "ollama":
        print("Using Ollama embedding function")
        app.state.ef = OllamaEmbeddingFunction(
            model_name=os.getenv("OLLAMA_EMBEDDING_MODEL"),
            url=os.getenv("OLLAMA_URL")
        )
    else: 
        app.state.ef = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
        
    app.state.description_collection = app.state.client.get_or_create_collection(name="dataset_descriptions", embedding_function=app.state.ef)
    app.state.use_case_collection = app.state.client.get_or_create_collection(name="dataset_use_cases", embedding_function=app.state.ef)
    app.state.domain_collection = app.state.client.get_or_create_collection(name="dataset_domains", embedding_function=app.state.ef)
    
    # Initialize LLM and chains
    if os.getenv("LLM_OPTION") == "ollama":
        app.state.llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0, max_tokens=None, base_url=os.getenv("OLLAMA_URL"))
    elif os.getenv("LLM_OPTION") == "groq":
        app.state.llm = ChatGroq(model=os.getenv("GROQ_MODEL"), temperature=0, max_tokens=None, base_url=os.getenv("GROQ_URL"))
    dataset_description_prompt = ChatPromptTemplate.from_template(DATASET_DESCRIPTION_PROMPT_TEMPLATE)
    app.state.dataset_description_chain = dataset_description_prompt | app.state.llm.with_structured_output(DatasetDescription)
    
    candidate_dataset_description_inference_prompt = ChatPromptTemplate.from_template(CANDIDATE_DATASET_DESCRIPTION_INFERENCE_PROMPT_TEMPLATE)
    app.state.candidate_dataset_description_chain = candidate_dataset_description_inference_prompt | app.state.llm.with_structured_output(DatasetDescription)
    

class AddDatasetRequest(BaseModel):
    dataset_id: str
    dataset_official_description: str
    dataset_profile_description: str
    dataset_metadata: Dict[str, Any]

@app.post("/add_dataset")
def add_dataset(request: AddDatasetRequest):
    """
    Add a dataset to the ChromaDB database collections.
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

        total_description = str("Official Description: " + str(official_description) + "\n" + "Profile Description extracted from the tool: " + str(profile_description))
        
        # Extract full descriptions
        dataset_description = app.state.dataset_description_chain.invoke({"column_descriptions": total_description})

        general_description = dataset_description.general_description
        purpose = dataset_description.purpose
        domain = dataset_description.domain

        # Ingest descriptions
        metadata = {"dataset_id": dataset_id}
        if request.dataset_metadata:
            # Convert any list values to JSON strings for ChromaDB compatibility
            processed_metadata = {}
            for key, value in request.dataset_metadata.items():
                if isinstance(value, list):
                    processed_metadata[key] = json.dumps(value)
                else:
                    processed_metadata[key] = value
            metadata.update(processed_metadata)
            
        app.state.description_collection.add(
            documents=[general_description],
            metadatas=[metadata],
            ids=[dataset_id]
        )
        app.state.use_case_collection.add(
            documents=[purpose],
            metadatas=[metadata],
            ids=[dataset_id]
        )
        app.state.domain_collection.add(
            documents=[domain],
            metadatas=[metadata],
            ids=[dataset_id]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success", "message": "Dataset ingested successfully"}


class DeleteDatasetRequest(BaseModel):
    dataset_id: str

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

class UpdateDatasetRequest(BaseModel):
    dataset_id: str
    dataset_official_description: str
    dataset_profile_description: str
    dataset_domain: str
    dataset_metadata: Dict[str, Any]
    
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

        total_description = str("Official Description: " + str(official_description) + "\n" + "Profile Description extracted from the tool: " + str(profile_description))
        
        # Extract full descriptions
        dataset_description = app.state.dataset_description_chain.invoke({"column_descriptions": total_description})

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
                ids=[dataset_id]
            )
            app.state.use_case_collection.update(
                documents=[purpose],
                metadatas=[processed_metadata],
                ids=[dataset_id]
            )
            app.state.domain_collection.update(
                documents=[domain],
                metadatas=[processed_metadata],
                ids=[dataset_id]
            )
        else:
            app.state.description_collection.update(
                documents=[general_description],
                ids=[dataset_id]
            )
            app.state.use_case_collection.update(
                documents=[purpose],
                ids=[dataset_id]
            )
            app.state.domain_collection.update(
                documents=[domain],
                ids=[dataset_id]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "success", "message": "Dataset updated successfully"}

class UpdateDatasetMetadataRequest(BaseModel):
    dataset_id: str
    dataset_metadata: Dict[str, Any]

@app.put("/update_dataset_metadata")
def update_dataset_metadata(request: UpdateDatasetMetadataRequest):
    """
    Update the metadata of a dataset in the ChromaDB database collections.
    """
    try:    
        dataset_id = request.dataset_id
        existing_metadata = app.state.description_collection.get(ids=[dataset_id])['metadatas'][0]
        new_metadata = {**existing_metadata, **request.dataset_metadata}
        app.state.description_collection.update(ids=[dataset_id], metadatas=[new_metadata])
        app.state.use_case_collection.update(ids=[dataset_id], metadatas=[new_metadata])
        app.state.domain_collection.update(ids=[dataset_id], metadatas=[new_metadata])    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success", "message": "Dataset metadata updated successfully"}

class SearchDatasetsRequest(BaseModel):
    query: str
    n_results: int = 5
    auth_scope: List[str] = []

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
        candidate_dataset_description = app.state.candidate_dataset_description_chain.invoke({"query": request.query})
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

        print(request.auth_scope)

        if general_description:
            general_description_results = app.state.description_collection.query(
                query_texts=[general_description],
                n_results=2*request.n_results,
                where={"auth_scope": {"$in": [json.dumps(request.auth_scope)]}}
            )
            if general_description_results['ids'][0]:  # Check if results are not empty
                general_description_results_dict = { k: v for k, v in zip(general_description_results['ids'][0], general_description_results['distances'][0]) }
                max_general_description_distance = max(general_description_results['distances'][0])
        
        # Search for the purpose
        if purpose:
            purpose_results = app.state.use_case_collection.query(
                query_texts=[purpose],
                n_results=2*request.n_results, 
                where={"auth_scope": {"$in": [json.dumps(request.auth_scope)]}}
            )
            if purpose_results['ids'][0]:  # Check if results are not empty
                purpose_results_dict = { k: v for k, v in zip(purpose_results['ids'][0], purpose_results['distances'][0]) }
                max_purpose_distance = max(purpose_results['distances'][0])
        
        # Search for the domain
        if domain:
            domain_results = app.state.domain_collection.query(
                query_texts=[domain],
                n_results=2*request.n_results,
                where={"auth_scope": {"$in": [json.dumps(request.auth_scope)]}}
            )
            if domain_results['ids'][0]:  # Check if results are not empty
                domain_results_dict = { k: v for k, v in zip(domain_results['ids'][0], domain_results['distances'][0]) }
                max_domain_distance = max(domain_results['distances'][0])
        
        # Collect all candidate datasets
        candidate_datasets = []
        if general_description_results and general_description_results['ids'][0]:
            candidate_datasets.extend(general_description_results['ids'][0])
        if purpose_results and purpose_results['ids'][0]:
            candidate_datasets.extend(purpose_results['ids'][0])
        if domain_results and domain_results['ids'][0]:
            candidate_datasets.extend(domain_results['ids'][0])
        
        if not candidate_datasets:
            return []
        
        candidate_datasets_distances = { k: 0 for k in candidate_datasets }
        for dataset in candidate_datasets:
            if general_description: 
                if dataset in general_description_results_dict:
                    candidate_datasets_distances[dataset] += general_description_results_dict[dataset]
                else: 
                    candidate_datasets_distances[dataset] += max_general_description_distance
            if purpose: 
                if dataset in purpose_results_dict:
                    candidate_datasets_distances[dataset] += purpose_results_dict[dataset]
                else: 
                    candidate_datasets_distances[dataset] += max_purpose_distance
            if domain: 
                if dataset in domain_results_dict:
                    candidate_datasets_distances[dataset] += domain_results_dict[dataset]
                else: 
                    candidate_datasets_distances[dataset] += max_domain_distance
        
        # sort the candidate datasets by the distances
        sorted_candidate_datasets_distances = sorted(candidate_datasets_distances.items(), key=lambda x: x[1])

        # return the top candidate datasets
        return sorted_candidate_datasets_distances[:request.n_results]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

from fastapi.responses import StreamingResponse
import json

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
            # Stream the initial processing message
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Analyzing query...'})}\n\n"
            
            candidate_dataset_description = app.state.candidate_dataset_description_chain.invoke({"query": request.query})
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
                    n_results=2*request.n_results,
                    where={"auth_scope": {"$in": [json.dumps(request.auth_scope)]}}
                )
                if general_description_results['ids'][0]:
                    general_description_results_dict = { k: v for k, v in zip(general_description_results['ids'][0], general_description_results['distances'][0]) }
                    max_general_description_distance = max(general_description_results['distances'][0])
            
            # Search for the purpose
            if purpose:
                purpose_results = app.state.use_case_collection.query(
                    query_texts=[purpose],
                    n_results=2*request.n_results, 
                    where={"auth_scope": {"$in": [json.dumps(request.auth_scope)]}}
                )
                if purpose_results['ids'][0]:
                    purpose_results_dict = { k: v for k, v in zip(purpose_results['ids'][0], purpose_results['distances'][0]) }
                    max_purpose_distance = max(purpose_results['distances'][0])
            
            # Search for the domain
            if domain:
                domain_results = app.state.domain_collection.query(
                    query_texts=[domain],
                    n_results=2*request.n_results,
                    where={"auth_scope": {"$in": [json.dumps(request.auth_scope)]}}
                )
                if domain_results['ids'][0]:
                    domain_results_dict = { k: v for k, v in zip(domain_results['ids'][0], domain_results['distances'][0]) }
                    max_domain_distance = max(domain_results['distances'][0])
            
            # Collect all candidate datasets
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Combining search results...'})}\n\n"
            candidate_datasets = []
            if general_description_results and general_description_results['ids'][0]:
                candidate_datasets.extend(general_description_results['ids'][0])
            if purpose_results and purpose_results['ids'][0]:
                candidate_datasets.extend(purpose_results['ids'][0])
            if domain_results and domain_results['ids'][0]:
                candidate_datasets.extend(domain_results['ids'][0])
            
            if not candidate_datasets:
                yield f"data: {json.dumps({'status': 'complete', 'results': []})}\n\n"
                return
            
            # Calculate distances
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Calculating relevance scores...'})}\n\n"
            candidate_datasets_distances = { k: 0 for k in candidate_datasets }
            for dataset in candidate_datasets:
                if general_description: 
                    if dataset in general_description_results_dict:
                        candidate_datasets_distances[dataset] += general_description_results_dict[dataset]
                    else: 
                        candidate_datasets_distances[dataset] += max_general_description_distance
                if purpose: 
                    if dataset in purpose_results_dict:
                        candidate_datasets_distances[dataset] += purpose_results_dict[dataset]
                    else: 
                        candidate_datasets_distances[dataset] += max_purpose_distance
                if domain: 
                    if dataset in domain_results_dict:
                        candidate_datasets_distances[dataset] += domain_results_dict[dataset]
                    else: 
                        candidate_datasets_distances[dataset] += max_domain_distance
            
            # Sort and return results
            sorted_candidate_datasets_distances = sorted(candidate_datasets_distances.items(), key=lambda x: x[1])
            final_results = sorted_candidate_datasets_distances[:request.n_results]
            
            yield f"data: {json.dumps({'status': 'complete', 'results': final_results})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_results(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
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
        candidate_dataset_description = app.state.candidate_dataset_description_chain.invoke({"query": request.query})
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
                n_results=2*request.n_results,
                where={"auth_scope": {"$in": [json.dumps(request.auth_scope)]}}
            )
            if general_description_results['ids'][0]:  # Check if results are not empty
                general_description_results_dict = { k: v for k, v in zip(general_description_results['ids'][0], general_description_results['distances'][0]) }
                max_general_description_distance = max(general_description_results['distances'][0])
        
        # Search for the purpose
        if purpose:
            purpose_results = app.state.use_case_collection.query(
                query_texts=[purpose],
                n_results=2*request.n_results,
                where={"auth_scope": {"$in": [json.dumps(request.auth_scope)]}}
            )
            if purpose_results['ids'][0]:  # Check if results are not empty
                purpose_results_dict = { k: v for k, v in zip(purpose_results['ids'][0], purpose_results['distances'][0]) }
                max_purpose_distance = max(purpose_results['distances'][0])
        
        # Search for the domain
        if domain:
            domain_results = app.state.domain_collection.query(
                query_texts=[domain],
                n_results=2*request.n_results,
                where={"auth_scope": {"$in": [json.dumps(request.auth_scope)]}}
            )
            if domain_results['ids'][0]:  # Check if results are not empty
                domain_results_dict = { k: v for k, v in zip(domain_results['ids'][0], domain_results['distances'][0]) }
                max_domain_distance = max(domain_results['distances'][0])
        
        candidate_datasets = general_description_results['ids'][0] + purpose_results['ids'][0] + domain_results['ids'][0]
        candidate_datasets_distances = { k: 0 for k in candidate_datasets }
        for dataset in candidate_datasets:
            if general_description: 
                if dataset in general_description_results_dict:
                    candidate_datasets_distances[dataset] += general_description_results_dict[dataset]
                else: 
                    candidate_datasets_distances[dataset] += max_general_description_distance
            if purpose: 
                if dataset in purpose_results_dict:
                    candidate_datasets_distances[dataset] += purpose_results_dict[dataset]
                else: 
                    candidate_datasets_distances[dataset] += max_purpose_distance
            if domain: 
                if dataset in domain_results_dict:
                    candidate_datasets_distances[dataset] += domain_results_dict[dataset]
                else: 
                    candidate_datasets_distances[dataset] += max_domain_distance
        
        # sort the candidate datasets by the distances
        sorted_candidate_datasets_distances = sorted(candidate_datasets_distances.items(), key=lambda x: x[1])

        results = []
        
        for dataset_id, _ in sorted_candidate_datasets_distances:
            dataset_info = {
                'dataset_description': app.state.description_collection.get(ids=[dataset_id])['documents'][0],
                'use_case': app.state.use_case_collection.get(ids=[dataset_id])['documents'][0],
                'domain': app.state.domain_collection.get(ids=[dataset_id])['documents'][0]
            }
            results.append((dataset_id, dataset_info))
            
        return results[:request.n_results]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/search_datasets_expanded")
async def search_datasets_expanded(request: SearchDatasetsRequest):
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
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Analyzing query...'})}\n\n"
            candidate_dataset_description = app.state.candidate_dataset_description_chain.invoke({"query": request.query})
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
                    n_results=2*request.n_results,
                    where={"auth_scope": {"$in": [json.dumps(request.auth_scope)]}}
                )
                if general_description_results['ids'][0]:  # Check if results are not empty
                    general_description_results_dict = { k: v for k, v in zip(general_description_results['ids'][0], general_description_results['distances'][0]) }
                    max_general_description_distance = max(general_description_results['distances'][0])
            
            # Search for the purpose
            if purpose:
                purpose_results = app.state.use_case_collection.query(
                    query_texts=[purpose],
                    n_results=2*request.n_results,
                    where={"auth_scope": {"$in": [json.dumps(request.auth_scope)]}}
                )
                if purpose_results['ids'][0]:  # Check if results are not empty
                    purpose_results_dict = { k: v for k, v in zip(purpose_results['ids'][0], purpose_results['distances'][0]) }
                    max_purpose_distance = max(purpose_results['distances'][0])
            
            # Search for the domain
            if domain:
                domain_results = app.state.domain_collection.query(
                    query_texts=[domain],
                    n_results=2*request.n_results,
                    where={"auth_scope": {"$in": [json.dumps(request.auth_scope)]}}
                )
                if domain_results['ids'][0]:  # Check if results are not empty
                    domain_results_dict = { k: v for k, v in zip(domain_results['ids'][0], domain_results['distances'][0]) }
                    max_domain_distance = max(domain_results['distances'][0])
            
            candidate_datasets = general_description_results['ids'][0] + purpose_results['ids'][0] + domain_results['ids'][0]
            candidate_datasets_distances = { k: 0 for k in candidate_datasets }
            for dataset in candidate_datasets:
                if general_description: 
                    if dataset in general_description_results_dict:
                        candidate_datasets_distances[dataset] += general_description_results_dict[dataset]
                    else: 
                        candidate_datasets_distances[dataset] += max_general_description_distance
                if purpose: 
                    if dataset in purpose_results_dict:
                        candidate_datasets_distances[dataset] += purpose_results_dict[dataset]
                    else: 
                        candidate_datasets_distances[dataset] += max_purpose_distance
                if domain: 
                    if dataset in domain_results_dict:
                        candidate_datasets_distances[dataset] += domain_results_dict[dataset]
                    else: 
                        candidate_datasets_distances[dataset] += max_domain_distance
            
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Combining search results...'})}\n\n"
            
            # sort the candidate datasets by the distances
            sorted_candidate_datasets_distances = sorted(candidate_datasets_distances.items(), key=lambda x: x[1])

            results = []
            results_count = 0
            
            for dataset_id, _ in sorted_candidate_datasets_distances:
                if results_count >= request.n_results:
                    break
                    
                dataset_info = {
                    'dataset_description': app.state.description_collection.get(ids=[dataset_id])['documents'][0],
                    'use_case': app.state.use_case_collection.get(ids=[dataset_id])['documents'][0],
                    'domain': app.state.domain_collection.get(ids=[dataset_id])['documents'][0]
                }
                
                results.append({
                    'dataset_id': dataset_id,
                    'dataset_info': dataset_info
                })
                
                results_count += 1
            
            yield f"data: {json.dumps({'status': 'complete', 'message': 'Search complete', 'results': results})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_results(), 
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
