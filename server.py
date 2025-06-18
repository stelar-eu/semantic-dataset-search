from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

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
    # Initialize ChromaDB
    app.state.client = chromadb.PersistentClient(path=os.getenv("CHROMA_DIR"))

    app.state.ef = OllamaEmbeddingFunction(
        model_name=os.getenv("OLLAMA_EMBEDDING_MODEL"),
        url=os.getenv("OLLAMA_URL")
    )
        
    app.state.description_collection = app.state.client.get_or_create_collection(name="dataset_descriptions", embedding_function=app.state.ef)
    app.state.use_case_collection = app.state.client.get_or_create_collection(name="dataset_use_cases", embedding_function=app.state.ef)
    app.state.domain_collection = app.state.client.get_or_create_collection(name="dataset_domains", embedding_function=app.state.ef)
    
    # Initialize LLM and chains
    app.state.llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0, max_tokens=None, base_url=os.getenv("OLLAMA_URL"))
    dataset_description_prompt = ChatPromptTemplate.from_template(DATASET_DESCRIPTION_PROMPT_TEMPLATE)
    app.state.dataset_description_chain = dataset_description_prompt | app.state.llm.with_structured_output(DatasetDescription)
    
    candidate_dataset_description_inference_prompt = ChatPromptTemplate.from_template(CANDIDATE_DATASET_DESCRIPTION_INFERENCE_PROMPT_TEMPLATE)
    app.state.candidate_dataset_description_chain = candidate_dataset_description_inference_prompt | app.state.llm.with_structured_output(DatasetDescription)
    

# class SetupRequest(BaseModel):
#     chroma_dir: str = Field(default="./chroma_storage", description="The directory to store the ChromaDB database.")
#     collection_names: List[str] = Field(default=['dataset_descriptions', 'dataset_use_cases', 'dataset_domains'], description="The names of the collections to create. Should be a list of three strings.")
#     ollama_url: str = Field(default="http://localhost:11434", description="The URL of the Ollama server that will be used to embed the data can be accessed.")
#     ollama_embedding_model: str = Field(default="nomic-embed-text", description="The model to use for embedding. Should be a string.")

# @app.post("/setup_sds")
# def setup_sds(request: SetupRequest):
#     try:
#         setup_chroma_db_infrastructure(request.chroma_dir, request.collection_names, request.ollama_url, request.ollama_embedding_model)
#         return {"status": "success", "message": "Semantic table search setup completed"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

class AddDatasetRequest(BaseModel):
    dataset_id: str
    dataset_official_description: str
    dataset_profile_description: str
    dataset_metadata: Dict[str, Any]

@app.post("/add_dataset")
def add_dataset(request: AddDatasetRequest):
    dataset_id = request.dataset_id

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
    app.state.description_collection.add(
        documents=[general_description],
        metadatas=[{"dataset_id": dataset_id}],
        ids=[dataset_id]
    )
    app.state.use_case_collection.add(
        documents=[purpose],
        metadatas=[{"dataset_id": dataset_id}],
        ids=[dataset_id]
    )
    app.state.domain_collection.add(
        documents=[domain],
        metadatas=[{"dataset_id": dataset_id}],
        ids=[dataset_id]
    )

    return {"status": "success", "message": "Dataset ingested successfully"}

class SearchDatasetsRequest(BaseModel):
    query: str
    n_results: int = 5

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

        if general_description:
            general_description_results = app.state.description_collection.query(
                query_texts=[general_description],
                n_results=2*request.n_results
            )
            general_description_results_dict = { k: v for k, v in zip(general_description_results['ids'][0], general_description_results['distances'][0]) }
            max_general_description_distance = max(general_description_results['distances'][0])
        
        # Search for the purpose
        if purpose:
            purpose_results = app.state.use_case_collection.query(
                query_texts=[purpose],
                n_results=2*request.n_results
            )
            purpose_results_dict = { k: v for k, v in zip(purpose_results['ids'][0], purpose_results['distances'][0]) }
            max_purpose_distance = max(purpose_results['distances'][0])
        
        # Search for the domain
        if domain:
            domain_results = app.state.domain_collection.query(
                query_texts=[domain],
                n_results=2*request.n_results
            )
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

        # return the top 3 candidate datasets
        return sorted_candidate_datasets_distances[:request.n_results]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

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

        if general_description:
            general_description_results = app.state.description_collection.query(
                query_texts=[general_description],
                n_results=2*request.n_results
            )
            general_description_results_dict = { k: v for k, v in zip(general_description_results['ids'][0], general_description_results['distances'][0]) }
            max_general_description_distance = max(general_description_results['distances'][0])
        
        # Search for the purpose
        if purpose:
            purpose_results = app.state.use_case_collection.query(
                query_texts=[purpose],
                n_results=2*request.n_results
            )
            purpose_results_dict = { k: v for k, v in zip(purpose_results['ids'][0], purpose_results['distances'][0]) }
            max_purpose_distance = max(purpose_results['distances'][0])
        
        # Search for the domain
        if domain:
            domain_results = app.state.domain_collection.query(
                query_texts=[domain],
                n_results=2*request.n_results
            )
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
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
