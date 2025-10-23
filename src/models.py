from pydantic import BaseModel, Field
from typing import Dict, Any, List

# Structure for dataset description
class DatasetDescription(BaseModel):
    general_description: str = Field(description="A general description of the dataset")
    purpose: str = Field(description="The purpose of the dataset")
    domain: str = Field(description="The domain of the dataset")

class AddDatasetRequest(BaseModel):
    dataset_id: str
    dataset_official_description: str
    dataset_profile_description: str
    dataset_metadata: Dict[str, Any]

class DeleteDatasetRequest(BaseModel):
    dataset_id: str

class UpdateDatasetRequest(BaseModel):
    dataset_id: str
    dataset_official_description: str
    dataset_profile_description: str
    dataset_domain: str
    dataset_metadata: Dict[str, Any]

class UpdateDatasetMetadataRequest(BaseModel):
    dataset_id: str
    dataset_metadata: Dict[str, Any]

class SearchDatasetsRequest(BaseModel):
    query: str
    n_results: int = 5
    auth_scope: List[str] = []

