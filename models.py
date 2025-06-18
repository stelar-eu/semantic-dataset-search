# This file contains the models for the semantic table search API

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


# Structure for dataset description
class DatasetDescription(BaseModel):
    general_description: str = Field(description="A general description of the dataset")
    purpose: str = Field(description="The purpose of the dataset")
    domain: str = Field(description="The domain of the dataset")