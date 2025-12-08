from pydantic import BaseModel
from typing import List, Optional

class NotebookRequest(BaseModel):
    name: str

class NotebookResponse(BaseModel):
    name: str
    resource_count: int

class QueryRequest(BaseModel):
    notebook_name: str
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    top3docs: List[str]

class ResourceResponse(BaseModel):
    id: str
    name: str
    type: str

class ResourceDeleteRequest(BaseModel):
    notebook_name: str
    resource_name: str

