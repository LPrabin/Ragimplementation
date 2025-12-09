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
    k : int 

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

# class QueryEvalRequest(BaseModel):
#     notebook_name: str
#     query: List[str]
#     k : int 
#     chunk_size : List[int]

# class Revelanceoutput(BaseModel):
#     relevant: List[int] = Field(description="A list of relevance scores (e.g., 0 or 1) for each document chunk.")
#     explanation: str = Field(description="A brief explanation of how the scores were derived.")

# class QueryEvalResponse(BaseModel):
#     value: List[int] = Field(description="A list of relevance scores (e.g., 0 or 1) for each document chunk.")
#     explanation: str = Field(description="A brief explanation of how the scores were derived.")

class QueryExperimentRequest(BaseModel):
    notebook_name: str
    k : int 
    chunk_size : List[int]
    chunk_overlap : List[int]
    dataset_name : str