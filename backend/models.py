from pydantic import BaseModel , Field
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



class EmbeddingConfig(BaseModel):
    type: str = Field(default="google", pattern="^(google|openai)$")
    model_name: str = "text-embedding-004"

class LLMConfig(BaseModel):
    type: str = Field(default="google", pattern="^(google|openai)$")
    model_name: str = "gemini-2.5-flash-lite"
    temperature: float = 0.1

class ChunkingConfig(BaseModel):
    strategy: str = Field(default="recursive", pattern="^(recursive|semantic)$")
    chunk_size: int = 1000
    chunk_overlap: int = 200

# 2. The Full Experiment Configuration
class ExperimentConfig(BaseModel):
    """Holds all settings for ONE specific experiment run."""
    experiment_id: str
    dataset_name: str
    notebook_name: str  # Temporary name for this experiment
    k: int = 4
    
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    
    # Metrics to calculate
    metrics: List[str] = Field(default_factory=lambda: ["semantic_similarity"])
    file_paths: List[str] = Field(default_factory=list)