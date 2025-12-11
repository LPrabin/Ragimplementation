
from email.policy import default
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from typing import List
import shutil
import os
from .models import NotebookRequest, NotebookResponse, QueryRequest, QueryResponse, ResourceResponse, ResourceDeleteRequest, ExperimentConfig, EmbeddingConfig, LLMConfig, ChunkingConfig
# from .rag_engine import RAGService
from fastapi import Form
import uuid
import sqlite3
import json
from datetime import datetime
import config

from .run_experiment import RunExperiment


app = FastAPI(title="Notebook API")
# rag_service = RAGService()
experiment = RunExperiment()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/notebooks", response_model=List[str])
def list_notebooks():
    return rag_service.list_notebooks()

@app.post("/notebooks", response_model=NotebookResponse)
def create_notebook(notebook: NotebookRequest):
    rag_service.create_notebook(notebook.name)
    return NotebookResponse(name=notebook.name, resource_count=0)

@app.post("/notebooks/delete")
def delete_notebook(notebook: NotebookRequest):
    rag_service.delete_notebook(notebook.name)
    return {"status": "deleted"}


# def init_db():
#     conn = sqlite3.connect(config.DB_path)
#     cursor = conn.cursor()
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS ingestion_status (
#             id TEXT PRIMARY KEY,
#             notebook_name TEXT,
#             files TEXT,
#             status TEXT,
#             timestamp TEXT
#         )
#     ''')
#     conn.commit()
#     conn.close()

# init_db()

@app.post("/resources/add")
def add_resource(
    background_tasks: BackgroundTasks,
    notebook_name: str = Form(...), 
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(default =1000),
    chunk_overlap: int = Form(default =100)
):

    if notebook_name not in rag_service.list_notebooks():
        return HTTPException(status_code=404, detail=f"Notebook {notebook_name} does not exist")
        
    task_id = str(uuid.uuid4())
    file_paths = []
    original_filenames = []
    
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)
        original_filenames.append(file.filename)
    
    print(f"Files uploaded successfully-{file_paths}, {original_filenames}")
    
    conn = sqlite3.connect(config.DB_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO ingestion_status (id, notebook_name, files, status, timestamp) VALUES (?, ?, ?, ?, ?)",
        (task_id, notebook_name, json.dumps(original_filenames), "processing", datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    
    background_tasks.add_task(
        rag_service.process_documents, 
        notebook_name, 
        file_paths, 
        original_filenames, 
        task_id,
        chunk_size,
        chunk_overlap
        
    )
    
            
    return {"status": "Ingestion started", "task_id": task_id}

@app.get("/ingestion/task_status")
def get_ingestion_status(task_id : str):
    conn = sqlite3.connect(config.DB_path)
    cursor = conn.cursor()
    cursor.execute("SELECT status, files, notebook_name FROM ingestion_status WHERE id = ?", (task_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {"id": task_id, "status": row[0], "files": row[1], "notebook": row[2]}
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@app.post("/resources/list", response_model=List[str])
def list_resources(notebook: NotebookRequest):
    return rag_service.list_resources(notebook.name)

@app.post("/resources/delete")
def delete_resource(request: ResourceDeleteRequest):
    rag_service.delete_resource(request.notebook_name, request.resource_name)
    return {"status": "deleted"}

@app.post("/query", response_model=QueryResponse)
def query_notebook(request: QueryRequest):
    return rag_service.query_notebook(request.notebook_name, request.query, request.k)


# @app.post("/experiment")
# def run_experiment(
#     files: List[UploadFile] = File(...),
#     experiment_config: str = Form(...)
# ):
#     try:
#         # Parse config
#         config_data = json.loads(experiment_config)
#         # Validate config_data against ExperimentConfig
#         try:
#             config = ExperimentConfig(**config_data)
#         except Exception as e:
#              raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")

#         # Save files
#         task_id = config.experiment_id
#         file_paths = []
        
#         for file in files:
#             file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
#             with open(file_path, "wb") as buffer:
#                 shutil.copyfileobj(file.file, buffer)
#             file_paths.append(file_path)
            
#         print(f"Files uploaded for experiment {task_id}: {file_paths}")
        
#         # Run Experiment
#         result = experiment.run_experiment(config, file_paths)
#         return result

#     except HTTPException as e:
#         # Re-raise HTTP exceptions
#         raise e
#     except Exception as e:
#         # Cleanup if failed before experiment cleanup (though experiment handles its own cleanup, early failure might not)
#         # Note: file_paths might not be defined if error happens before loop
#         raise HTTPException(status_code=500, detail=str(e))

# main.py

# ... imports and setup ...

# @app.post("/experiment/files/upload")
# def upload_experiment_files(
#     files: List[UploadFile] = File(...)
# ):
#     """Saves files to a temporary server location and returns their paths."""
    
#     upload_time_id = str(uuid.uuid4())
#     file_paths = []
#     original_filenames = []
    
#     for file in files:
#         # Use the upload ID to group the files temporarily
#         file_path = os.path.join(UPLOAD_DIR, f"{upload_time_id}_{file.filename}")
        
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         file_paths.append(file_path)
#         original_filenames.append(file.filename)
        
#     return {
#         "status": "files_saved",
#         "upload_id": upload_time_id,
#         "file_paths": file_paths,
#         "filenames": original_filenames
#     }

# # main.py

# # ... imports and setup ...

# @app.post("/experiment/run")
# def start_experiment_run(
#     request: ExperimentConfig, # Accepts the full JSON body now
#     background_tasks: BackgroundTasks
# ):
#     """
#     Starts an experiment using pre-uploaded files. 
#     'request.file_paths' must contain the paths returned by the upload endpoint.
#     """
#     # 1. Validation: Ensure files exist before running the task
#     for path in request.file_paths:
#         if not os.path.exists(path):
#             raise HTTPException(status_code=400, detail=f"File not found on server: {path}. Did you upload them first?")

#     # 2. Add Experiment to Background Queue
#     background_tasks.add_task(experiment.run_experiment, request) 
    
#     return {
#         "status": "Experiment ingestion started in background", 
#         "experiment_id": request.experiment_id,
#         "detail": "Monitor the experiment status on the Langfuse dashboard."
#     }


@app.post("/experiment/run")
def run_experiment_endpoint(
    files: List[UploadFile] = File(...),
    experiment_id: str = Form(...),
    dataset_name: str = Form(...),
    notebook_name: str = Form(...),
    k: int = Form(default=4),
    embedding_type: str = Form(default="google"),
    embedding_model_name: str = Form(default="text-embedding-004"),
    llm_type: str = Form(default="google"),
    llm_model_name: str = Form(default="gemini-2.5-flash-lite"),
    llm_temperature: float = Form(default=0.1),
    chunking_strategy: str = Form(default="recursive"),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200)
):
    """
    Combined endpoint: Upload files AND run experiment in one call.
    All config fields are separate form fields.
    """
    file_paths = []
    
    try:
        # 1. Build config from form fields
        exp_config = ExperimentConfig(
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            notebook_name=notebook_name,
            k=k,
            embedding=EmbeddingConfig(type=embedding_type, model_name=embedding_model_name),
            llm=LLMConfig(type=llm_type, model_name=llm_model_name, temperature=llm_temperature),
            chunking=ChunkingConfig(strategy=chunking_strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            file_paths=[]
        )
        
        # 2. Save uploaded files
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, f"{experiment_id}_{file.filename}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(file_path)
        
        print(f"Files uploaded for experiment {experiment_id}: {file_paths}")
        
        # 3. Set file_paths on config
        exp_config.file_paths = file_paths
        
        # 4. Run Experiment (blocking)
        result = experiment.run_experiment(exp_config)
        return result

    except Exception as e:
        # Cleanup files on failure
        for fp in file_paths:
            if os.path.exists(fp):
                os.remove(fp)
        raise HTTPException(status_code=500, detail=str(e))