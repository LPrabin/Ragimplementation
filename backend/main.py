from fastapi import FastAPI, UploadFile, File, HTTPException , BackgroundTasks
from typing import List
import shutil
import os
from .models import NotebookRequest, NotebookResponse, QueryRequest, QueryResponse, ResourceResponse, ResourceDeleteRequest 
from .rag_engine import RAGService
from fastapi import Form
import uuid
import sqlite3
import json
from datetime import datetime
import config





app = FastAPI(title="Notebook API")
rag_service = RAGService()

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





def init_db():
    conn = sqlite3.connect(config.DB_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ingestion_status (
            id TEXT PRIMARY KEY,
            notebook_name TEXT,
            files TEXT,
            status TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.post("/resources/add")
def add_resource(
    background_tasks: BackgroundTasks,
    notebook_name: str = Form(...), 
    files: List[UploadFile] = File(...),
    chunk_size: int = Form,
    chunk_overlap: int = Form
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
    return rag_service.query_notebook(request.notebook_name, request.query)
