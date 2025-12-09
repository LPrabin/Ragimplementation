# NotebookLM-Style RAG Application

A full-stack Retrieval-Augmented Generation (RAG) application inspired by NotebookLM. This project allows users to create "notebooks" (collections of documents), upload resources (PDF/TXT), and interact with them via a chat interface powered by Google Gemini.

## Features

- **Notebook Management**: Create, list, and delete notebooks (ChromaDB collections).
- **Multi-File Ingestion**: Upload multiple PDF or Text files simultaneously.
- **Background Processing**: File ingestion runs in the background to keep the UI responsive.
- **Status Tracking**: Track the progress of file ingestion (Processing/Completed/Failed) via SQLite.
- **Configurable Chunking**: Customize `chunk_size` and `chunk_overlap` during upload.
- **Chat Interface**: Query your notebooks using a Streamlit chat UI with context retrieval.
- **Source Citation**: View the source documents and specific chunks used to generate answers.

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI (Python)
- **LLM/Embeddings**: Google Gemini (`gemini-2.0-flash-lite`, `text-embedding-004`) via LangChain.
- **Vector Store**: ChromaDB (Persistent)
- **Database**: SQLite (for ingestion task status)
- **Orchestration**: BackgroundTasks (FastAPI)

## Prerequisites

- Python 3.10+
- A Google Cloud API Key with access to Gemini API.

## Installation

1.  **Clone the repository** (or navigate to the project directory):
    ```bash
    cd ragimplementation/vectordatabase
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup**:
    Create a `.env` file in the root directory and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

## Running the Application

This application consists of two parts: the Backend API and the Frontend UI. You need to run them in separate terminal windows.

### 1. Start the Backend
```bash
uvicorn backend.main:app --reload
```
The API will run at `http://localhost:8000`. API docs available at `http://localhost:8000/docs`.

### 2. Start the Frontend
In a new terminal window:
```bash
streamlit run frontend/app.py
```
The web interface will open automatically in your browser (usually at `http://localhost:8501`).

## Running with Docker(getting credentials error so not running)

Alternatively, you can run the entire application using Docker:

### 1. Build the Docker Image
```bash
docker build -t ragapp .
```

### 2. Run the Container
```bash
docker run -p 8000:8000 -p 8501:8501 \
  -e GOOGLE_API_KEY=your_api_key_here \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/ingestion_status.db:/app/ingestion_status.db \
  ragapp
```

**Note**: 
- Replace `your_api_key_here` with your actual Google API key
- The `-v` flags mount volumes to persist your data (ChromaDB and SQLite database)
- Backend will be available at `http://localhost:8000`
- Frontend will be available at `http://localhost:8501`

### 3. Using Docker Compose (Recommended)
If you have a `docker-compose.yml` file:
```bash
docker-compose up
```

## Usage Guide

1.  **Create a Notebook**: Use the sidebar to enter a name and create a new notebook.
2.  **Add Sources**: Go to the "Sources" tab.
    *   Select PDF or TXT files.
    *   (Optional) Adjust Chunk Size and Overlap.
    *   Click "Upload".
    *   Note the Task ID and check status if needed.
3.  **Chat**: Switch to the "Chat" tab to ask questions based on your uploaded documents.

## Directory Structure

```
├── backend/
│   ├── main.py          # FastAPI Application & Endpoints
│   ├── rag_engine.py    # Core RAG Logic (ChromaDB, LangChain)
│   └── models.py        # Pydantic Models
├── frontend/
│   └── app.py           # Streamlit UI
├── db.py                # Legacy CLI script
├── config.py            # Configuration (DB path)
├── requirements.txt     # Dependencies
└── application_flow.md  # Architecture Flow Description
```
![Project Screenshot](flowchart.png)


## Langfuse

To use Langfuse, you need to set up the following environment variables:

```bash
export LANGFUSE_PUBLIC_KEY=your_public_key_here
export LANGFUSE_SECRET_KEY=your_secret_key_here
export LANGFUSE_BASE_URL=your_base_url_here
```

To run the application with Langfuse, you can use the following command:
docker pull langfuse/langfuse
cd langfuse
docker compose up
