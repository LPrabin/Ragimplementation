# import os
# import shutil
# from typing import List
# from langchain_core.documents import Document
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
# from dotenv import load_dotenv
# import chromadb
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# import sqlite3
# from datetime import datetime
# import config
# from langfuse.langchain import CallbackHandler
# from langfuse import get_client, observe, Evaluation    

# load_dotenv()

# PERSIST_DIRECTORY = "chroma_db"

# if not os.path.exists(PERSIST_DIRECTORY):
#     os.makedirs(PERSIST_DIRECTORY)

# # Initialize Langfuse handler AFTER loading environment variables
# langfuse = get_client()
# langfusehandler = CallbackHandler()

# class RAGService:
#     def __init__(self):
#         self.embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

#     def _get_vectorstore(self, collection_name: str):
#         return Chroma(
#             persist_directory=PERSIST_DIRECTORY,
#             embedding_function=self.embeddings,
#             collection_name=collection_name
#         )

#     def create_notebook(self, name: str):
#         client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
#         client.get_or_create_collection(name)

#     def list_notebooks(self) -> List[str]:
#         client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
#         return [c.name for c in client.list_collections()]

#     def delete_notebook(self, name: str):
#         client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
#         try:
#             client.delete_collection(name)
#         except Exception as e:
#             print(f"Error deleting collection {name}: {e}")
#             pass

#     def process_documents(self, notebook_name: str, file_paths: List[str], original_filenames: List[str], task_id: str, chunk_size: int = 1000, chunk_overlap: int = 100):
       

#         print(f"Starting background ingestion task {task_id}")
#         #to check notebook exists
        
#         try:
#             total_chunks = 0
#             for file_path, original_filename in zip(file_paths, original_filenames):
#                 print(f"Processing {original_filename}...")
                
#                 try:
#                     if original_filename.lower().endswith(".pdf"):
#                         loader = PyPDFLoader(file_path)
#                     else:
#                         loader = TextLoader(file_path)
#                     docs = loader.load()
                    
#                     for doc in docs:
#                         doc.metadata["source_name"] = original_filename
#                         doc.metadata["notebook"] = notebook_name

#                     #to use multiple text splitters as choice as pass parameters as arguments 
                    
#                     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#                     chunks = text_splitter.split_documents(docs)
                    
#                     if chunks:
#                         vectorstore = self._get_vectorstore(notebook_name)
#                         vectorstore.add_documents(chunks)
#                         total_chunks += len(chunks)
                        
#                 except Exception as e:
#                     print(f"Error processing file {original_filename}: {e}")
#                     # Continue to next file? Or fail whole batch? 
#                     # For now, let's log and continue
                
#                 # Cleanup temp file
#                 if os.path.exists(file_path):
#                     os.remove(file_path)

#             conn = sqlite3.connect(config.DB_path) # Need to reference global DB_path or pass it
#             cursor = conn.cursor()
#             cursor.execute("UPDATE ingestion_status SET status = ?, timestamp = ? WHERE id = ?", ("completed", datetime.now().isoformat(), task_id))
#             conn.commit()
#             conn.close()
#             print(f"Task {task_id} completed. Added {total_chunks} chunks.")
            
#         except Exception as e:
#             print(f"Task {task_id} failed: {e}")
#             conn = sqlite3.connect(config.DB_path)
#             cursor = conn.cursor()
#             cursor.execute("UPDATE ingestion_status SET status = ?, timestamp = ? WHERE id = ?", ("failed", datetime.now().isoformat(), task_id))
#             conn.commit()
#             conn.close()
        
    
#     def list_resources(self, notebook_name: str):
#         # let have persistent resource directory to update, delete from 
        
#         client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
#         try:
#             collection = client.get_collection(notebook_name)
#             # Get all metadata (limit to avoid crash if huge, but for demo it's fine)
#             result = collection.get(include=["metadatas"])
#             metadatas = result["metadatas"]
#             resources = set()
#             for m in metadatas:
#                 if "source_name" in m:
#                     resources.add(m["source_name"])
#             return list(resources)
#         except Exception:
#             return []

#     def delete_resource(self, notebook_name: str, resource_name: str):
#         vectorstore = self._get_vectorstore(notebook_name)
#         # Delete by metadata
#         # LangChain Chroma wrapper doesn't expose delete by metadata easily?
#         # Use client
#         client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
#         collection = client.get_collection(notebook_name)
#         collection.delete(where={"source_name": resource_name})
#     @observe
#     def query_notebook(self, notebook_name: str, query: str,k : int):
#         vectorstore = self._get_vectorstore(notebook_name)
#         retriever = vectorstore.as_retriever(search_kwargs={"k":k})
        
        

#         llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        
#         template = """Answer the question based only on the following context:
#         {context}

#         Question: {question}
#         """
#         prompt = ChatPromptTemplate.from_template(template)
        
#         def format_docs(docs):
#             return "\n\n".join([d.page_content for d in docs])

#         with langfuse.start_as_current_observation(
#             as_type="retriever",
#             name="rag_retriever",
#             input=query,
                
#         ) as span:
#             docs = retriever.invoke(query)
#             span.update(output = docs)

#         chain = (
#             {"context": lambda x: format_docs(docs), "question": lambda x: query}
#             | prompt
#             | llm
#             | StrOutputParser()
#         )
        
#         answer = chain.invoke(       
#             query,
#             config={"callbacks": [langfusehandler]}
#             )
#         langfuse.flush()
#         return {"answer": answer, "sources": [d.metadata.get("source_name", "unknown") for d in docs],"top3docs": [d.page_content for d in docs]}


   