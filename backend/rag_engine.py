
import os
import shutil
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import sqlite3
from datetime import datetime
import config
from langfuse.langchain import CallbackHandler
from langfuse import get_client, observe
from autoevals import ContextRecall, AnswerCorrectness, AnswerSimilarity

# from langchain.agents import tools 

load_dotenv()

PERSIST_DIRECTORY = "chroma_db"

if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)

# Initialize Langfuse handler AFTER loading environment variables
langfuse = get_client()
langfusehandler = CallbackHandler()

class RAGService:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    def _get_vectorstore(self, collection_name: str):
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )

    def create_notebook(self, name: str):
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        client.get_or_create_collection(name)

    def list_notebooks(self) -> List[str]:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        return [c.name for c in client.list_collections()]

    def delete_notebook(self, name: str):
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        try:
            client.delete_collection(name)
        except Exception as e:
            print(f"Error deleting collection {name}: {e}")
            pass

    def process_documents(self, notebook_name: str, file_paths: List[str], original_filenames: List[str], task_id: str, chunk_size: int = 1000, chunk_overlap: int = 100):
       

        print(f"Starting background ingestion task {task_id}")
        #to check notebook exists
        
        try:
            total_chunks = 0
            for file_path, original_filename in zip(file_paths, original_filenames):
                print(f"Processing {original_filename}...")
                
                try:
                    if original_filename.lower().endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                    else:
                        loader = TextLoader(file_path)
                    docs = loader.load()
                    
                    for doc in docs:
                        doc.metadata["source_name"] = original_filename
                        doc.metadata["notebook"] = notebook_name

                    #to use multiple text splitters as choice as pass parameters as arguments 
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    chunks = text_splitter.split_documents(docs)
                    
                    if chunks:
                        vectorstore = self._get_vectorstore(notebook_name)
                        vectorstore.add_documents(chunks)
                        total_chunks += len(chunks)
                        
                except Exception as e:
                    print(f"Error processing file {original_filename}: {e}")
                    # Continue to next file? Or fail whole batch? 
                    # For now, let's log and continue
                
                # Cleanup temp file
                if os.path.exists(file_path):
                    os.remove(file_path)

            conn = sqlite3.connect(config.DB_path) # Need to reference global DB_path or pass it
            cursor = conn.cursor()
            cursor.execute("UPDATE ingestion_status SET status = ?, timestamp = ? WHERE id = ?", ("completed", datetime.now().isoformat(), task_id))
            conn.commit()
            conn.close()
            print(f"Task {task_id} completed. Added {total_chunks} chunks.")
            
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            conn = sqlite3.connect(config.DB_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE ingestion_status SET status = ?, timestamp = ? WHERE id = ?", ("failed", datetime.now().isoformat(), task_id))
            conn.commit()
            conn.close()
        
    
    def list_resources(self, notebook_name: str):
        # let have persistent resource directory to update, delete from 
        
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        try:
            collection = client.get_collection(notebook_name)
            # Get all metadata (limit to avoid crash if huge, but for demo it's fine)
            result = collection.get(include=["metadatas"])
            metadatas = result["metadatas"]
            resources = set()
            for m in metadatas:
                if "source_name" in m:
                    resources.add(m["source_name"])
            return list(resources)
        except Exception:
            return []

    def delete_resource(self, notebook_name: str, resource_name: str):
        vectorstore = self._get_vectorstore(notebook_name)
        # Delete by metadata
        # LangChain Chroma wrapper doesn't expose delete by metadata easily?
        # Use client
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = client.get_collection(notebook_name)
        collection.delete(where={"source_name": resource_name})
    @observe
    def query_notebook(self, notebook_name: str, query: str,k : int):
        vectorstore = self._get_vectorstore(notebook_name)
        retriever = vectorstore.as_retriever(search_kwargs={"k":k})
        
        

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        with langfuse.start_as_current_observation(
            as_type="retriever",
            name="rag_retriever",
            input=query,
                
        ) as span:
            docs = retriever.invoke(query)
            span.update(output = docs)

        chain = (
            {"context": lambda x: format_docs(docs), "question": lambda x: query}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(       
            query,
            config={"callbacks": [langfusehandler]}
            )
        langfuse.flush()
        return {"answer": answer, "sources": [d.metadata.get("source_name", "unknown") for d in docs],"top3docs": [d.page_content for d in docs]}

    def run_experiment(self, notebook_name: str, dataset_name: str, chunk_size: List[int], chunk_overlap: List[int], k: int):
        """
        Run experiments on a dataset to test different chunking configurations.
        
        Note: This is a work-in-progress implementation. The current approach has limitations:
        - Chunking happens during document ingestion, not during querying
        - To properly test different chunk sizes, you would need to re-ingest documents
        - The task function needs to match the expected signature: def task(*, item, **kwargs)
        - Evaluators need to be imported from autoevals or defined as custom functions
        """
        
        # dataset = langfuse.get_dataset(name=dataset_name)
        dataset = get_client()
        
        # Define the task function that will be called for each dataset item
        def query_task(*, item, **kwargs):
            """Task function that processes each dataset item."""
            query = item.input
          
            result = self.query_notebook(notebook_name, query, k)
            return result["answer"]
        
        local_data = [
            {"input": "What is the core innovation of the LatentMAS framework?", "expected_output": "It enables LLM agents to collaborate directly within the continuous latent space, rather than relying on text-based mediation for reasoning and communication."},
            {"input": "What specific mechanism does LatentMAS use for information exchange between agents?", "expected_output": "It uses a shared latent working memory that preserves and transfers each agent's internal representations (last-layer hidden embeddings), ensuring lossless information exchange."},
            {"input": "What are the key efficiency gains of LatentMAS compared to text-based multi-agent systems?", "expected_output": "It reduces output token usage by 70.8%-83.7% and provides 4x-4.3x faster end-to-end inference."},
            {"input": "Does LatentMAS require additional training to achieve its results?", "expected_output": "No, it is an end-to-end training-free framework."}
        ]
        for cs, co in zip(chunk_size, chunk_overlap):
            result = dataset.run_experiment(
                name="local_data_experiment",
                description="Testing basic functionality",
                data=local_data,
                task=query_task,
                evaluators=[
                        ContextRecall(),         
                        AnswerCorrectness(),     
                        AnswerSimilarity()
                    ]
                )
            print(f"Experiment completed for chunk_size={cs}, chunk_overlap={co}")
        
        return {"status": "experiments completed"}
        
        

        # Run experiment for each chunking configuration
        # Note: This doesn't actually change the chunking - that happens during ingestion
        # for cs, co in zip(chunk_size, chunk_overlap):
        #     result = dataset.run_experiment(
        #         name=f"chunk precision: chunk_size {cs} and chunk overlap {co}",
        #         task=query_task,
        #         evaluators=[
        #             ContextRecall(),         # Needs to be imported/defined
        #             AnswerCorrectness(),     # Needs to be imported/defined
        #             AnswerSimilarity()
        #         ]
        #     )
        #     print(f"Experiment completed for chunk_size={cs}, chunk_overlap={co}")
        
        # return {"status": "experiments completed"}
        
    # @observe
    # def relevant_chunks_evaluator(*, input, output, expected_output, metadata, **kwargs):
    #     retrieval_relevance_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    #     retrieval_relevance_instructions = """You are evaluating the relevance of a set of 
    #     chunks to a question. You will be given a QUESTION, an EXPECTED OUTPUT, and a set 
    #     of DOCUMENTS retrieved from the retriever.
        
    #     Here is the grade criteria to follow:
    #     (1) Your goal is to identify DOCUMENTS that are completely unrelated to the QUESTION
    #     (2) It is OK if the facts have SOME information that is unrelated as long as 
    #     it is close to the EXPECTED OUTPUT
        
    #     You should return a list of numbers, one for each chunk, indicating the relevance 
    #     of the chunk to the question.
    #     """
    #     retrival_relevance_result = retrieval_relevance_llm.invoke(
    #         retrieval_relevance_instructions
    #         + "\n\nQUESTION: "
    #         + input["question"]
    #         + "\n\nEXPECTED OUTPUT: "
    #         + expected_output["answer"]
    #         + "\n\nDOCUMENTS: "
    #         + "\n\n".join(doc.page_content for doc in output["documents"])
    #     )
    #     relevance_score = retrival_relevance_result["relevant"]
    #     avg_score = sum(relevance_score) / len(relevance_score) if relevance_scores else 0
    #     return Evaluation(
    #         name="retrieval_relevance", 
    #         value=avg_score, 
    #         comment=retrieval_relevance_result.get("explanation", "")
    #     )

    
    # @observe
    # def test_passing_listof_chunks_size(notebook_name: str, query: str, k: int, chunk_size: int, chunk_overlap: int):


    #     for chunk in chunk_size:
    #         for chunk_overlap in chunk_overlap:
    #             text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    #             chunks = text_splitter.split_documents(docs)
    #             query_notebook(notebook_name, query, k, chunks, chunk, chunk_overlap)



        
