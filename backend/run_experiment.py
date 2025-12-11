from ragas.metrics._faithfulness import faithfulness
from .models import ExperimentConfig, EmbeddingConfig, LLMConfig
import os
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse import Langfuse
from langfuse.model import DatasetItem
from datetime import datetime
import config
from langfuse import get_client, observe, Evaluation
from ragas.metrics import answer_relevancy, context_precision, context_recall
from ragas import SingleTurnSample, evaluate

load_dotenv()
langfuse = get_client()

PERSIST_DIRECTORY = "chroma_db"

class RunExperiment:
    """
    Stateless experiment runner.
    Configuration and state are passed via arguments for every method.
    """
    
    def _initialize_embeddings(self, embedding_config: EmbeddingConfig):
        if embedding_config.type == "google":
            return GoogleGenerativeAIEmbeddings(model=embedding_config.model_name)
        elif embedding_config.type == "openai":
            return OpenAIEmbeddings(model=embedding_config.model_name)
        else:
            raise ValueError(f"Unsupported embedding model type: {embedding_config.type}")

    def _initialize_llm(self, llm_config: LLMConfig):
        if llm_config.type == "google":
            return ChatGoogleGenerativeAI(
                model=llm_config.model_name,
                temperature=llm_config.temperature
            )
        elif llm_config.type == "openai":
            return ChatOpenAI(
                model=llm_config.model_name,
                temperature=llm_config.temperature
            )
        else:
            raise ValueError(f"Unsupported model type: {llm_config.type}")

    def _get_vectorstore(self, collection_name: str, embedding_function):
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function,
            collection_name=collection_name
        )

    def create_notebook(self, name: str):
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        client.get_or_create_collection(name)

    def delete_notebook(self, name: str):
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        try:
            client.delete_collection(name)
        except Exception as e:
            print(f"Error deleting collection {name}: {e}")

    def upload_documents(self, notebook_name: str, file_paths: List[str], original_filenames: List[str], embedding_function):
        docs_buffer = []
        
        for file_path, original_filename in zip(file_paths, original_filenames):
            try:
                if original_filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif original_filename.lower().endswith(".txt"):
                    loader = TextLoader(file_path)
                elif original_filename.lower().endswith(".md"):
                    loader = TextLoader(file_path) 
                elif original_filename.lower().endswith(".csv"):
                    loader = CSVLoader(file_path)
                elif original_filename.lower().endswith(".json"):
                    loader = JSONLoader(file_path, jq_schema='.', text_content=False)
                else:
                    print(f"Unsupported file type: {original_filename}")
                    continue
                
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source_name"] = original_filename
                    doc.metadata["notebook"] = notebook_name
                docs_buffer.extend(loaded_docs)
                
            except Exception as e:
                print(f"Error loading file {original_filename}: {e}")

        return docs_buffer

    def process_and_ingest(self, notebook_name: str, file_paths: List[str], original_filenames: List[str], 
                          chunking_config, embedding_function):
        
        print(f"DEBUG: process_and_ingest called with {len(file_paths)} files: {file_paths}")
        
        try:
            docs = self.upload_documents(notebook_name, file_paths, original_filenames, embedding_function)
            
            if chunking_config.strategy == "recursive":
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunking_config.chunk_size, 
                    chunk_overlap=chunking_config.chunk_overlap
                )
            elif chunking_config.strategy == "semantic":
                text_splitter = SemanticChunker(
                    embedding_function, 
                    breakpoint_threshold_type="percentile" 
                )
            else:
                 text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunking_config.chunk_size, 
                    chunk_overlap=chunking_config.chunk_overlap
                )

            chunks = text_splitter.split_documents(docs)
            
            if chunks:
                vectorstore = self._get_vectorstore(notebook_name, embedding_function)
                vectorstore.add_documents(chunks)
                print(f"Ingested {len(chunks)} chunks into {notebook_name}")
            else:
                print("DEBUG: No chunks generated. Docs were empty or split failed.")

        finally:
            print("DEBUG: Cleaning up temporary files...")
            for fp in file_paths:
                if os.path.exists(fp):
                    os.remove(fp)
                    print(f"Removed: {fp}")

    @observe
    def query_notebook(self, notebook_name: str, query: str, k: int, embedding_function, llm):
        vectorstore = self._get_vectorstore(notebook_name, embedding_function)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        
        docs = retriever.invoke(query)
        
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": lambda x: "\n\n".join([d.page_content for d in docs]), "question": lambda x: query}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(query)
        
        return {
            "answer": answer, 
            "sources": [d.metadata.get("source_name", "unknown") for d in docs],
            "top3docs": [d.page_content for d in docs]
        }

    def _get_experiment_task(self, notebook_name: str, k: int, embedding_function, llm):
        def task_wrapper(item: DatasetItem):
            user_query = item.input 
            if isinstance(user_query, (dict, list)):
                user_query = str(user_query)
                
            result = self.query_notebook(notebook_name, user_query, k, embedding_function, llm)
            
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "top3docs": result["top3docs"]
            }
        return task_wrapper

    def _create_ragas_evaluators(self, embedding_fn, llm):
        """
        Create Langfuse-compatible evaluators from Ragas metrics.
        """
        
        def faithfulness_evaluator(output, expected_output, input, metadata):
            try:
                contexts = output.get("top3docs", []) if isinstance(output, dict) else []
                
                sample = SingleTurnSample(
                    user_input=input,
                    response=output.get("answer", "") if isinstance(output, dict) else str(output),
                    retrieved_contexts=[str(c) for c in contexts]
                )
                
                result = evaluate(
                    dataset=[sample],
                    metrics=[faithfulness],
                    llm=llm,
                    embeddings=embedding_fn
                )
                
                return result.scores.get('faithfulness', [0.0])[0]
            except Exception as e:
                print(f"Faithfulness eval failed: {e}")
                return 0.0
        
        def answer_relevancy_evaluator(output, expected_output, input, metadata):
            try:
                sample = SingleTurnSample(
                    user_input=input,
                    response=output.get("answer", "") if isinstance(output, dict) else str(output),
                    retrieved_contexts=[str(c) for c in output.get("top3docs", [])] if isinstance(output, dict) else []
                )
                
                result = evaluate(
                    dataset=[sample],
                    metrics=[answer_relevancy],
                    llm=llm,
                    embeddings=embedding_fn
                )
                
                return result.scores.get('answer_relevancy', [0.0])[0]
            except Exception as e:
                print(f"Answer relevancy eval failed: {e}")
                return 0.0
        
        def context_precision_evaluator(output, expected_output, input, metadata):
            try:
                sample = SingleTurnSample(
                    user_input=input,
                    response=output.get("answer", "") if isinstance(output, dict) else str(output),
                    retrieved_contexts=[str(c) for c in output.get("top3docs", [])] if isinstance(output, dict) else [],
                    reference=expected_output if expected_output else ""
                )
                
                result = evaluate(
                    dataset=[sample],
                    metrics=[context_precision],
                    llm=llm,
                    embeddings=embedding_fn
                )
                
                return result.scores.get('context_precision', [0.0])[0]
            except Exception as e:
                print(f"Context precision eval failed: {e}")
                return 0.0
        
        def context_recall_evaluator(output, expected_output, input, metadata):
            try:
                sample = SingleTurnSample(
                    user_input=input,
                    response=output.get("answer", "") if isinstance(output, dict) else str(output),
                    retrieved_contexts=[str(c) for c in output.get("top3docs", [])] if isinstance(output, dict) else [],
                    reference=expected_output if expected_output else ""
                )
                
                result = evaluate(
                    dataset=[sample],
                    metrics=[context_recall],
                    llm=llm,
                    embeddings=embedding_fn
                )
                
                return result.scores.get('context_recall', [0.0])[0]
            except Exception as e:
                print(f"Context recall eval failed: {e}")
                return 0.0
        
        return [
            faithfulness_evaluator,
            answer_relevancy_evaluator,
            context_precision_evaluator,
            context_recall_evaluator
        ]

    def run_experiment(self, config: ExperimentConfig):
        """
        Main entry point for running an experiment synchronously.
        """
        print(f"Starting experiment {config.experiment_id}")
        
        # 1. Initialize Resources
        embedding_fn = self._initialize_embeddings(config.embedding)
        llm = self._initialize_llm(config.llm)
        
        # 2. Setup Notebook (Collection)
        self.create_notebook(config.notebook_name)
        
        try:
            # 3. Ingest Documents
            self.process_and_ingest(
                notebook_name=config.notebook_name,
                file_paths=config.file_paths,
                original_filenames=[os.path.basename(f) for f in config.file_paths],
                chunking_config=config.chunking,
                embedding_function=embedding_fn
            )
            
            # 4. Run Langfuse Experiment with Ragas evaluators
            try:
                dataset = langfuse.get_dataset(config.dataset_name)
            except Exception as e:
                 print(f"Dataset {config.dataset_name} not found in Langfuse. Ensure it exists.")
                 raise e

            print(f"Running inference on dataset: {config.dataset_name}")
            
            # Create Ragas evaluators
            evaluators = self._create_ragas_evaluators(embedding_fn, llm)
            
            result = dataset.run_experiment(
                name=f"Exp_{config.experiment_id}_{datetime.now().strftime('%Y%m%d%H%M')}",
                task=self._get_experiment_task(config.notebook_name, config.k, embedding_fn, llm),
                evaluators=evaluators
            )
            
            print(f"DEBUG: {result}")
            print("Experiment run completed successfully.")
            return {"status": "completed", "experiment_id": config.experiment_id}
            
        except Exception as e:
            print(f"Experiment failed: {e}")
            raise e
        
        finally:
            # 5. Teardown
            print(f"Tearing down notebook {config.notebook_name}")
            self.delete_notebook(config.notebook_name)
