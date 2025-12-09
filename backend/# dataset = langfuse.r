# dataset = langfuse.get_dataset(name="rag_bot_evals")
# chunk_sizes = [128, 256, 512]
 
# for chunk_size in chunk_sizes:
#   dataset.run_experiment(
#     name=f"Chunk precision: chunk_size {chunk_size} and chunk_overlap 0",
#     task=create_retriever_task(chunk_size=chunk_size, chunk_overlap=0),
#     evaluators=[relevant_chunks_evaluator],
#   )

# retrieval_relevance_instructions = """You are evaluating the relevance of a set of 
# chunks to a question. You will be given a QUESTION, an EXPECTED OUTPUT, and a set 
# of DOCUMENTS retrieved from the retriever.
 
# Here is the grade criteria to follow:
# (1) Your goal is to identify DOCUMENTS that are completely unrelated to the QUESTION
# (2) It is OK if the facts have SOME information that is unrelated as long as 
# it is close to the EXPECTED OUTPUT
 
# You should return a list of numbers, one for each chunk, indicating the relevance 
# of the chunk to the question.
# """
 
# ... # Define retrieval_relevance_llm
 
# # Define evaluation functions
# def relevant_chunks_evaluator(*, input, output, expected_output, metadata, **kwargs):
#   retrieval_relevance_result = retrieval_relevance_llm.invoke(
#     retrieval_relevance_instructions
#     + "\n\nQUESTION: "
#     + input["question"]
#     + "\n\nEXPECTED OUTPUT: "
#     + expected_output["answer"]
#     + "\n\nDOCUMENTS: "
#     + "\n\n".join(doc.page_content for doc in output["documents"])
#   )
 
#   # Calculate average relevance score
#   relevance_scores = retrieval_relevance_result["relevant"]
#   avg_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
 
#   return Evaluation(
#     name="retrieval_relevance", 
#     value=avg_score, 
#     comment=retrieval_relevance_result.get("explanation", "")
#   )


# Define your task function
# def my_task(*, item, **kwargs):
#     question = item["input"]
#     response = OpenAI().chat.completions.create(
#         model="gpt-4.1", messages=[{"role": "user", "content": question}]
#     )
 
#     return response.choices[0].message.content
 
 
# # Run experiment on local data
# local_data = [
#     {"input": "What is the capital of France?", "expected_output": "Paris"},
#     {"input": "What is the capital of Germany?", "expected_output": "Berlin"},
# ]
 
# result = langfuse.run_experiment(
#     name="Geography Quiz",
#     description="Testing basic functionality",
#     data=local_data,
#     task=my_task,
# )
# #  