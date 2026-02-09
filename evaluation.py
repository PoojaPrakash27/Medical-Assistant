"""Evaluate RAG system using RAGAS metrics.

RAGAS (Retrieval-Augmented Generation Assessment) provides automated
evaluation of RAG responses across multiple quality dimensions:
- Faithfulness: Does the answer align with the retrieved context?
- AnswerRelevancy: Is the answer relevant to the question?
- ContextPrecision: Is the retrieved context relevant to the question?
"""

from langchain_openai import ChatOpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
)


def ragas_evaluation(retriever, questions, responses, embedding_model):
    """Evaluate generated responses against RAGAS quality metrics.

    Args:
        retriever: Retriever instance for fetching relevant documents.
        questions (List[str]): List of input questions.
        responses (List[str]): List of generated responses (one per question).
        embedding_model (str | Embedding): Embedding model for evaluation.

    Returns:
        EvaluationResults: A dictionary-like object with scores for each metric.
    """
    # Initialize an LLM (evaluator) for running the RAGAS metrics
    # Uses GPT-4o with deterministic temperature for reproducible evaluations
    evaluator_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Initialize the RAGAS metrics to compute
    faithfulness = Faithfulness()  # Measure answer fidelity to context
    answer_relevancy = AnswerRelevancy()  # Measure answer relevance to question
    context_precision = LLMContextPrecisionWithoutReference()  # Measure context relevance

    # Retrieve top-k documents as context for each question
    # This mirrors what the RAG system retrieves during response generation
    contexts = [
        [chunk.page_content for chunk in retriever.invoke(question)]
        for question in questions
    ]

    # Wrap the evaluation data into a HuggingFace Dataset for RAGAS
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": responses,
        "contexts": contexts,
        "reference": questions
    })

    # Run RAGAS evaluation on the dataset with the selected metrics
    results = evaluate(
        ragas_dataset,
        metrics=[
            answer_relevancy,
            context_precision,
            faithfulness,
        ],
        llm=evaluator_llm,
        embeddings=embedding_model
    )
    return results
