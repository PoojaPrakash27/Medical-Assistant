"""Orchestrator for the RAG pipeline.

Steps:
 - load configuration and set OpenAI environment
 - build or load a Chroma vector DB from a PDF
 - initialize a retriever and run example queries
 - generate responses using retrieved context + LLM and evaluate them
"""

import os
import json
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from chunk_data import load_and_chunk_data
from create_vector_store import create_vector_db
from retriever import initialize_retriever
from response import generate_response
from evaluation import ragas_evaluation


# --- Configuration: read API credentials from `config.json` and export to env ---
with open("config.json", 'r') as file:
    config = json.load(file)
    OPENAI_API_KEY = config.get("OPENAI_API_KEY")
    OPENAI_API_BASE = config.get("OPENAI_API_BASE")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE


# --- Initialize OpenAI client ---
client = OpenAI()


# --- Vector DB setup: create embeddings and persist if database missing ---
DB_NAME = "chroma_db"

# Initialize the embeddings wrapper for the chosen model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

if not os.path.exists(DB_NAME):
    # Default PDF to index; change as needed
    FILE_PATH = "medical_diagnosis_manual.pdf"
    # Load and split the PDF into chunks
    document_chunks = load_and_chunk_data(FILE_PATH)
    print(f"Created {len(document_chunks)} chunks.")
    # Create and persist the Chroma vector database
    create_vector_db(document_chunks, DB_NAME, embedding_model)
    print("Created vector db.")
else:
    print("Vector db already exists.")


# --- Retriever initialization ---
SEARCH_TYPE = "similarity"  # retrieval strategy (e.g., similarity, mmr)
k = 5  # number of documents to retrieve
retriever = initialize_retriever(DB_NAME, embedding_model, SEARCH_TYPE, k)


# --- Example questions to demonstrate RAG ---
question_1 = "What is the protocol for managing sepsis in a critical care unit?"
question_2 = (
    "What are the common symptoms for appendicitis, and can it be cured via medicine? "
    "If not, what surgical procedure should be followed to treat it?"
)
question_3 = (
    "What are the effective treatments or solutions for addressing sudden patchy hair loss, "
    "commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?"
)
question_4 = (
    "What treatments are recommended for a person who has sustained a physical injury to brain tissue, "
    "resulting in temporary or permanent impairment of brain function?"
)
# questions = [question_1]
questions = [question_1, question_2, question_3, question_4]


# --- Generate responses using the RAG flow (retrieve -> condition prompt -> LLM) ---
responses = []
for question in questions:
    response = generate_response(client, retriever, question)
    responses.append(response)

print("Generated responses for the questions.")


# --- Evaluate generated responses using RAGAS metrics ---
results = ragas_evaluation(retriever, questions, responses, embedding_model)
print("Done evaluating the responses. Here are the results: ")
print(results)
fr = results.to_pandas()
print(fr.head())
