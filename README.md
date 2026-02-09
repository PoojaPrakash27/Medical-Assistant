# Medical PDF Assistant - Retrieval-Augmented Generation (RAG)

**Project Overview**
- **Problem:**: This project converts a medical reference PDF into a searchable vector knowledge base and answers clinical questions by combining retrieval from that knowledge base with an LLM (RAG). It helps surface concise, context-grounded answers from manual excerpts instead of relying solely on the model's parametric memory.

**Key Features**
- **Document chunking**: Splits PDFs into token-aware chunks for accurate retrieval (see [chunk_data.py](chunk_data.py)).
- **Embeddings & Vector DB**: Creates and persists embeddings using OpenAI embeddings and Chroma as the vector store ([create_vector_store.py](create_vector_store.py)).
- **Retriever**: Wraps the vector store into an easy retriever with configurable search strategy and top-k retrieval ([retriever.py](retriever.py)).
- **RAG response generation**: Builds a system+user prompt using retrieved context and queries an OpenAI chat model to generate grounded answers ([response.py](response.py)).
- **Evaluation**: Runs automated RAG evaluation (RAGAS) measuring faithfulness, relevancy, and context precision ([evaluation.py](evaluation.py)).
- **Config-driven**: Uses `config.json` for API base and key, and `pyproject.toml` for dependencies.

**Project Structure**
- **main.py**: Orchestrates the pipeline — reads `config.json`, builds or loads the vector DB, initializes the retriever, runs queries and evaluation. See [main.py](main.py).
- **chunk_data.py**: Loads PDFs (via PyMuPDF) and splits text into chunks using token-aware splitting. See [chunk_data.py](chunk_data.py).
- **create_vector_store.py**: Converts document chunks to embeddings and stores them into a persistent Chroma DB directory. See [create_vector_store.py](create_vector_store.py).
- **retriever.py**: Initializes and configures the Chroma retriever used for nearest-neighbour lookups. See [retriever.py](retriever.py).
- **response.py**: Assembles system and user prompts, retrieves relevant chunks and calls the LLM to produce final answers. See [response.py](response.py).
- **evaluation.py**: Uses `ragas` to evaluate generated answers against metrics like faithfulness and relevancy. See [evaluation.py](evaluation.py).
- **config.json**: Small JSON file containing `OPENAI_API_KEY` and `OPENAI_API_BASE` used by `main.py`.
- **pyproject.toml**: Declares project metadata and Python dependencies.

**Quick Start**
1. Populate `config.json` with your OpenAI API credentials and base URL.
2. Install dependencies (recommended to use a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

3. Place your PDF to index in the project root (default used by `main.py` is `medical_diagnosis_manual.pdf`) and run:

```bash
python main.py
```

This will create the `chroma_db/` directory (if missing), run the RAG pipeline for sample questions, and print evaluation results.

**Potential Improvements**
- **Config & secrets**: Use environment-based secrets or `.env` + a safer secrets manager instead of storing keys in `config.json`.
- **Modular CLI**: Add a CLI with commands to `index`, `query`, and `evaluate` so users can run individual steps.
- **Batch indexing & updates**: Add support for incremental updates and metadata (source, page) preservation.
- **More robust retrieval**: Experiment with hybrid search (embedding + sparse BM25) and re-ranking.
- **Tests & CI**: Add unit tests for chunking, indexing, and retrieval; add GitHub Actions for reproducible CI checks.
- **Sanity checks & validation**: Validate PDF text extraction and handle non-text PDFs or OCR needs.

**Where to look next**
- Start reading the pipeline in [main.py](main.py). To change the embedding model or LLM, update `create_vector_store.py` and `response.py` respectively.

---
