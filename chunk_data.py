"""Helpers to load a PDF and split it into token-aware text chunks.

This module exposes a small API used by the indexing pipeline:
- `load_data` loads raw documents from a PDF using PyMuPDF.
- `load_and_chunk_data` splits those documents into smaller chunks suitable
  for embedding and retrieval.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import tiktoken


def load_data(file_path):
    """Load PDF pages and metadata using PyMuPDF.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        List[Document]: LangChain Document objects extracted from the PDF.
    """
    loader = PyMuPDFLoader(file_path)
    return loader.load()


# Token-aware text splitter: tuned for moderate chunk size and some overlap.
# - `chunk_size` controls max tokens per chunk (approx). Adjust to your LLM.
# - `chunk_overlap` keeps context between adjacent chunks to preserve continuity.
# - `length_function` uses tiktoken encoding to approximate token counts.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50,
    length_function=lambda x: len(
        tiktoken.get_encoding('cl100k_base').encode(x)),
)


def load_and_chunk_data(file_path):
    """Load a PDF and split it into document chunks ready for embedding.

    Args:
        file_path (str): Path to the PDF to index.

    Returns:
        List[Document]: A list of chunked `Document` objects for downstream indexing.
    """
    document_chunks = text_splitter.split_documents(load_data(file_path))
    return document_chunks
