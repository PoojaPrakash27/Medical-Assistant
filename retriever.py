"""Initialize a retriever from a persistent Chroma vector store.

Provides a simple wrapper around Chroma for retrieving relevant documents
given a query string (used by the RAG pipeline).
"""

from langchain_chroma import Chroma


def initialize_retriever(db_name, model, search_type, k):
    """Load a Chroma vector store and wrap it in a retriever interface.

    Args:
        db_name (str): Directory containing the persisted Chroma database.
        model (str | Embedding): Embedding model or embedding function to use.
        search_type (str): Retrieval strategy (e.g., "similarity", "mmr").
        k (int): Number of top documents to retrieve per query.

    Returns:
        Retriever: A LangChain Retriever object wrapping the Chroma vectorstore.
    """
    # Load the persisted Chroma DB and configure embeddings
    vectorstore = Chroma(
        persist_directory=db_name,
        embedding_function=model,
    )

    # Wrap the vectorstore in a retriever with configurable search strategy
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )

    return retriever
