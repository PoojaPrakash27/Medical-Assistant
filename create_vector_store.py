"""Create and persist a Chroma vector store from document chunks.

This module converts text `Document` chunks into embeddings using
OpenAI embeddings and stores them in a persistent Chroma directory.
"""


from langchain_chroma import Chroma


def create_vector_db(document_chunks, db_name, embedding_model):
    """Embed and persist documents in batches.

    Args:
        document_chunks (List[Document]): Chunked documents to index.
        db_name (str): Directory where Chroma will persist the DB.
        embedding_model (Embeddings): An embeddings instance (e.g., OpenAIEmbeddings).
    """

    # Batch size controls memory/throughput trade-off during indexing
    BATCH_SIZE = 100

    # Create the vectorstore and add documents in batches to avoid large memory spikes
    vectorstore = None

    for i in range(0, len(document_chunks), BATCH_SIZE):
        batch_chunks = document_chunks[i:i + BATCH_SIZE]
        if i == 0:
            # On first batch, create the Chroma DB from documents and embeddings
            vectorstore = Chroma.from_documents(
                documents=batch_chunks,
                embedding_function=embedding_model,
                persist_directory=db_name,
            )
        else:
            # For subsequent batches, append documents to the existing store
            vectorstore.add_documents(batch_chunks)

    # Function returns None; the Chroma DB is persisted to `db_name` directory
    return
