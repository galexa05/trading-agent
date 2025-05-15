"""
Vector Database Utilities for Financial Articles

This module provides functions to process financial articles and store them in a vector database.
It handles text splitting, embedding generation, and database operations using ChromaDB.
"""

import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional, Union, Tuple

# Default paths and settings
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma_db")
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "articles.csv")
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

def init_embedding_model(model_name: str = DEFAULT_MODEL) -> Any:
    """
    Initialize an embedding model.
    
    Args:
        model_name: Name of the model to use for embeddings (default: all-MiniLM-L6-v2)
        
    Returns:
        An embedding function that can be used with ChromaDB
    """
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        print("Make sure you have installed sentence-transformers: pip install sentence-transformers")
        raise


def check_database_exists(persist_directory: str = DEFAULT_DB_PATH) -> bool:
    """
    Check if a ChromaDB database exists at the specified location.
    
    Args:
        persist_directory: Directory to check
        
    Returns:
        True if the database exists, False otherwise
    """
    if not os.path.exists(persist_directory):
        return False
    
    # Check for characteristic ChromaDB files/directories
    chroma_files = ["chroma.sqlite3", "index"]
    for file in chroma_files:
        if os.path.exists(os.path.join(persist_directory, file)):
            return True
    
    return False


def init_vector_db(
    persist_directory: str = DEFAULT_DB_PATH,
    embedding_function: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL,
    collection_name: str = "financial_articles"
) -> Tuple[Any, Any]:
    """
    Initialize a ChromaDB client and collection.
    
    Args:
        persist_directory: Directory to store the database
        embedding_function: Pre-initialized embedding function (optional)
        model_name: Model name to use for embeddings if embedding_function is not provided
        collection_name: Name of the collection to create or load
        
    Returns:
        A tuple containing (chroma_client, collection)
    """
    # Create the persist directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize the client with persistence
    print(f"Initializing ChromaDB with persistence directory: {persist_directory}")
    # chroma_client = chromadb.Client(Settings(
    #     persist_directory=persist_directory,
    #     chroma_db_impl="duckdb+parquet"  # Explicitly set the storage backend
    # ))
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    # Get or create the embedding function
    if embedding_function is None:
        embedding_function = init_embedding_model(model_name)
    
    # Get or create the collection
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        print(f"Loaded existing collection '{collection_name}'")
    except Exception as e:
        print(f"Could not load existing collection: {e}")
        print(f"Creating new collection '{collection_name}'")
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        print(f"Created new collection '{collection_name}'")
    
    return chroma_client, collection


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE, 
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter for chunking documents.
    
    Args:
        chunk_size: The size of each chunk in characters
        chunk_overlap: The overlap between chunks in characters
        
    Returns:
        A configured text splitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def process_article_text(
    text: str, 
    article_metadata: Dict[str, Any],
    text_splitter: RecursiveCharacterTextSplitter
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Process an article's text by splitting it into chunks and creating metadata.
    
    Args:
        text: The article text to process
        article_metadata: Metadata for the article
        text_splitter: Text splitter to use
        
    Returns:
        A tuple of (chunk_ids, chunks, metadatas)
    """
    if pd.isna(text) or text == "":
        return [], [], []
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    # Generate IDs for each chunk
    article_id = article_metadata.get('article_id', article_metadata.get('id', None))
    if article_id is None:
        raise ValueError("Article metadata must contain 'article_id' or 'id'")
        
    chunk_ids = [f"{article_id}_{i}" for i in range(len(chunks))]
    
    # Prepare metadata for each chunk
    metadatas = [{
        'article_id': article_id,
        'title': article_metadata.get('title', ''),
        'pubDate': article_metadata.get('pubDate', ''),
        'source': article_metadata.get('source_name', ''),
        'link': article_metadata.get('link', ''),
        'summary': article_metadata.get('summary', ''),
        'creator': article_metadata.get('creator', ''),
        'chunk_index': i,
        'total_chunks': len(chunks)
    } for i in range(len(chunks))]
    
    return chunk_ids, chunks, metadatas


def load_articles_to_vectordb(
    csv_path: str = DEFAULT_CSV_PATH,
    collection: Optional[Any] = None,
    chroma_client: Optional[Any] = None,
    text_splitter: Optional[RecursiveCharacterTextSplitter] = None,
    batch_size: int = 100,
    max_articles: Optional[int] = None,
    persist_directory: str = DEFAULT_DB_PATH
) -> int:
    """
    Load articles from a CSV file into a vector database.
    
    Args:
        csv_path: Path to the CSV file containing articles
        collection: ChromaDB collection to load articles into
        chroma_client: ChromaDB client
        text_splitter: Text splitter to use for chunking
        batch_size: Number of articles to process at once
        max_articles: Maximum number of articles to load (None for all)
        persist_directory: Directory to persist the database
        
    Returns:
        Number of articles processed
    """
    # Create a new client and collection if not provided
    if collection is None or chroma_client is None:
        chroma_client, collection = init_vector_db(persist_directory=persist_directory)
    
    # Create a text splitter if not provided
    if text_splitter is None:
        text_splitter = create_text_splitter()
    
    # Load the CSV data
    df = pd.read_csv(csv_path)
    
    # Limit the number of articles if specified
    if max_articles is not None:
        df = df.head(max_articles)
    
    # Process articles in batches
    total_articles = 0
    total_chunks = 0
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        all_ids = []
        all_documents = []
        all_metadatas = []
        
        for _, row in batch.iterrows():
            article_metadata = row.to_dict()
            
            # Skip if text is missing
            if 'text' not in article_metadata or pd.isna(article_metadata['text']) or article_metadata['text'] == "":
                continue
                
            # Process the article text
            chunk_ids, chunks, metadatas = process_article_text(
                article_metadata['text'], 
                article_metadata,
                text_splitter
            )
            
            if chunks:  # Only add if there are chunks
                all_ids.extend(chunk_ids)
                all_documents.extend(chunks)
                all_metadatas.extend(metadatas)
                total_articles += 1
                total_chunks += len(chunks)
        
        # Add chunks to the collection
        if all_ids:
            collection.add(
                ids=all_ids,
                documents=all_documents,
                metadatas=all_metadatas
            )
            
            # Explicitly persist after each batch
            # chroma_client.persist()
            print(f"Persisted batch to disk at {persist_directory}")
        
        print(f"Processed batch {i//batch_size + 1}, articles: {total_articles}, chunks: {total_chunks}")
    
    print(f"Finished processing {total_articles} articles with {total_chunks} chunks total")
    return total_articles


def query_vector_db(
    query_text: str,
    collection: Optional[Any] = None,
    chroma_client: Optional[Any] = None,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
    include_metadata: bool = True,
    include_documents: bool = True,
    persist_directory: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Query the vector database.
    
    Args:
        query_text: Text to query for
        collection: ChromaDB collection to query
        chroma_client: ChromaDB client
        n_results: Number of results to return
        where: Metadata filter
        include_metadata: Whether to include metadata in results
        include_documents: Whether to include documents in results
        persist_directory: Directory where the database is stored
        
    Returns:
        Query results
    """
    # Create a new client and collection if not provided
    if collection is None:
        chroma_client, collection = init_vector_db(persist_directory=persist_directory)
    
    # Execute the query
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
        include=["metadatas", "documents", "distances"] if include_metadata and include_documents else 
               ["metadatas", "distances"] if include_metadata else
               ["documents", "distances"] if include_documents else
               ["distances"]
    )
    
    return results


def get_article_by_id(
    article_id: str,
    collection: Optional[Any] = None,
    chroma_client: Optional[Any] = None,
    reconstruct_full_text: bool = True,
    persist_directory: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Retrieve an article by its ID.
    
    Args:
        article_id: ID of the article to retrieve
        collection: ChromaDB collection to query
        chroma_client: ChromaDB client
        reconstruct_full_text: Whether to reconstruct the full article text from chunks
        persist_directory: Directory where the database is stored
        
    Returns:
        Article data including metadata and text
    """
    # Create a new client and collection if not provided
    if collection is None:
        chroma_client, collection = init_vector_db(persist_directory=persist_directory)
    
    # Query for chunks belonging to the article
    results = collection.get(
        where={"article_id": article_id},
        include=["metadatas", "documents"]
    )
    
    if not results["ids"]:
        return {"error": f"Article with ID {article_id} not found"}
    
    # Extract metadata from the first chunk (should be the same for all chunks)
    article_metadata = results["metadatas"][0].copy()
    
    # Remove chunk-specific metadata
    if "chunk_index" in article_metadata:
        del article_metadata["chunk_index"]
    if "total_chunks" in article_metadata:
        del article_metadata["total_chunks"]
    
    if reconstruct_full_text and results["documents"]:
        # Sort chunks by chunk_index
        sorted_chunks = sorted(
            zip(results["metadatas"], results["documents"]),
            key=lambda x: x[0].get("chunk_index", 0)
        )
        
        # Reconstruct the full text
        full_text = " ".join(chunk[1] for chunk in sorted_chunks)
        article_metadata["text"] = full_text
    
    return article_metadata


def delete_collection(
    collection_name: str = "financial_articles",
    persist_directory: str = DEFAULT_DB_PATH
) -> None:
    """
    Delete a collection from the database.
    
    Args:
        collection_name: Name of the collection to delete
        persist_directory: Directory where the database is stored
    """
    chroma_client = chromadb.Client(Settings(persist_directory=persist_directory))
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Deleted collection '{collection_name}'")
        chroma_client.persist()
        print(f"Changes persisted to disk")
    except Exception as e:
        print(f"Error deleting collection: {e}")


if __name__ == "__main__":
    print("Vector database utilities loaded. Import this module to use its functions.")
