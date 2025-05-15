#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vector Database Manager Script
------------------------------
This script provides a comprehensive interface for managing a vector database
of financial articles. It includes functions for initializing the database,
processing articles, loading them into the database, and querying the database.
"""

import os
import sys
import argparse
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Add the parent directory to the path so we can import from trading-agent
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts import vector_db as vdb

DEFAULT_DB_PATH = "../data/chroma_db"
DEFAULT_ARTICLES_PATH = "../data/articles.csv"
DEFAULT_COLLECTION_NAME = "financial_articles"


class VectorDatabaseManager:
    """
    A class to manage vector database operations for financial articles.
    """

    def __init__(self, 
                 db_path: str = DEFAULT_DB_PATH, 
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 force_recreate: bool = False):
        """
        Initialize the Vector Database Manager.
        
        Args:
            db_path: Path to store the ChromaDB database
            collection_name: Name of the collection to use
            embedding_model_name: Name of the sentence transformer model to use
            chunk_size: Size of text chunks for splitting articles
            chunk_overlap: Overlap between text chunks
            force_recreate: If True, delete and recreate the collection if it exists
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create the embedding function
        self.embedding_function = vdb.get_sentence_transformer_embedding_function(
            model_name=embedding_model_name
        )
        
        # Create the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Initialize database
        self._init_database(force_recreate)
    
    def _init_database(self, force_recreate: bool = False):
        """
        Initialize the vector database and get the collection.
        
        Args:
            force_recreate: If True, delete and recreate the collection if it exists
        """
        # Create db directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Delete collection if requested
        if force_recreate:
            try:
                vdb.delete_collection(
                    collection_name=self.collection_name,
                    persist_directory=self.db_path
                )
                print(f"Deleted existing collection: {self.collection_name}")
            except Exception as e:
                print(f"Error deleting collection: {str(e)}")
        
        # Initialize the database and collection
        self.chroma_client, self.collection = vdb.init_vector_db(
            collection_name=self.collection_name,
            persist_directory=self.db_path,
            embedding_function=self.embedding_function
        )
        
        # Check if the collection exists
        collection_count = self.collection.count()
        print(f"Collection '{self.collection_name}' contains {collection_count} documents")
    
    def process_articles_file(self, 
                             csv_path: str = DEFAULT_ARTICLES_PATH, 
                             batch_size: int = 10,
                             max_articles: Optional[int] = None) -> int:
        """
        Load articles from a CSV file and add them to the vector database.
        
        Args:
            csv_path: Path to the CSV file containing articles
            batch_size: Number of articles to process in each batch
            max_articles: Maximum number of articles to process (None for all)
            
        Returns:
            int: Number of articles processed
        """
        # Check if the file exists
        if not os.path.exists(csv_path):
            print(f"Error: File not found: {csv_path}")
            return 0
            
        # Load articles into the vector database
        num_articles_processed = vdb.load_articles_to_vectordb(
            csv_path=csv_path,
            collection=self.collection,
            chroma_client=self.chroma_client,
            text_splitter=self.text_splitter,
            batch_size=batch_size,
            max_articles=max_articles
        )
        
        print(f"Processed {num_articles_processed} articles")
        print(f"Collection now contains {self.collection.count()} documents")
        return num_articles_processed
    
    def query_database(self, 
                      query: str, 
                      n_results: int = 5, 
                      metadata_filter: Optional[Dict[str, str]] = None,
                      include_summary: bool = True):
        """
        Query the vector database for similar articles.
        
        Args:
            query: The search query
            n_results: Number of results to return
            metadata_filter: Filter results by metadata fields (dict)
            include_summary: Include summary in output
            
        Returns:
            Dict: Query results containing documents, metadatas, distances, and ids
        """
        results = vdb.query_vector_db(
            query_text=query,
            collection=self.collection,
            n_results=n_results,
            metadata_filter=metadata_filter
        )
        
        self._display_search_results(results, include_summary=include_summary)
        return results
    
    def _display_search_results(self, results, limit: int = 3, include_summary: bool = True):
        """
        Display search results in a readable format.
        
        Args:
            results: Results from query_vector_db
            limit: Maximum number of results to display
            include_summary: Whether to include summary in output
        """
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        
        if not documents:
            print("No results found")
            return
            
        print(f"Found {len(documents)} results. Showing top {min(limit, len(documents))}:\n")
        
        for i in range(min(limit, len(documents))):
            print(f"Result #{i+1} (Similarity: {1 - distances[i]:.4f})")
            print(f"Title: {metadatas[i].get('title', 'N/A')}")
            print(f"Source: {metadatas[i].get('source', 'N/A')}")
            print(f"Date: {metadatas[i].get('pubDate', 'N/A')}")
            print(f"Chunk: {metadatas[i].get('chunk_index', 0) + 1} of {metadatas[i].get('total_chunks', 'N/A')}")
            
            if include_summary:
                print(f"Link: {metadatas[i].get('link', 'N/A')}")
                print(f"Creator: {metadatas[i].get('creator', 'N/A')}")
                print(f"Summary: {metadatas[i].get('summary', 'N/A')}")
                print(f"Article ID: {metadatas[i].get('article_id', 'N/A')}")
            
            print(f"\nEXCERPT:\n{documents[i][:300]}...\n")
            print("-" * 80)


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Vector Database Manager for Financial Articles')
    
    # Setup subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Initialize database command
    init_parser = subparsers.add_parser('init', help='Initialize the vector database')
    init_parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, 
                            help='Path to store the ChromaDB database')
    init_parser.add_argument('--collection-name', type=str, default=DEFAULT_COLLECTION_NAME,
                            help='Name of the collection to use')
    init_parser.add_argument('--force-recreate', action='store_true',
                            help='Delete and recreate the collection if it exists')
    
    # Load articles command
    load_parser = subparsers.add_parser('load', help='Load articles into the vector database')
    load_parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH,
                           help='Path to the ChromaDB database')
    load_parser.add_argument('--collection-name', type=str, default=DEFAULT_COLLECTION_NAME,
                           help='Name of the collection to use')
    load_parser.add_argument('--articles-path', type=str, default=DEFAULT_ARTICLES_PATH,
                           help='Path to the CSV file containing articles')
    load_parser.add_argument('--batch-size', type=int, default=10,
                           help='Number of articles to process in each batch')
    load_parser.add_argument('--max-articles', type=int, default=None,
                           help='Maximum number of articles to process')
    
    # Query database command
    query_parser = subparsers.add_parser('query', help='Query the vector database')
    query_parser.add_argument('query', type=str, help='The search query')
    query_parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH,
                            help='Path to the ChromaDB database')
    query_parser.add_argument('--collection-name', type=str, default=DEFAULT_COLLECTION_NAME,
                            help='Name of the collection to use')
    query_parser.add_argument('--n-results', type=int, default=5,
                            help='Number of results to return')
    query_parser.add_argument('--source', type=str, default=None,
                            help='Filter results by source')
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'init':
        # Initialize database
        manager = VectorDatabaseManager(
            db_path=args.db_path,
            collection_name=args.collection_name,
            force_recreate=args.force_recreate
        )
        print(f"Vector database initialized at {args.db_path}")
        
    elif args.command == 'load':
        # Load articles
        manager = VectorDatabaseManager(
            db_path=args.db_path,
            collection_name=args.collection_name
        )
        
        num_processed = manager.process_articles_file(
            csv_path=args.articles_path,
            batch_size=args.batch_size,
            max_articles=args.max_articles
        )
        
        print(f"Processed {num_processed} articles")
        
    elif args.command == 'query':
        # Query database
        manager = VectorDatabaseManager(
            db_path=args.db_path,
            collection_name=args.collection_name
        )
        
        metadata_filter = None
        if args.source:
            metadata_filter = {"source": args.source}
            
        manager.query_database(
            query=args.query,
            n_results=args.n_results,
            metadata_filter=metadata_filter
        )
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
