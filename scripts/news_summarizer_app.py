#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
News Summarizer Streamlit App
-----------------------------
A Streamlit application that displays summarized financial news articles
from a vector database. Users can select different stock portfolios
and view relevant news summaries.
"""

import os
import sys
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

# Import project modules
from scripts.vector_db_manager import VectorDatabaseManager
from scripts.finetune_summarizer import SummarizerFineTuner

# Set up paths
DB_PATH = os.path.join(project_dir, "data", "chroma_db")
COLLECTION_NAME = "financial_articles"
FINETUNED_MODEL_DIR = os.path.join(project_dir, "models", "finetuned_summarizer")
BASELINE_MODEL = "facebook/bart-large-cnn"  # Default baseline model for summarization

@st.cache_data
def get_available_stocks(_db_manager):
    """
    Get a list of available stocks in the database.
    
    Args:
        _db_manager: VectorDatabaseManager instance
        
    Returns:
        List of stock symbols
    """
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "ADBE", "CRM", "INTC"]
    
    # Extract stock symbols from metadata
    stocks = set()
    if results and 'metadatas' in results:
        metadatas = results['metadatas']
        # Handle nested lists
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
            
        for metadata in metadatas:
            if 'stock_symbol' in metadata and metadata['stock_symbol']:
                stocks.add(metadata['stock_symbol'].upper())
    return sorted(list(stocks))

@st.cache_data
def get_news_for_stock(_db_manager, stock_symbol: str, limit: int = 10):
    """
    Get news articles for a specific stock.
    
    Args:
        _db_manager: VectorDatabaseManager instance
        stock_symbol: Stock symbol to filter by
        limit: Maximum number of articles to return
        
    Returns:
        List of news articles with summaries
    """
    # Query with filter for the stock symbol
    results = _db_manager.query_database(
        query=f"news about {stock_symbol}",
        n_results=limit,
        # metadata_filter={"stock_symbol": stock_symbol} if stock_symbol else None,
        include_summary=True
    )
    
    news_items = []
    if results and 'documents' in results and results['documents']:
        # Handle nested lists
        documents = results['documents']
        if documents and isinstance(documents[0], list):
            documents = documents[0]
            
        metadatas = results['metadatas']
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        
        # Process each article
        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Create news item with available metadata
            news_item = {
                'text': doc,
                'title': metadata.get('title', 'No Title Available'),
                'summary': metadata.get('summary', ''),
                'source': metadata.get('source', 'Unknown Source'),
                'pubDate': metadata.get('pubDate', ''),
                'link': metadata.get('link', '#'),
                'stock_symbol': metadata.get('stock_symbol', '')
            }
            news_items.append(news_item)
    
    return news_items

def format_date(date_str: str) -> str:
    """Format date string to a more readable format."""
    try:
        if not date_str:
            return ""
        # Try different date formats
        for fmt in ['%a, %d %b %Y %H:%M:%S %z', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%B %d, %Y')
            except:
                continue
        return date_str
    except:
        return date_str

def generate_new_summary(model_path, article_text, use_baseline=False):
    """
    Generate a new summary for an article using either the fine-tuned model or a baseline model.
    
    Args:
        model_path: Path to the fine-tuned model or name of baseline model
        article_text: Text to summarize
        use_baseline: Whether to use the baseline model instead of finetuned model
        
    Returns:
        Generated summary
    """
    try:
        # Import required libraries
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path,early_stopping=True)
        
        # Tokenize the text
        inputs = tokenizer(article_text, return_tensors="pt", max_length=1024, truncation=True)
        
        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return "Unable to generate summary."

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Financial News Summarizer",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📰 Financial News Summarizer")
    st.write("View summarized news articles for different stocks in your portfolio.")
    
    # Initialize database manager
    try:
        _db_manager = VectorDatabaseManager(
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME
        )
        
        # Get available stocks
        stocks = get_available_stocks(_db_manager)
        
        if not stocks:
            st.warning("No stock data found in the database. Please make sure your vector database contains financial news articles with stock symbols.")
            return
            
        # Create sidebar for filtering
        st.sidebar.title("Options")
        
        # Stock selection
        selected_stock = st.sidebar.selectbox(
            "Select Stock", 
            options=["All Stocks"] + stocks,
            index=0
        )
        
        # Number of articles to show
        num_articles = st.sidebar.slider(
            "Number of Articles", 
            min_value=1,
            max_value=20,
            value=5
        )
        
        # Option to regenerate summaries
        regenerate_summaries = st.sidebar.checkbox(
            "Regenerate Summaries",
            value=False
        )
        
        # Model selection for summarization
        use_baseline_model = st.sidebar.radio(
            "Summarization Model",
            options=["Fine-tuned Model", "Baseline Model"],
            index=0,
            help="Choose between your fine-tuned model or a baseline pre-trained model for generating summaries"
        ) == "Baseline Model"  # Returns True if Baseline is selected
        
        # Filter for the selected stock
        filter_stock = None if selected_stock == "All Stocks" else selected_stock
        
        # Get news articles
        news_items = get_news_for_stock(
            _db_manager=_db_manager,
            stock_symbol=filter_stock,
            limit=num_articles
        )
        
        # Display news articles
        if news_items:
            st.write(f"### Showing {len(news_items)} news articles for {selected_stock}")
            
            # Display each article
            for i, item in enumerate(news_items):
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.image("https://via.placeholder.com/150x100.png?text=News", width=150)
                        st.caption(f"Source: {item['source']}")
                        st.caption(f"Date: {format_date(item['pubDate'])}")
                        
                    with col2:
                        st.subheader(item['title'])
                        
                        # Display summary
                        st.markdown("#### Summary")
                        
                        # If regenerate option is selected, create a new summary
                        if regenerate_summaries:
                            # Check if finetuned model exists when not using baseline
                            if not use_baseline_model and os.path.exists(FINETUNED_MODEL_DIR):
                                with st.spinner("Generating new summary using fine-tuned model..."):
                                    item['summary'] = generate_new_summary(FINETUNED_MODEL_DIR, item['text'], use_baseline=False)
                            # For baseline model
                            elif use_baseline_model:
                                with st.spinner("Generating new summary using baseline model..."):
                                    item['summary'] = generate_new_summary(BASELINE_MODEL, item['text'], use_baseline=True)
                        
                        if item['summary']:
                            st.write(item['summary'])
                        else:
                            st.write("No summary available.")
                        
                        # Expandable original text
                        with st.expander("View Original Article"):
                            st.write(item['text'])
                            
                            # Add link to original source if available
                            if item['link'] and item['link'] != '#':
                                st.markdown(f"[Read original article]({item['link']})")
                    
                    st.divider()
        else:
            st.info(f"No news articles found for {selected_stock}. Try selecting a different stock.")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Make sure your vector database is properly set up and contains financial news articles.")

if __name__ == "__main__":
    main()
