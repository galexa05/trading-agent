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
# FINETUNED_MODEL_DIR = os.path.join(project_dir, "models", "finetuned_summarizer")
FINETUNED_MODEL_DIR = os.path.join(project_dir, "models", "bart-finetuned-2")


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
    # Query with filter for the stock symbol, requesting more results to ensure we get enough unique articles
    fetch_limit = limit * 3  # Request more results to ensure we have enough after deduplication
    results = _db_manager.query_database(
        query=f"news about {stock_symbol}",
        n_results=fetch_limit,
        # metadata_filter={"stock_symbol": stock_symbol} if stock_symbol else None,
        include_summary=True
    )
    
    news_items = []
    seen_articles = set()  # Track unique articles using article_id or link
    
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
            if len(news_items) >= limit:  # Stop once we have enough unique articles
                break
                
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Use article_id or link as unique identifier
            article_id = metadata.get('article_id', None)
            link = metadata.get('link', None)
            article_key = article_id if article_id else link
            
            # Skip if we've already seen this article
            if not article_key or article_key in seen_articles:
                continue
                
            seen_articles.add(article_key)
            
            # Create news item with available metadata
            news_item = {
                'text': doc,
                'title': metadata.get('title', 'No Title Available'),
                'summary': metadata.get('summary', ''),
                'source': metadata.get('source', 'Unknown Source'),
                'pubDate': metadata.get('pubDate', ''),
                'link': link if link else '#',
                'stock_symbol': metadata.get('stock_symbol', ''),
                'article_id': article_id
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

def generate_new_summary(model_path, article_text, approach="fine-tuned"):
    """
    Generate a new summary for an article using different approaches:
    - fine-tuned: Use the domain-adapted fine-tuned model
    - zero-shot: Use the baseline model with no examples
    - few-shot: Use the baseline model with relevant examples
    
    Args:
        model_path: Path to the fine-tuned model or name of baseline model
        article_text: Text to summarize
        approach: Which approach to use ("fine-tuned", "zero-shot", or "few-shot")
        
    Returns:
        Generated summary
    """
    try:
        # Truncate article text if it's too long to prevent issues
        max_text_length = 4000
        if len(article_text) > max_text_length:
            article_text = article_text[:max_text_length]
            st.info(f"Article text was truncated to {max_text_length} characters for summarization.")
        
        # Import required libraries
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        import torch
        
        # Different handling based on approach
        if approach == "zero-shot" or approach == "few-shot":
            # For baseline models, use the summarization pipeline which is more robust
            try:
                summarizer = pipeline(
                    "summarization", 
                    model=model_path,
                    device=-1  # CPU
                )
                
                if approach == "zero-shot":
                    # Standard zero-shot approach
                    summary_output = summarizer(
                        article_text,
                        max_length=150,
                        min_length=40,
                        do_sample=False
                    )
                    
                    return summary_output[0]['summary_text']
                
                elif approach == "few-shot":
                    # Few-shot approach with examples
                    few_shot_prompt = """
                    Here are some examples of good financial news summaries:
                    
                    Original: Apple Inc. reported record-breaking quarterly revenue of $91.8 billion, an increase of 9% from the same quarter a year ago. The company's services and wearables divisions showed strong growth, offsetting a slight decline in iPhone sales. CEO Tim Cook attributed the success to high customer satisfaction and loyalty.
                    Summary: Apple reported record quarterly revenue of $91.8B, up 9% year-over-year, with strong growth in services and wearables compensating for lower iPhone sales.
                    
                    Original: Tesla's Q4 earnings beat Wall Street expectations with revenue reaching $24.3 billion. The electric vehicle maker delivered a record 405,278 vehicles in the quarter despite production challenges and increasing competition. The company announced plans to expand its manufacturing capacity and introduce new models in the coming year.
                    Summary: Tesla exceeded Q4 expectations with $24.3B revenue and record 405,278 vehicle deliveries, while announcing manufacturing expansion and new model plans.
                    
                    Now summarize this financial news article:
                    {article}
                    """
                    
                    # Format the few-shot prompt with the article text
                    few_shot_input = few_shot_prompt.format(article=article_text)
                    
                    # Truncate if too long
                    if len(few_shot_input) > max_text_length:
                        st.warning("Few-shot prompt was too long and had to be truncated. This might affect summary quality.")
                        few_shot_input = few_shot_input[-max_text_length:]
                    
                    # Generate summary using few-shot approach
                    summary_output = summarizer(
                        few_shot_input,
                        max_length=150,
                        min_length=40,
                        do_sample=False
                    )
                    
                    return summary_output[0]['summary_text']
            except Exception as baseline_error:
                st.error(f"Error with baseline model: {str(baseline_error)}")
                # Fallback to manual model loading if pipeline fails
        
        # Manual model loading (for fine-tuned model or as fallback)
        try:
            # Load tokenizer first - this should work for both model types
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load the model without early_stopping in the constructor
            # early_stopping should only be in generate() method
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path,early_stopping=True)
            
            # Tokenize the text with proper padding
            inputs = tokenizer(article_text, return_tensors="pt", max_length=1024, 
                              truncation=True, padding="max_length")
            
            # Generate summary with properly configured parameters
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                min_length=40,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            # Decode the summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return summary
        except Exception as model_error:
            st.error(f"Error loading or running model: {str(model_error)}")
            return "Unable to generate summary using the model."
            
    except Exception as e:
        st.error(f"Unexpected error generating summary: {str(e)}")
        return "Unable to generate summary due to an unexpected error."

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
        model_selection = st.sidebar.radio(
            "Summarization Model",
            options=["Fine-tuned Model", "Zero-shot Baseline", "Few-shot Baseline", "Compare All Models"],
            index=0,
            help="Choose which model(s) to use for generating summaries. Zero-shot uses the model directly, few-shot provides examples to the model first."
        )
        
        # Determine which models to use
        use_zero_shot = model_selection == "Zero-shot Baseline"
        use_few_shot = model_selection == "Few-shot Baseline"
        use_all_models = model_selection == "Compare All Models"
        
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
                        
                        # Display original summary in a styled container
                        st.markdown(
                            """<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
                            <h4 style='color: #1E3A8A; margin-bottom: 10px;'>Original Summary</h4>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                        
                        original_summary = item['summary']
                        if original_summary:
                            st.markdown(f"""<div style='background-color: #f8f9fa; padding: 15px; border-left: 4px solid #1E3A8A; 
                                        border-radius: 5px; margin-bottom: 20px; font-style: italic; color: black;'>
                                        {original_summary}</div>""", unsafe_allow_html=True)
                        else:
                            st.markdown("""<div style='background-color: #f8f9fa; padding: 15px; border-left: 4px solid #DC2626; 
                                        border-radius: 5px; margin-bottom: 20px; color: #6B7280;'>
                                        No original summary available.</div>""", unsafe_allow_html=True)
                        
                        # Generated summary section(s)
                        if regenerate_summaries or not any([item.get('summary'), item.get('zero_shot_summary'), item.get('few_shot_summary')]):
                            # Generate summaries based on selected model or all models if in compare mode
                            if use_all_models or (not use_zero_shot and not use_few_shot):
                                # Use fine-tuned model if not explicitly using baseline models
                                try:
                                    with st.spinner("Generating fine-tuned model summary..."):
                                        summary = generate_new_summary(
                                            model_path=FINETUNED_MODEL_DIR,
                                            article_text=item['text'],
                                            approach="fine-tuned"
                                        )
                                        item['summary'] = summary
                                except Exception as e:
                                    st.error(f"Error generating fine-tuned summary: {str(e)}")
                                    item['summary'] = "Error generating summary with fine-tuned model."
                            
                            if use_all_models or use_zero_shot:
                                # Generate zero-shot summary
                                try:
                                    with st.spinner("Generating zero-shot baseline summary..."):
                                        zero_shot_summary = generate_new_summary(
                                            model_path=BASELINE_MODEL,
                                            article_text=item['text'],
                                            approach="zero-shot"
                                        )
                                        item['zero_shot_summary'] = zero_shot_summary
                                except Exception as e:
                                    st.error(f"Error generating zero-shot summary: {str(e)}")
                                    item['zero_shot_summary'] = "Error generating summary with zero-shot approach."
                            
                            if use_all_models or use_few_shot:
                                # Generate few-shot summary
                                try:
                                    with st.spinner("Generating few-shot baseline summary..."):
                                        few_shot_summary = generate_new_summary(
                                            model_path=BASELINE_MODEL,
                                            article_text=item['text'],
                                            approach="few-shot"
                                        )
                                        item['few_shot_summary'] = few_shot_summary
                                except Exception as e:
                                    st.error(f"Error generating few-shot summary: {str(e)}")
                                    item['few_shot_summary'] = "Error generating summary with few-shot approach."
                        
                        # Display generated summaries
                        if item.get('summary'):
                            with st.expander("🤖 Generated Summary (Fine-tuned Model)", expanded=True):
                                st.markdown(f"<div style='background-color: #f0fff0; padding: 10px; border-radius: 5px; color: black;'>{item['summary']}</div>", unsafe_allow_html=True)
                        
                        if item.get('zero_shot_summary'):
                            with st.expander("🤖 Generated Summary (Zero-shot Baseline)", expanded=True):
                                st.markdown(f"<div style='background-color: #fff8f0; padding: 10px; border-radius: 5px; color: black;'>{item['zero_shot_summary']}</div>", unsafe_allow_html=True)
                        
                        if item.get('few_shot_summary'):
                            with st.expander("🤖 Generated Summary (Few-shot Baseline)", expanded=True):
                                st.markdown(f"<div style='background-color: #f0f0ff; padding: 10px; border-radius: 5px; color: black;'>{item['few_shot_summary']}</div>", unsafe_allow_html=True)
                        
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
