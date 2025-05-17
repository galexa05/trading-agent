#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Financial News Summarization Agent
---------------------------------
An automated agent that monitors financial news related to a stock portfolio,
summarizes key articles using both baseline and fine-tuned models, and
evaluates the quality of the summaries.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from rouge_score import rouge_scorer
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Add the parent directory to path to import from scripts
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.vector_db_manager import VectorDatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# LangChain imports
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import numpy as np

class SummarizationAgent:
    """
    Agent that monitors financial news and generates summaries using
    multiple models for comparison.
    """
    
    def __init__(self, 
                db_path: str = "../data/chroma_db",
                collection_name: str = "financial_articles",
                openai_api_key: Optional[str] = None,
                huggingface_api_token: Optional[str] = None,
                model_dir: str = "../models"):
        """
        Initialize the summarization agent.
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection to use
            openai_api_key: OpenAI API key (optional)
            huggingface_api_token: HuggingFace API token (optional)
            model_dir: Directory to store fine-tuned models
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_dir = model_dir
        
        # API keys for external services
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.huggingface_api_token = huggingface_api_token or os.environ.get("HUGGINGFACE_API_TOKEN")
        
        # Initialize the vector database connection
        self.vector_db = VectorDatabaseManager(
            db_path=db_path,
            collection_name=collection_name
        )
        
        # Rouge scorer for evaluation
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize models
        self._init_baseline_model()
        self._init_finetuned_model()
    
    def _init_baseline_model(self):
        """Initialize the baseline (zero-shot) summarization model."""
        try:
            if self.openai_api_key:
                # Use OpenAI as baseline if API key is available
                self.baseline_llm = OpenAI(temperature=0.3, openai_api_key=self.openai_api_key)
                logger.info("Using OpenAI as baseline model")
            else:
                # Use an open-source model from HuggingFace as fallback
                self.baseline_llm = HuggingFaceHub(
                    repo_id="facebook/bart-large-cnn", 
                    huggingfacehub_api_token=self.huggingface_api_token,
                    model_kwargs={"temperature": 0.3, "max_length": 150}
                )
                logger.info("Using HuggingFace BART model as baseline")
                
            # Create the prompt template
            self.baseline_prompt = PromptTemplate(
                input_variables=["text"],
                template="""Summarize the following financial news article in a concise, factual manner:

{text}

Summary:"""
            )
            
            # Create the summarization chain
            self.baseline_chain = LLMChain(
                llm=self.baseline_llm,
                prompt=self.baseline_prompt
            )
        except Exception as e:
            logger.error(f"Error initializing baseline model: {str(e)}")
            self.baseline_chain = None
    
    def _init_finetuned_model(self):
        """Initialize the fine-tuned summarization model if available."""
        try:
            # Use a fine-tuned model from HuggingFace
            # This could be your own fine-tuned model or a pre-trained one
            self.finetuned_llm = HuggingFaceHub(
                repo_id="facebook/bart-large-cnn",  # Replace with your fine-tuned model
                huggingfacehub_api_token=self.huggingface_api_token,
                model_kwargs={"temperature": 0.3, "max_length": 150}
            )
            
            # Create the prompt template (same as baseline for now)
            self.finetuned_prompt = PromptTemplate(
                input_variables=["text"],
                template="""Summarize the following financial news article in a concise, factual manner:

{text}

Summary:"""
            )
            
            # Create the summarization chain
            self.finetuned_chain = LLMChain(
                llm=self.finetuned_llm,
                prompt=self.finetuned_prompt
            )
            logger.info("Fine-tuned model initialized")
        except Exception as e:
            logger.warning(f"Fine-tuned model not available: {str(e)}")
            self.finetuned_chain = None
            
    def get_articles_for_portfolio(self, 
                                  tickers: List[str], 
                                  days_back: int = 7, 
                                  article_limit: int = 5) -> List[Dict]:
        """
        Retrieve articles related to the stock portfolio from the vector database.
        
        Args:
            tickers: List of stock tickers to search for
            days_back: How many days back to search
            article_limit: Max number of articles per ticker
            
        Returns:
            List of dictionaries containing article info
        """
        all_articles = []
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        for ticker in tickers:
            # Create a query to find articles about this ticker
            query = f"{ticker} financial news stock market"
            
            # Search the vector database
            results = self.vector_db.query_database(
                query=query,
                n_results=article_limit,
                metadata_filter=None,  # Can filter by date here if needed
                include_summary=True
            )
            
            # Process and filter results
            if results.get("documents"):
                for i, doc in enumerate(results["documents"]):
                    metadata = results["metadatas"][i]
                    article_info = {
                        "ticker": ticker,
                        "document": doc,
                        "metadata": metadata,
                        "similarity": 1.0 - results["distances"][0][i]
                    }
                    article_info['document'] = np.unique(article_info['document']).tolist()
                    print(article_info['document'])
                    all_articles.append(article_info)
        
        # Sort by similarity score (descending)
        all_articles.sort(key=lambda x: x["similarity"], reverse=True)
        all_articles
        return all_articles
    
    def generate_summaries(self, articles: List[Dict]) -> List[Dict]:
        """
        Generate summaries for a list of articles using both baseline and fine-tuned models.
        
        Args:
            articles: List of articles to summarize
            
        Returns:
            List of articles with summaries added
        """
        for i in tqdm(range(len(articles)), desc="Generating summaries"):
        # for i, article in enumerate(articles):
            # text = ' '.join(articles[i]["document"])
            text_list = articles[i]["document"]
            
            # Generate baseline summary
            if self.baseline_chain:
                try:
                    articles[i]["baseline_summary"] = [self.baseline_chain.invoke(text[:500]) for text in text_list]
                except Exception as e:
                    logger.error(f"Error generating baseline summary: {str(e)}")
                    articles[i]["baseline_summary"] = "Error generating summary."
            
            # Generate fine-tuned summary
            if self.finetuned_chain:
                try:
                    articles[i]["finetuned_summary"] = [self.finetuned_chain.invoke(text[:500]) for text in text_list]
                except Exception as e:
                    logger.error(f"Error generating fine-tuned summary: {str(e)}")
                    articles[i]["finetuned_summary"] = "Error generating summary."
            
        return articles
    
    def evaluate_summaries(self, articles: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate the quality of generated summaries using Rouge metrics.
        
        Args:
            articles: List of articles with generated summaries
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Initialize metrics
        metrics = {
            "baseline": {"rouge1": [], "rouge2": [], "rougeL": []},
            "finetuned": {"rouge1": [], "rouge2": [], "rougeL": []},
            "comparison": {"better": 0, "worse": 0, "same": 0}
        }
        
        for article in articles:
            # Use the article's summary (if available) as reference
            reference = article["metadata"].get("summary", "")
            
            if reference and "baseline_summary" in article and "finetuned_summary" in article:
                # Score baseline summary
                baseline_scores = self.scorer.score(reference, article["baseline_summary"])
                for metric, score in baseline_scores.items():
                    metrics["baseline"][metric].append(score.fmeasure)
                
                # Score fine-tuned summary
                finetuned_scores = self.scorer.score(reference, article["finetuned_summary"])
                for metric, score in finetuned_scores.items():
                    metrics["finetuned"][metric].append(score.fmeasure)
                
                # Compare ROUGE-L scores
                if finetuned_scores["rougeL"].fmeasure > baseline_scores["rougeL"].fmeasure:
                    metrics["comparison"]["better"] += 1
                elif finetuned_scores["rougeL"].fmeasure < baseline_scores["rougeL"].fmeasure:
                    metrics["comparison"]["worse"] += 1
                else:
                    metrics["comparison"]["same"] += 1
        
        # Calculate averages
        for model in ["baseline", "finetuned"]:
            for metric in ["rouge1", "rouge2", "rougeL"]:
                if metrics[model][metric]:
                    metrics[model][f"avg_{metric}"] = np.mean(metrics[model][metric])
                else:
                    metrics[model][f"avg_{metric}"] = 0.0
        
        return metrics
    
    def run(self, 
            portfolio: List[str], 
            days_back: int = 7, 
            article_limit: int = 5,
            output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the full pipeline: fetch articles, generate summaries, evaluate.
        
        Args:
            portfolio: List of stock tickers
            days_back: How many days back to search
            article_limit: Maximum articles per ticker
            output_file: Optional path to save results to JSON
            
        Returns:
            Dictionary with processed articles and metrics
        """
        logger.info(f"Processing portfolio: {', '.join(portfolio)}")
        
        # 1. Get relevant articles
        articles = self.get_articles_for_portfolio(
            tickers=portfolio,
            days_back=days_back,
            article_limit=article_limit
        )
        logger.info(f"Found {len(articles)} relevant articles")
        
        # 2. Generate summaries
        articles_with_summaries = self.generate_summaries(articles)
        
        # 3. Evaluate summaries
        evaluation = self.evaluate_summaries(articles_with_summaries)
        logger.info(f"Evaluation results: {json.dumps(evaluation, indent=2)}")
        
        # 4. Prepare results
        results = {
            "timestamp": datetime.now().isoformat(),
            "portfolio": portfolio,
            "articles_count": len(articles),
            "articles": articles_with_summaries,
            "evaluation": evaluation
        }
        
        # 5. Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Financial News Summarization Agent")
    parser.add_argument("--portfolio", type=str, nargs="+", default=["AAPL", "TSLA", "MSFT"],
                       help="List of stock tickers in the portfolio")
    parser.add_argument("--db-path", type=str, default="../data/chroma_db",
                       help="Path to ChromaDB database")
    parser.add_argument("--collection", type=str, default="financial_articles",
                       help="Name of the collection to use")
    parser.add_argument("--days-back", type=int, default=7,
                       help="How many days back to search for articles")
    parser.add_argument("--article-limit", type=int, default=5,
                       help="Maximum number of articles per ticker")
    parser.add_argument("--output", type=str, default="../data/summaries.json",
                       help="Path to save results")
    
    args = parser.parse_args()
    
    # Initialize the agent
    agent = SummarizationAgent(
        db_path=args.db_path,
        collection_name=args.collection
    )
    
    # Run the full pipeline
    results = agent.run(
        portfolio=args.portfolio,
        days_back=args.days_back,
        article_limit=args.article_limit,
        output_file=args.output
    )
    
    # Print summary
    print("\n======= SUMMARY RESULTS =======")
    print(f"Portfolio: {', '.join(args.portfolio)}")
    print(f"Articles processed: {len(results['articles'])}")
    print(f"ROUGE-1 Average (Baseline): {results['evaluation']['baseline'].get('avg_rouge1', 0):.4f}")
    print(f"ROUGE-1 Average (Fine-tuned): {results['evaluation']['finetuned'].get('avg_rouge1', 0):.4f}")
    print(f"Improvement: {results['evaluation']['comparison']['better']} better, {results['evaluation']['comparison']['worse']} worse, {results['evaluation']['comparison']['same']} same")
    print("===============================")


if __name__ == "__main__":
    main()
