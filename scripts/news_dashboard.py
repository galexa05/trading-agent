#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Financial News Summarization Dashboard
--------------------------------------
A simple web-based dashboard that displays the summarized financial news
for a stock portfolio, comparing baseline and fine-tuned model summaries.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional

# Add the parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.summarization_agent import SummarizationAgent


class NewsDashboard:
    """Dashboard for displaying financial news summaries."""
    
    def __init__(self, 
                results_file: Optional[str] = None,
                db_path: str = "../data/chroma_db",
                collection_name: str = "financial_articles"):
        """
        Initialize the dashboard.
        
        Args:
            results_file: Path to a saved results JSON file
            db_path: Path to ChromaDB database
            collection_name: Name of the collection to use
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Load agent to run new analyses if needed
        self.agent = SummarizationAgent(
            db_path=db_path,
            collection_name=collection_name
        )
        
        # Load existing results if file provided
        self.results = None
        if results_file and os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    self.results = json.load(f)
                print(f"Loaded results from {results_file}")
            except Exception as e:
                print(f"Error loading results: {str(e)}")
    
    def run_analysis(self, 
                    tickers: str,
                    days_back: int = 7,
                    article_limit: int = 5) -> Dict[str, Any]:
        """
        Run a new analysis using the summarization agent.
        
        Args:
            tickers: Comma-separated list of stock tickers
            days_back: How many days back to search
            article_limit: Maximum articles per ticker
            
        Returns:
            Results from the summarization agent
        """
        portfolio = [ticker.strip().upper() for ticker in tickers.split(",")]
        
        self.results = self.agent.run(
            portfolio=portfolio,
            days_back=days_back,
            article_limit=article_limit
        )
        
        return self.results
    
    def display_summary_comparison(self, article_index: int) -> List[str]:
        """
        Display the original article, baseline summary, and fine-tuned summary.
        
        Args:
            article_index: Index of the article to display
            
        Returns:
            List of original text, baseline summary, and fine-tuned summary
        """
        if not self.results or not self.results.get("articles"):
            return ["No results available", "", ""]
        
        articles = self.results["articles"]
        if article_index >= len(articles):
            return ["Invalid article index", "", ""]
        
        article = articles[article_index]
        
        original_text = article["document"]
        baseline_summary = article.get("baseline_summary", "No baseline summary available")
        finetuned_summary = article.get("finetuned_summary", "No fine-tuned summary available")
        
        metadata = article["metadata"]
        title = metadata.get("title", "No title")
        source = metadata.get("source", "Unknown source")
        date = metadata.get("pubDate", "Unknown date")
        
        header = f"## {title}\n**Source:** {source} | **Date:** {date}\n\n"
        
        return [
            header + original_text,
            baseline_summary,
            finetuned_summary
        ]
    
    def generate_metrics_plot(self) -> Optional[plt.Figure]:
        """
        Generate a plot comparing ROUGE metrics for baseline and fine-tuned models.
        
        Returns:
            Matplotlib figure with metrics comparison
        """
        if not self.results or not self.results.get("evaluation"):
            return None
        
        evaluation = self.results["evaluation"]
        
        # Create DataFrame for plotting
        metrics = ["avg_rouge1", "avg_rouge2", "avg_rougeL"]
        baseline_values = [evaluation["baseline"].get(m, 0) for m in metrics]
        finetuned_values = [evaluation["finetuned"].get(m, 0) for m in metrics]
        
        df = pd.DataFrame({
            "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
            "Baseline": baseline_values,
            "Fine-tuned": finetuned_values
        })
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        df_melted = pd.melt(df, id_vars=["Metric"], var_name="Model", value_name="Score")
        
        sns.barplot(x="Metric", y="Score", hue="Model", data=df_melted, ax=ax)
        ax.set_title("ROUGE Metrics Comparison")
        ax.set_ylim(0, 1.0)
        
        return fig
    
    def article_selector_text(self) -> str:
        """Generate text showing available articles for selection."""
        if not self.results or not self.results.get("articles"):
            return "No articles available"
        
        articles = self.results["articles"]
        text = "### Available Articles\n\n"
        
        for i, article in enumerate(articles):
            metadata = article["metadata"]
            title = metadata.get("title", "No title")
            source = metadata.get("source", "Unknown source")
            text += f"{i}: {title} ({source})\n"
        
        return text
    
    def create_dashboard(self) -> gr.Blocks:
        """
        Create the Gradio dashboard interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Financial News Summarization Dashboard") as dashboard:
            gr.Markdown("# Financial News Summarization Dashboard")
            gr.Markdown("This dashboard shows summarized financial news for a stock portfolio, comparing baseline and fine-tuned models.")
            
            with gr.Tab("Run Analysis"):
                with gr.Row():
                    with gr.Column():
                        tickers_input = gr.Textbox(label="Portfolio (comma-separated tickers)", value="AAPL, TSLA, MSFT")
                        days_back = gr.Slider(minimum=1, maximum=30, value=7, step=1, label="Days to look back")
                        article_limit = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Articles per ticker")
                        run_btn = gr.Button("Run Analysis")
                    
                    with gr.Column():
                        results_output = gr.JSON(label="Analysis Results")
            
            with gr.Tab("View Summaries"):
                with gr.Row():
                    available_articles = gr.Markdown(self.article_selector_text())
                
                with gr.Row():
                    article_index = gr.Number(value=0, label="Select Article Index", precision=0)
                    view_btn = gr.Button("View Article")
                
                with gr.Row():
                    original_text = gr.Markdown(label="Original Article")
                    baseline_summary = gr.Markdown(label="Baseline Summary")
                    finetuned_summary = gr.Markdown(label="Fine-tuned Summary")
            
            with gr.Tab("Metrics"):
                with gr.Row():
                    metrics_plot = gr.Plot(label="ROUGE Metrics Comparison")
                    refresh_metrics_btn = gr.Button("Refresh Metrics")
            
            # Define interactions
            run_btn.click(
                fn=self.run_analysis,
                inputs=[tickers_input, days_back, article_limit],
                outputs=[results_output]
            ).then(
                fn=self.article_selector_text,
                inputs=[],
                outputs=[available_articles]
            ).then(
                fn=self.generate_metrics_plot,
                inputs=[],
                outputs=[metrics_plot]
            )
            
            view_btn.click(
                fn=self.display_summary_comparison,
                inputs=[article_index],
                outputs=[original_text, baseline_summary, finetuned_summary]
            )
            
            refresh_metrics_btn.click(
                fn=self.generate_metrics_plot,
                inputs=[],
                outputs=[metrics_plot]
            )
        
        return dashboard
    
    def launch(self, share: bool = False):
        """Launch the dashboard."""
        dashboard = self.create_dashboard()
        dashboard.launch(share=share)


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Financial News Summarization Dashboard")
    parser.add_argument("--results", type=str, default=None,
                       help="Path to saved results JSON file")
    parser.add_argument("--db-path", type=str, default="../data/chroma_db",
                       help="Path to ChromaDB database")
    parser.add_argument("--collection", type=str, default="financial_articles",
                       help="Name of the collection to use")
    parser.add_argument("--share", action="store_true",
                       help="Generate a public URL for the dashboard")
    
    args = parser.parse_args()
    
    dashboard = NewsDashboard(
        results_file=args.results,
        db_path=args.db_path,
        collection_name=args.collection
    )
    
    dashboard.launch(share=args.share)


if __name__ == "__main__":
    main()
