#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune a Summarization Model
-------------------------------
This script prepares a dataset of financial news articles and their summaries,
then fine-tunes a pre-trained model (e.g., T5, BART) for the summarization task.
It is designed to work with limited computational resources.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from rouge_score import rouge_scorer
import torch
from tqdm import tqdm

# Add the parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.vector_db_manager import VectorDatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class SummarizerFineTuner:
    """Fine-tune a pre-trained model for financial news summarization."""
    
    def __init__(self, 
                base_model: str = "facebook/bart-large-cnn",
                db_path: str = "../data/chroma_db",
                collection_name: str = "financial_articles",
                output_dir: str = "../models/finetuned_summarizer",
                max_source_length: int = 1024,
                max_target_length: int = 128):
        """
        Initialize the fine-tuning process.
        
        Args:
            base_model: HuggingFace model ID to use as starting point
            db_path: Path to ChromaDB database
            collection_name: Name of the collection to use
            output_dir: Directory to save the fine-tuned model
            max_source_length: Maximum token length for input texts
            max_target_length: Maximum token length for summaries
        """
        self.base_model = base_model
        self.db_path = db_path
        self.collection_name = collection_name
        self.output_dir = output_dir
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize vector DB connection to get training data
        self.vector_db = VectorDatabaseManager(
            db_path=db_path,
            collection_name=collection_name
        )
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        
        # Rouge scorer for evaluation
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def prepare_dataset(self, 
                      max_articles: int = 500,
                      test_split: float = 0.1) -> DatasetDict:
        """
        Prepare a dataset of articles and summaries from the vector database.
        
        Args:
            max_articles: Maximum number of articles to include
            test_split: Fraction of data to use for testing
            
        Returns:
            DatasetDict with train and test splits
        """
        logger.info("Querying vector database for training data...")
        
        # Get all articles from the vector database
        collection = self.vector_db.collection
        results = collection.get(limit=max_articles)
        
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        
        # Filter articles that have both text and summary
        article_data = []
        for i, doc in enumerate(documents):
            metadata = metadatas[i]
            summary = metadata.get("summary", "")
            
            if doc and summary and len(summary) > 20:  # Ensure summary is meaningful
                article_data.append({
                    "text": doc,
                    "summary": summary,
                    "article_id": metadata.get("article_id", ""),
                    "source": metadata.get("source", "")
                })
        
        logger.info(f"Found {len(article_data)} articles with valid summaries")
        
        # Convert to DataFrame and create splits
        df = pd.DataFrame(article_data)
        
        # Create train/test split
        np.random.seed(42)
        indices = np.random.permutation(len(df))
        test_size = int(len(df) * test_split)
        
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        train_df = df.iloc[train_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)
        
        logger.info(f"Training set: {len(train_df)} articles, Test set: {len(test_df)} articles")
        
        # Convert to Datasets
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        return DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
    
    def preprocess_data(self, examples):
        """Tokenize inputs and targets for the model."""
        # Tokenize inputs
        inputs = self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_source_length
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            examples["summary"],
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length
        )
        
        # Replace padding token id's with -100 so they are ignored in loss computation
        targets["input_ids"] = [
            [(t if t != self.tokenizer.pad_token_id else -100) for t in target]
            for target in targets["input_ids"]
        ]
        
        # Add tokenized targets to inputs
        inputs["labels"] = targets["input_ids"]
        
        return inputs
    
    def compute_metrics(self, eval_pred):
        """Compute ROUGE metrics for evaluation."""
        predictions, labels = eval_pred
        
        # Decode predictions
        predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        # Replace -100 in the labels with pad token id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        # Decode reference summaries
        references = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        # Compute ROUGE scores
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
            rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
            rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
        
        # Average scores
        results = {
            "rouge1": np.mean(rouge_scores["rouge1"]),
            "rouge2": np.mean(rouge_scores["rouge2"]),
            "rougeL": np.mean(rouge_scores["rougeL"])
        }
        
        return results
    
    def finetune(self, 
                batch_size: int = 4,
                epochs: int = 3,
                learning_rate: float = 5e-5) -> Dict[str, Any]:
        """
        Fine-tune the summarization model.
        
        Args:
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Fine-tuning {self.base_model} for summarization...")
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        
        # Preprocess the dataset
        tokenized_dataset = dataset.map(
            self.preprocess_data,
            batched=True,
            batch_size=batch_size
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding="max_length"
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=epochs,
            predict_with_generate=True,
            generation_max_length=self.max_target_length,
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            report_to="none"  # Disable wandb/tensorboard to save resources
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Start training
        logger.info("Starting training...")
        train_results = trainer.train()
        
        # Save the model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Evaluate
        logger.info("Evaluating on test set...")
        eval_results = trainer.evaluate(
            max_length=self.max_target_length,
            num_beams=4
        )
        
        # Log results
        logger.info(f"Evaluation results: {eval_results}")
        
        # Save metrics
        metrics = {
            "train": train_results.metrics,
            "eval": eval_results
        }
        
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def run_sample_predictions(self, num_samples: int = 5) -> None:
        """
        Run predictions on a few samples to demonstrate the model.
        
        Args:
            num_samples: Number of test samples to run predictions on
        """
        logger.info("Running sample predictions...")
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        test_samples = dataset["test"].select(range(min(num_samples, len(dataset["test"]))))
        
        # Get samples
        for i, sample in enumerate(test_samples):
            text = sample["text"]
            reference = sample["summary"]
            
            # Generate prediction
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_source_length, truncation=True)
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=4,
                min_length=20,
                max_length=self.max_target_length,
                early_stopping=True
            )
            prediction = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Calculate ROUGE scores
            scores = self.scorer.score(reference, prediction)
            
            # Print results
            print(f"\n===== SAMPLE {i+1} =====")
            print(f"SOURCE: {text[:300]}...")
            print(f"\nREFERENCE: {reference}")
            print(f"\nPREDICTION: {prediction}")
            print(f"\nROUGE-1: {scores['rouge1'].fmeasure:.4f}")
            print(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
            print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Fine-tune a summarization model on financial news")
    parser.add_argument("--base-model", type=str, default="facebook/bart-large-cnn",
                       help="Base model to fine-tune")
    parser.add_argument("--db-path", type=str, default="../data/chroma_db",
                       help="Path to ChromaDB database")
    parser.add_argument("--collection", type=str, default="financial_articles",
                       help="Name of the collection to use")
    parser.add_argument("--output-dir", type=str, default="../models/finetuned_summarizer",
                       help="Directory to save fine-tuned model")
    parser.add_argument("--max-articles", type=int, default=500,
                       help="Maximum number of articles to use for training")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--sample-predictions", action="store_true",
                       help="Run sample predictions after training")
    
    args = parser.parse_args()
    
    # Check for GPU
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU found. Training will be slow.")
    
    # Initialize fine-tuner
    fine_tuner = SummarizerFineTuner(
        base_model=args.base_model,
        db_path=args.db_path,
        collection_name=args.collection,
        output_dir=args.output_dir
    )
    
    # Fine-tune the model
    metrics = fine_tuner.finetune(
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Run sample predictions if requested
    if args.sample_predictions:
        fine_tuner.run_sample_predictions()
    
    # Print final results
    print("\n======= FINE-TUNING RESULTS =======")
    print(f"Base model: {args.base_model}")
    print(f"Training epochs: {args.epochs}")
    print(f"Final ROUGE-1: {metrics['eval'].get('rouge1', 0):.4f}")
    print(f"Final ROUGE-2: {metrics['eval'].get('rouge2', 0):.4f}")
    print(f"Final ROUGE-L: {metrics['eval'].get('rougeL', 0):.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("==================================")


if __name__ == "__main__":
    main()
