# Financial News Monitoring & Summarization Agent

An automated agent that monitors financial news related to a stock portfolio, summarizes key articles using both baseline and fine-tuned models, and presents concise, high-quality summaries. This project uses a vector database to store and retrieve financial news articles, and LangChain for agent orchestration and summarization tasks.

## Features

- **News Monitoring**: Retrieves financial news related to a user-defined stock portfolio
- **Vector Database**: Stores articles with embeddings for similarity search
- **Summarization Pipeline**: Generates concise summaries using:
  - Baseline model (zero-shot/few-shot)
  - Fine-tuned model for improved performance
- **Evaluation Framework**: Compares summarization quality using ROUGE metrics
- **Interactive Dashboard**: Optional web interface to view and compare summaries

## Project Structure

```
trading-agent/
├── data/
│   ├── articles.csv            # Collected financial news articles
│   ├── chroma_db/              # Vector database storage
│   └── summaries.json          # Generated summaries and evaluations
├── scripts/
│   ├── vector_db.py            # Vector database operations
│   ├── vector_db_manager.py    # Interface for vector DB management
│   ├── summarization_agent.py  # Main agent for news summarization
│   ├── finetune_summarizer.py  # Fine-tuning pipeline for summarization model
│   └── news_dashboard.py       # Interactive web dashboard for results
├── models/
│   └── finetuned_summarizer/   # Fine-tuned model storage
├── jupyter_files/
│   └── vector_db_walkthrough.ipynb  # Notebook for vector DB exploration
├── Pipfile                     # Dependencies
└── README.md                   # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip or pipenv for dependency management
- (Optional) GPU for faster model fine-tuning

### Installation

1. Clone this repository
   ```
   git clone https://github.com/yourusername/trading-agent.git
   cd trading-agent
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
   
   Or using pipenv:
   ```
   pipenv install
   pipenv shell
   ```

### Usage

#### 1. Initialize Vector Database

```bash
python -m scripts.vector_db_manager init --db-path ./data/chroma_db --collection-name financial_articles
```

#### 2. Load Articles into Vector Database

```bash
python -m scripts.vector_db_manager load --articles-path ./data/articles.csv
```

#### 3. Fine-tune Summarization Model (Optional)

```bash
python -m scripts.finetune_summarizer --base-model facebook/bart-large-cnn --epochs 3
```

#### 4. Run Summarization Agent

```bash
python -m scripts.summarization_agent --portfolio AAPL,TSLA,MSFT --days-back 7 --output ./data/summaries.json
```

#### 5. Launch Dashboard (Optional)

```bash
python -m scripts.news_dashboard --results ./data/summaries.json
```

## Evaluation

The project evaluates summarization quality using:

- **ROUGE Metrics**: Quantitative evaluation of summary quality
- **Factual Accuracy**: Manual assessment of hallucinations and factual errors
- **Comparison**: Side-by-side comparison of baseline vs. fine-tuned models

## Implementation Details

### Vector Database (ChromaDB)

- Uses sentence-transformers for embedding generation
- Persistent storage with duckdb+parquet backend
- Text chunking for long articles

### Summarization Models

- **Baseline**: Zero-shot summarization with pre-trained models
- **Fine-tuned**: Domain-adapted models for financial news summarization
- Integration with LangChain for flexible model switching

### LangChain Integration

- Modular LLMChain design for summarization
- Prompt templating for consistent outputs
- Support for multiple LLM backends (OpenAI, HuggingFace, etc.)

## License

This project is licensed under the MIT License.