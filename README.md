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
│   ├── raw_articles/           # Raw article data for vector DB initialization
│   ├── chroma_db/              # Vector database storage
│   └── summaries.json          # Generated summaries and evaluations
├── scripts/
│   ├── collect_articles.py     # Script for collecting news articles 
│   ├── finetune_summarizer.py  # Fine-tuning pipeline for summarization model
│   ├── news_dashboard.py       # Interactive web dashboard for results
│   ├── news_summarizer_app.py  # Streamlit app for viewing news and summaries
│   ├── summarization_agent.py  # Main agent for news summarization
│   ├── transform_articles.py   # Preprocessing for articles
│   ├── vector_db.py            # Vector database operations
│   └── vector_db_manager.py    # Interface for vector DB management
├── models/
│   └── bart-finetuned-2/       # Fine-tuned BART model storage
├── jupyter_files/
│   ├── create_notebook.py      # Script to create Jupyter notebooks
│   ├── model_evaluation.py     # Evaluation metrics for summarization models
│   └── vector_db_walkthrough.ipynb  # Notebook for vector DB exploration
├── Dockerfile                  # Docker configuration for containerization
├── docker-compose.yml          # Docker Compose configuration
├── entrypoint.sh              # Container entrypoint script
├── run_app.sh                 # Script to build and run the Docker container
├── Pipfile                    # Python dependencies
├── .env                       # Environment variables (API keys, etc.)
└── README.md                  # Project documentation
```

## Getting Started

You can run this application either locally or using Docker.

### Prerequisites

#### For Local Development
- Python 3.12+
- pip or pipenv for dependency management
- (Optional) GPU for faster model fine-tuning

#### For Docker Deployment
- Docker and docker-compose installed
- No other dependencies needed - everything runs in the container!

### Installation

1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/trading-agent.git
   cd trading-agent
   ```

2. Install dependencies (local development only)
   ```bash
   pipenv install
   pipenv shell
   ```

### Docker Deployment

This project can be easily deployed using Docker, which handles all dependencies, the vector database, and model requirements automatically.

1. Make sure Docker and docker-compose are installed on your system

2. Run the deployment script
   ```bash
   ./run_app.sh
   ```

3. Access the application at http://localhost:8501

The deployment script will:
- Check for required components (vector database, fine-tuned model)
- Create a template .env file if one doesn't exist
- Build and start the Docker container
- Mount volumes for persistent data storage

#### Environment Variables

Before running, you may want to set up your API keys in the `.env` file:

```bash
# Trading Agent Environment Variables
NEWS_DATA_API=your_api_key_here  # For article collection
HUGGINGFACE_TOKEN=your_huggingface_token_here  # For model fine-tuning
```

#### Docker Management Commands

- **View logs**: `docker-compose logs -f`
- **Stop the app**: `docker-compose down`
- **Restart the app**: `docker-compose restart`

### Local Usage

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

## Running the Streamlit News Summarizer App

### Docker (Recommended)

The easiest way to run the News Summarizer app is through Docker:

```bash
# One command setup and launch
./run_app.sh
```

This handles all dependencies, data management, and server setup automatically. The web interface will be available at http://localhost:8501.

### Local Development

If you prefer to run the app locally:

```bash
# First, activate your pipenv environment
pipenv shell

# Then run the streamlit app
python -m streamlit run scripts/news_summarizer_app.py
```

### Features of the News Summarizer App

- **Stock Selection**: Choose from available stocks in your portfolio
- **Multiple Models**: Generate summaries using baseline, fine-tuned, or both models
- **Side-by-Side Comparison**: Compare original vs. generated summaries
- **Interactive UI**: Modern, user-friendly interface for browsing financial articles

## License

This project is licensed under the MIT License.