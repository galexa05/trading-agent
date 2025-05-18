#!/bin/bash
set -e

echo "==== Financial News Summarizer App Launcher ===="
echo "This script will build and run the Trading Agent application with Docker"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Create necessary directories if they don't exist
mkdir -p data/chroma_db
mkdir -p data/raw_articles
mkdir -p models/bart-finetuned-2

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Creating a template .env file."
    echo "# Trading Agent Environment Variables" > .env
    echo "NEWS_DATA_API=your_api_key_here" >> .env
    echo "HUGGINGFACE_TOKEN=your_huggingface_token_here" >> .env
    echo "Please edit the .env file with your actual API keys before continuing."
    echo "Press Enter to continue or Ctrl+C to exit and edit the .env file..."
    read
fi

# Check if vector database exists
if [ -z "$(ls -A data/chroma_db 2>/dev/null)" ]; then
    echo "Note: Vector database directory is empty."
    echo "The app will attempt to initialize the database on startup using:"
    echo "1. Existing article data (if available in data/raw_articles)"
    echo "2. Collecting new articles (if NEWS_DATA_API is configured in .env)"
    echo "3. Otherwise, you'll need to load data manually through the app interface"
fi

# Check if fine-tuned model exists
if [ -z "$(ls -A models/bart-finetuned-2 2>/dev/null)" ]; then
    echo "Note: Fine-tuned model directory is empty."
    echo "The app will:"
    echo "1. Attempt to fine-tune a model if enough articles are available in the vector database"
    echo "2. Otherwise, use the baseline model (facebook/bart-large-cnn)"
fi

# Build and start the application
echo "Building Docker container..."
docker-compose build

echo "Starting the application..."
docker-compose up -d

echo "==== App is now running! ===="
echo "Access the application at: http://localhost:8501"
echo
echo "Logs can be viewed with: docker-compose logs -f"
echo "To stop the app, run: docker-compose down"

