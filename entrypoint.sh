#!/bin/bash
set -e

# Function to initialize the vector database
initialize_vector_db() {
    echo "Initializing vector database..."
    
    # Check if we have raw article data
    if [ -d "/app/data/raw_articles" ] && [ "$(ls -A /app/data/raw_articles 2>/dev/null)" ]; then
        echo "Found raw article data, loading into vector database..."
        python -m scripts.vector_db_manager --initialize --db_path /app/data/chroma_db --collection_name financial_articles
        return $?
    else
        echo "No raw articles found. Attempting to collect articles..."
        # Try to collect articles using the collector script if configured
        if [ -n "$NEWS_DATA_API" ]; then
            echo "NEWS_DATA_API found, collecting articles..."
            python -m scripts.collect_articles --query "AAPL MSFT GOOG AMZN META" --max_iterations 5 --output_dir /app/data/raw_articles
            
            if [ $? -eq 0 ]; then
                echo "Articles collected successfully. Loading into vector database..."
                python -m scripts.vector_db_manager --initialize --db_path /app/data/chroma_db --collection_name financial_articles
                return $?
            fi
        fi
    fi
    
    echo "WARNING: Could not initialize vector database. The app will start with an empty database."
    return 1
}

# Function to fine-tune the model if needed
finetune_model() {
    echo "Starting model fine-tuning..."
    
    # Check if we have enough articles in the vector database
    # We need to query the database to check article count
    ARTICLE_COUNT=$(python -c 'from scripts.vector_db_manager import VectorDatabaseManager; db = VectorDatabaseManager(db_path="/app/data/chroma_db", collection_name="financial_articles"); print(len(db.get_all_articles()))')
    
    if [ "$ARTICLE_COUNT" -gt 50 ]; then
        echo "Found $ARTICLE_COUNT articles in database, sufficient for fine-tuning."
        python -m scripts.finetune_summarizer --base_model facebook/bart-large-cnn --db_path /app/data/chroma_db --collection_name financial_articles --output_dir /app/models/bart-finetuned-2 --epochs 1 --batch_size 2
        return $?
    else
        echo "Insufficient articles ($ARTICLE_COUNT) for fine-tuning. Need at least 50 articles."
        return 1
    fi
}

# Check if vector database exists and has content
echo "Checking vector database..."
if [ -z "$(ls -A /app/data/chroma_db 2>/dev/null)" ]; then
    echo "Vector database not found or empty."
    mkdir -p /app/data/chroma_db
    initialize_vector_db
    DB_INIT_RESULT=$?
    
    if [ $DB_INIT_RESULT -ne 0 ]; then
        echo "Note: The app will start with an empty database. You can initialize it from the app interface."
    fi
else
    echo "Vector database found."
 fi

# Check if fine-tuned model exists
echo "Checking fine-tuned model..."
if [ -z "$(ls -A /app/models/bart-finetuned-2 2>/dev/null)" ]; then
    echo "Fine-tuned model not found."
    mkdir -p /app/models/bart-finetuned-2
    
    # Check if we have vector database with articles for fine-tuning
    if [ -d "/app/data/chroma_db" ] && [ "$(ls -A /app/data/chroma_db 2>/dev/null)" ]; then
        echo "Vector database exists, attempting to fine-tune model..."
        finetune_model
        FT_RESULT=$?
        
        if [ $FT_RESULT -ne 0 ]; then
            echo "Fine-tuning failed or insufficient data. The app will use the baseline model (facebook/bart-large-cnn)."
        else
            echo "Fine-tuning completed successfully!"
        fi
    else
        echo "Vector database is empty. Cannot fine-tune model. The app will use the baseline model (facebook/bart-large-cnn)."
    fi
else
    echo "Fine-tuned model found."
fi

# Start the Streamlit application
echo "Starting Financial News Summarizer app..."
exec streamlit run scripts/news_summarizer_app.py --server.port=8501 --server.address=0.0.0.0
