FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including Chrome and dependencies for Selenium
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    gnupg \
    ca-certificates \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy Pipfile and install dependencies
COPY Pipfile Pipfile.lock* ./
RUN pipenv install --system --deploy

# No need for additional ML dependencies as they're in the Pipfile

# Create directories for data and models
RUN mkdir -p data/chroma_db data/raw_articles models/bart-finetuned-2

# Copy application code
COPY scripts/ ./scripts/
COPY .env .

# Copy only the specific vector database and model if they exist
# (these will be handled by the entrypoint script if missing)
COPY data/chroma_db/ ./data/chroma_db/
COPY models/bart-finetuned-2/ ./models/bart-finetuned-2/

# Set the entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose Streamlit port
EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
