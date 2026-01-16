FROM python:3.9-slim

# Install system dependencies (needed for some python packages like pandas/numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --upgrade -r requirements.txt

# Copy application files
COPY --chown=user . .

# Hugging Face Spaces port
EXPOSE 7860

# Launch application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
