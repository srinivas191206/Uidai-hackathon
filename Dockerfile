NameError: name 'capacity_adj_pct' is not defined
Traceback:
File "/home/user/.local/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 542, in _run_script
exec(code, module.__dict__)
File "/home/user/app/app.py", line 945, in <module>
if capacity_adj_pct > 0 or extra_hours > 0:FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
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
