# 1) Start from a small Linux image that already has Python.
FROM python:3.12-slim

# 2) Work inside /app in the container.
WORKDIR /app

# 3) Install the libraries. PyTorch first, CPU-only version (much smaller than the GPU one).
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# 4) Copy our code, the data and the fine-tuned model.
COPY data.py matchers.py search.py app.py ./
COPY data/deepmatcher/ data/deepmatcher/
COPY models/minilm-abtbuy/ models/minilm-abtbuy/
COPY models/hybrid_config.json models/

# 5) The API listens on port 8000.
EXPOSE 8000

# 6) Start the server. 0.0.0.0 = accept requests from outside the container.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
