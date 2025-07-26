FROM python:3.11-slim
RUN apt-get update && apt-get install -y gcc libfontconfig1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY web_server.py ./
COPY run.sh ./
RUN chmod +x run.sh && mkdir -p input output models
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1
EXPOSE 5000
CMD ["python", "web_server.py"]
