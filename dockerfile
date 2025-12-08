
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/rag_engine.py backend/main.py backend/models.py frontend/app.py  .



EXPOSE 8080, 8501

CMD ["python", "main.py"]

