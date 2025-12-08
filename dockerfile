FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend directory
COPY backend/ ./backend/
RUN touch backend/__init__.py

# Copy frontend directory
COPY frontend/ ./frontend/

# Copy config and startup script
COPY config.py .
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000 8501

CMD ["sh", "start.sh"]
