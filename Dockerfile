FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords rslp punkt punkt_tab

RUN mkdir -p /app/modelos

RUN wget -q -O /app/modelos/modelos.zip "https://github.com/oliveirarennan/classificador_projetos_pesquisa/releases/download/v1.0-modelos/modelos.zip"

RUN unzip /app/modelos/modelos.zip -d /app/ && rm /app/modelos/modelos.zip

COPY . .

EXPOSE 8025

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8025", "--timeout-keep-alive", "300"]