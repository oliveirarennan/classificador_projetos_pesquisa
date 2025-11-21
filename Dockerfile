# Usa uma imagem leve do Python 3.10
FROM python:3.11-slim

# Define variáveis de ambiente para evitar arquivos .pyc e logs em buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema necessárias para algumas libs de ML
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo de requisitos primeiro (para aproveitar o cache do Docker)
COPY requirements.txt .

# Instala as dependências do Python
# O --no-cache-dir ajuda a manter a imagem menor
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Opcional: Pré-baixa os dados do NLTK na construção da imagem para agilizar o boot
RUN python -m nltk.downloader stopwords rslp punkt punkt_tab

# Copia todo o restante do código para dentro do container
COPY . .

# Expõe a porta 8025
EXPOSE 8025

# Comando para iniciar a aplicação
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8025"]