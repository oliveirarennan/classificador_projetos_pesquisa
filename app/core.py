# app/core.py
import os
import re
import string
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from transformers import BertForSequenceClassification, BertTokenizer
from unidecode import unidecode

MAPA_CLASSES = {
    0: "Artes Visuais",
    1: "Artes da Cena",
    2: "Cinema",
    3: "Ensino Aprendizagem em Arte",
    4: "Poéticas Tecnológicas",
    5: "Preservação do Patrimônio Cultural",
}

# ==============================================================================
# 1. FUNÇÕES DE PRÉ-PROCESSAMENTO (COPIE DO SEU NOTEBOOK)
# ==============================================================================


try:
    stop_words_nltk = set(stopwords.words("portuguese"))
    stemmer_nltk = RSLPStemmer()
    print("Recursos de NLP (NLTK) carregados.")
except Exception as e:
    print(f"Erro ao carregar recursos de NLP: {e}")


def preprocess_texto_classico(texto):
    """
    Versão simplificada para produção.
    Aplica as regras fixas do seu melhor modelo RF:
    - Remover Acentos: Sim
    - Remover Stopwords: Sim
    - Stemming: Sim
    """
    if not isinstance(texto, str):
        return ""

    # Lógica copiada do seu notebook
    texto = texto.lower()
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)
    texto = re.sub(r"\S+@\S+", "", texto)
    texto = re.sub(r"\d+", "", texto)
    texto = unidecode(texto)  # Remove acentos
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto).strip()

    tokens = word_tokenize(texto, language="portuguese")

    # Stopwords + Stemming (Regra do RF)
    tokens_filtrados = [p for p in tokens if p not in stop_words_nltk and len(p) > 1]
    tokens_finais = [stemmer_nltk.stem(p) for p in tokens_filtrados]

    return " ".join(tokens_finais)


# --- Função de Pré-processamento para BERT ---
def preprocess_texto_bert(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)
    texto = re.sub(r"\S+@\S+", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# ==============================================================================
# 2. CARREGAMENTO DOS MODELOS
# ==============================================================================

model_rf_pipeline = None
model_bert = None
tokenizer_bert = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))

RAIZ_PROJETO = os.path.dirname(DIRETORIO_ATUAL)

PASTA_MODELOS = os.path.join(RAIZ_PROJETO, "modelos")


# --- 2. Função para Carregar Modelos (Chamada ao iniciar) ---
def carregar_modelos():
    global model_rf_pipeline, model_bert, tokenizer_bert

    print(f"--- Iniciando carregamento ---")
    print(f"📂 Procurando modelos na pasta: {PASTA_MODELOS}")

    # --- 1. Carrega Random Forest ---
    path_rf = os.path.join(PASTA_MODELOS, "rf_pipeline.joblib")

    if os.path.exists(path_rf):
        try:
            model_rf_pipeline = joblib.load(path_rf)
            print(f"✅ RF Pipeline carregado!")
        except Exception as e:
            print(f"❌ ERRO ao ler o arquivo RF: {e}")
    else:
        print(f"❌ ARQUIVO NÃO ENCONTRADO: {path_rf}")
        # Lista o que tem na pasta para ajudar a debugar
        if os.path.exists(PASTA_MODELOS):
            print(f"   Conteúdo da pasta modelos: {os.listdir(PASTA_MODELOS)}")

    # --- 2. Carrega BERT ---
    path_bert = os.path.join(PASTA_MODELOS, "bert_finetuned")

    if os.path.exists(path_bert):
        try:
            tokenizer_bert = BertTokenizer.from_pretrained(path_bert)
            model_bert = BertForSequenceClassification.from_pretrained(path_bert)
            model_bert.to(device)
            model_bert.eval()
            print(f"✅ BERT carregado!")
        except Exception as e:
            print(f"❌ ERRO ao ler o BERT: {e}")
    else:
        print(f"❌ PASTA BERT NÃO ENCONTRADA: {path_bert}")


# ==============================================================================
# 3. FUNÇÕES DE PREDIÇÃO (USADAS PELO MAIN.PY)
# ==============================================================================


def prever_rf(texto_original):
    inicio = time.time()  #  Inicia o cronômetro

    texto_limpo = preprocess_texto_classico(texto_original)

    if model_rf_pipeline:
        probas = model_rf_pipeline.predict_proba([texto_limpo])[0]
        resultado = {
            MAPA_CLASSES[i]: float(round(p * 100, 2)) for i, p in enumerate(probas)
        }
    else:
        resultado = {}

    fim = time.time()  #  Para o cronômetro
    tempo_total = round(fim - inicio, 4)  # Calcula com 4 casas decimais

    return resultado, tempo_total


def prever_bert(texto_original):
    inicio = time.time()  # ⏱ Inicia o cronômetro

    texto_limpo = preprocess_texto_bert(texto_original)

    if model_bert:
        inputs = tokenizer_bert(
            texto_limpo,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            outputs = model_bert(**inputs)

        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        resultado = {
            MAPA_CLASSES[i]: float(round(p * 100, 2)) for i, p in enumerate(probs)
        }
    else:
        resultado = {}

    fim = time.time()
    tempo_total = round(fim - inicio, 4)

    return resultado, tempo_total


# ==============================================================================
# 4. WRAPPER PARA O LIME (O SEGREDO DAS PALAVRAS ORIGINAIS)
# ==============================================================================


def wrapper_lime_rf(lista_textos_originais):
    """
    O LIME manda uma lista de variações do texto ORIGINAL.
    Nós aplicamos o stemming AGORA, antes de enviar pro modelo.
    Assim o LIME vê palavras reais, mas o modelo vê radicais.
    """
    # 1. Cria uma lista nova aplicando stemming em cada item
    lista_processada = [preprocess_texto_classico(t) for t in lista_textos_originais]

    # 2. Passa a lista processada pro Pipeline (Vectorizer -> RF)
    return model_rf_pipeline.predict_proba(lista_processada)


def wrapper_lime_bert(lista_textos):
    """
    Wrapper específico para o LIME funcionar com o BERT.
    O LIME envia uma lista de strings e espera uma matriz de probabilidades numpy.
    """
    # 1. Limpa o texto (mesma limpeza da predição)
    lista_limpa = [preprocess_texto_bert(t) for t in lista_textos]

    # 2. Tokeniza em lote (batch)
    inputs = tokenizer_bert(
        lista_limpa, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    # 3. Predição (sem calcular gradientes para economizar memória)
    with torch.no_grad():
        outputs = model_bert(**inputs)

    # 4. Softmax para converter logits em probabilidades (0 a 1)
    probs = F.softmax(outputs.logits, dim=1)

    # 5. Retorna como array numpy (exigência do LIME)
    return probs.cpu().numpy()


# ==============================================================================
# 4. Carregar Projetos de Testes
# ==============================================================================


def carregar_exemplos_demo():
    """
    Lê o CSV de demonstração e retorna uma lista de dicionários para o HTML.
    """
    caminho_csv = os.path.join(RAIZ_PROJETO, "modelos", "exemplos.csv")

    if not os.path.exists(caminho_csv):
        return []

    try:
        df = pd.read_csv(caminho_csv)
        exemplos = []

        for _, row in df.iterrows():
            classe = row["linha_de_pesquisa"]
            texto_completo = row["texto_completo"]

            # Pega a primeira linha ou os primeiros 100 caracteres para exibir no menu
            resumo = texto_completo.split(".")[0][:80]
            label_visual = f"[{classe}] {resumo}..."

            exemplos.append(
                {
                    "label": label_visual,  # O que o usuário vê
                    "valor": texto_completo,  # O que vai para a caixa de texto
                }
            )

        return exemplos
    except Exception as e:
        print(f"Erro ao ler CSV de demo: {e}")
        return []
