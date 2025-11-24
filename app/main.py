# app/main.py
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import core

# --- 1. CONFIGURAÇÃO DE SEGURANÇA ---
security = HTTPBasic()


def verificar_credenciais(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Função que intercepta a requisição e verifica usuário/senha.
    Pega os valores das variáveis de ambiente (definidas no Docker).
    """
    # Defina aqui um padrão caso a variável de ambiente não exista
    usuario_correto = os.getenv("APP_USER", "admin")
    senha_correta = os.getenv("APP_PASSWORD", "admin")

    # secrets.compare_digest é mais seguro que '==' para evitar ataques de tempo
    is_user_ok = secrets.compare_digest(credentials.username, usuario_correto)
    is_pass_ok = secrets.compare_digest(credentials.password, senha_correta)

    if not (is_user_ok and is_pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais Incorretas",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# --- FIM DA CONFIGURAÇÃO DE SEGURANÇA ---


# Evento de vida útil: Carrega modelos antes do app começar
@asynccontextmanager
async def lifespan(app: FastAPI):
    core.carregar_modelos()
    yield


app = FastAPI(lifespan=lifespan)

# Configuração de templates e estáticos
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", dependencies=[Depends(verificar_credenciais)])
def home(request: Request):
    # Carrega a lista do CSV
    lista_exemplos = core.carregar_exemplos_demo()

    # Passa para o HTML
    return templates.TemplateResponse(
        "index.html", {"request": request, "exemplos": lista_exemplos}
    )


@app.post("/classificar", dependencies=[Depends(verificar_credenciais)])
def classificar(request: Request, texto_projeto: str = Form(...)):
    # Desempacota a tupla: resultado e tempo
    res_rf, tempo_rf = core.prever_rf(texto_projeto)
    res_bert, tempo_bert = core.prever_bert(texto_projeto)

    return templates.TemplateResponse(
        "resultados.html",
        {
            "request": request,
            "texto_original": texto_projeto,
            "res_rf": res_rf,
            "tempo_rf": tempo_rf,
            "res_bert": res_bert,
            "tempo_bert": tempo_bert,
        },
    )


@app.post("/analise_lime", dependencies=[Depends(verificar_credenciais)])
def analisar_lime(
    request: Request, texto: str = Form(...), modelo_escolhido: str = Form(...)
):
    from lime.lime_text import LimeTextExplainer

    # Pega os nomes das classes (garante que é uma lista de strings)
    nomes_classes = [core.MAPA_CLASSES[i] for i in range(len(core.MAPA_CLASSES))]

    explainer = LimeTextExplainer(class_names=nomes_classes)

    try:
        if modelo_escolhido == "rf":
            # Verifica se a função existe antes de chamar
            if not hasattr(core, "wrapper_lime_rf"):
                return "Erro: Função wrapper_lime_rf não encontrada no core."

            exp = explainer.explain_instance(
                texto,
                core.wrapper_lime_rf,  # <--- Sem aspas!
                num_features=10,
                num_samples=2000,
                top_labels=2,
            )

        elif modelo_escolhido == "bert":
            # Verifica se o modelo BERT está carregado
            if core.model_bert is None:
                return "Erro: Modelo BERT não está carregado na memória."

            # Verifica se o wrapper do BERT existe
            if not hasattr(core, "wrapper_lime_bert"):
                return "Erro: Função wrapper_lime_bert não implementada no core.py"

            exp = explainer.explain_instance(
                texto,
                core.wrapper_lime_bert,  # <--- Chama a nova função
                num_features=10,
                num_samples=200,  # 200 amostras para não travar o servidor
                top_labels=2,
            )

        else:
            return templates.TemplateResponse(
                "error.html", {"request": request, "msg": "Modelo inválido"}
            )

        lime_html = exp.as_html()

        return templates.TemplateResponse(
            "lime_view.html", {"request": request, "lime_html": lime_html}
        )

    except Exception as e:
        import traceback

        traceback.print_exc()  # Imprime o erro real no terminal para você ver
        return f"Erro ao gerar LIME: {str(e)}"
