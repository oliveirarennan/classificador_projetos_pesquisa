# app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import core


# Evento de vida útil: Carrega modelos antes do app começar
@asynccontextmanager
async def lifespan(app: FastAPI):
    core.carregar_modelos()
    yield


app = FastAPI(lifespan=lifespan)

# Configuração de templates e estáticos
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def home(request: Request):
    # Carrega a lista do CSV
    lista_exemplos = core.carregar_exemplos_demo()

    # Passa para o HTML
    return templates.TemplateResponse(
        "index.html", {"request": request, "exemplos": lista_exemplos}
    )


@app.post("/classificar")
def classificar(request: Request, texto_projeto: str = Form(...)):
    # 1. Pega as predições dos dois modelos
    resultado_rf = core.prever_rf(texto_projeto)
    resultado_bert = core.prever_bert(texto_projeto)

    # 2. Renderiza a página de resultados
    return templates.TemplateResponse(
        "resultados.html",
        {
            "request": request,
            "texto_original": texto_projeto,
            "res_rf": resultado_rf,
            "res_bert": resultado_bert,
        },
    )


@app.post("/analise_lime")
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
