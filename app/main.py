from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import core


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
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )


@app.post("/classificar")
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
