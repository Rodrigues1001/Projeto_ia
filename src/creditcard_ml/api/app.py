from fastapi import FastAPI
from creditcard_ml.api.routes.predict_route import router as predict_router
from creditcard_ml.api.routes.batch_route import router as batch_router

app = FastAPI(title="Credit Card Fraud API")

app.include_router(predict_router)
app.include_router(batch_router)

"""
inicia a aplicação FastAPI com Uvicorn

Esta função iniciará a aplicação FastAPI com Uvicorn
usando o host "0.0.0.0" e a porta 8000.

Ela destina-se a ser usada como ponto de entrada para a aplicação
quando executada diretamente com Python (por exemplo, `python app.py`).

Ao executar a aplicação com Docker, esta função não é necessária
pois o contêiner Docker iniciará a aplicação automaticamente.
"""
def start():
  import uvicorn
  uvicorn.run("creditcard_ml.api.app:app", host="0.0.0.0", port=8000)
