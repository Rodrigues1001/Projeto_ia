from fastapi import FastAPI
from creditcard_ml.api.routes.predict_route import router as predict_router
from creditcard_ml.api.routes.batch_route import router as batch_router
from creditcard_ml.api.routes.mlops_route import router as mlops_router

app = FastAPI(title="Credit Card Fraud API")

app.include_router(predict_router)
app.include_router(batch_router)
app.include_router(mlops_router)

def start():
    import uvicorn
    uvicorn.run(
        "creditcard_ml.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
