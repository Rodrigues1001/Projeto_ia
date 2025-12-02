from fastapi import APIRouter
from creditcard_ml.api.schemas.predict_schema import PredictRequest
from creditcard_ml.core.predict_single import predict_single

router = APIRouter(prefix="/predict", tags=["Predict"])

@router.post("/")
def predict(data: PredictRequest):
    return predict_single(data.dict())
