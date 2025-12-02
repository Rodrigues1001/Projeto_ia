from fastapi import APIRouter
from creditcard_ml.api.schemas.predict_schema import PredictRequest
from creditcard_ml.core.predict_single import predict_single

router = APIRouter()

"""
Recebe um objeto PredictRequest e retorna as probabilidades de fraude
calculadas com base nos dados do objeto.

:param data: PredictRequest
:return: dict
"""
@router.post("/predict")
def predict(data: PredictRequest):
  return predict_single(data.dict())
