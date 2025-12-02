from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
import shutil
from creditcard_ml.core.predict_batch import predict_batch

router = APIRouter()

"""
Faça uma previsão com base em um conjunto de dados.

Args:
  file: Arquivo: O arquivo sobre o qual serão feitas as previsões.

Returns:
  Um arquivo de resposta (FileResponse) contendo as previsões em formato CSV.
"""
@router.post("/batch")
async def batch(file: UploadFile = File(...)):
  input_path = f"/app/{file.filename}"
  output_path = f"/app/output_{file.filename}"

  with open(input_path, "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)

  result_file = predict_batch(input_path, output_path)

  return FileResponse(
    path=result_file,
    filename=f"output_{file.filename}",
    media_type="text/csv"
  )
