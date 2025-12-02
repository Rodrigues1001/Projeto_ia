from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
import shutil
from creditcard_ml.core.predict_batch import predict_batch

router = APIRouter(prefix="/batch", tags=["Batch Predict"])

@router.post("/")
async def batch(file: UploadFile = File(...)):
    input_path = f"/app/{file.filename}"
    output_path = f"/app/output_{file.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result_path = predict_batch(input_path, output_path)

    return FileResponse(
        result_path,
        filename=f"output_{file.filename}",
        media_type="text/csv"
    )
