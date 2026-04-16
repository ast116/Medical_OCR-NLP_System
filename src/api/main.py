from fastapi import FastAPI, UploadFile, File
from typing import Annotated, List
import os
import shutil
import uuid

from src.main_ocr_pipeline import process_image
from src.config.settings import RAW_DIR

app = FastAPI(title="Medical OCR API")


def save_upload_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


@app.post("/process")
async def process_files(
    files: Annotated[list[UploadFile], File(description="Upload medical reports (PNG, JPG, PDF)")]
):
    results = []

    for file in files:

        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(RAW_DIR, filename)

        save_upload_file(file, file_path)

        try:
            # Récupération du JSON complet
            structured_data = process_image(filename)
            results.append({
                "filename": file.filename,
                "status": "success",
                "data": structured_data
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
        print("[DEBUG] Returning response")
    return {
        "total_files": len(files),
        "results": results
    }