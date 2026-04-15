from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    return {"msg": "ok"}