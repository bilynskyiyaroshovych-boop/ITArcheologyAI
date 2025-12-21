from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io

from core.predict import predict_pil

app = FastAPI(title="Archaeological Artifact Classifier")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None
    })


@app.post("/", response_class=HTMLResponse)
async def classify(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = predict_pil(image)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result
    })


@app.get("/health")
async def health():
    return {"status": "ok"}
