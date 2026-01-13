import base64  
import io
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles 
from PIL import Image

from core.predict import predict_pil

app = FastAPI(title="Archaeological Artifact Classifier")
# 
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "image_data": None
    })

@app.post("/", response_class=HTMLResponse)
async def classify(request: Request, file: UploadFile = File(...)):

    image_bytes = await file.read()
    

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    result = predict_pil(image)
    
  
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    image_data = f"data:image/jpeg;base64,{encoded_image}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "image_data": image_data  
    })

@app.get("/health")
async def health():
    return {"status": "ok"}