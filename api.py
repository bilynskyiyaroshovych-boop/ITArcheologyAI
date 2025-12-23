import base64
import io

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from core.predict import predict_pil

app = FastAPI(title="Archaeological Artifact Classifier")
#
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": None, "image_data": None}
    )


@app.post("/", response_class=HTMLResponse)
async def classify(request: Request, file: UploadFile | None = File(None)):

    if file is None:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "image_data": None, "error": None},
        )

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    try:
        result = predict_pil(image)
        error = None
    except FileNotFoundError:
        result = None
        error = "Model checkpoint not found. Please train the model first."
    except Exception as e:
        result = None
        error = f"Prediction error: {e}"

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    image_data = f"data:image/jpeg;base64,{encoded_image}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "image_data": image_data,
            "error": error,
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
