import json
import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "artifact_classifier.pth")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.json")

IMAGE_SIZE = 160
DEVICE = None  # will be set when torch is available

RESPONSES = {
    "ceramics": "This looks like an ancient ceramic artifact.",
    "jewelry": "This appears to be an ancient piece of jewelry.",
    "tools": "This looks like an ancient tool.",
    "fragments": "This appears to be an archaeological fragment.",
    "beads": "This looks like ancient beads.",
}

_model = None
_classes = None
_transform = None
_torch_available = False


def _load_model():
    global _model, _classes, _transform, DEVICE, _torch_available, _torch

    if _model is not None:
        return

    # Import heavy ML libs lazily so the API can start even if they're missing
    try:
        import torch
        import torch.nn as nn
        from torchvision import transforms
        from torchvision.models import resnet34
    except Exception as e:
        raise RuntimeError(f"PyTorch or torchvision import failed: {e}")

    _torch_available = True
    _torch = torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError("Model checkpoint not found")

    if not os.path.exists(CLASSES_FILE):
        raise FileNotFoundError("Classes file not found")

    with open(CLASSES_FILE, "r") as f:
        _classes = json.load(f)

    # Do not request pretrained weights at inference time (may attempt network download).
    weights = None
    _model = resnet34(weights=weights)
    _model.fc = nn.Linear(_model.fc.in_features, len(_classes))
    _model.load_state_dict(_torch.load(CHECKPOINT_FILE, map_location=DEVICE))
    _model.to(DEVICE)
    _model.eval()

    _transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict_pil(image: Image.Image):
    _load_model()

    if not _torch_available:
        raise RuntimeError("PyTorch is not available in this environment.")

    tensor = _transform(image).unsqueeze(0).to(DEVICE)

    with _torch.no_grad():
        outputs = _model(tensor)
        probs = _torch.softmax(outputs, dim=1)
        conf, pred = _torch.max(probs, 1)

    cls = _classes[pred.item()]
    confidence = conf.item()
    text = RESPONSES.get(cls.lower(), cls)

    return {
        "class": cls,
        "confidence": confidence,
        "text": text,
    }


def predict(image_path: str):
    image = Image.open(image_path).convert("RGB")
    return predict_pil(image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Archaeological artifact classification"
    )
    parser.add_argument("image", type=str, help="Path to image file")

    args = parser.parse_args()

    result = predict(args.image)

    print("Class:", result["class"])
    print("Confidence:", f"{result['confidence']:.1%}")
    print("Description:", result["text"])
