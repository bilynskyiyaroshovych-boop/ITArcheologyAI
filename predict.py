import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
import argparse

# ================= CONFIG =================
MODEL_DIR = "models"
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "artifact_classifier.pth")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.json")
IMAGE_SIZE = 160

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESPONSES = {
    "ceramics": "This looks like an ancient ceramic artifact.",
    "jewelry": "This appears to be an ancient piece of jewelry.",
    "tools": "This looks like an ancient tool.",
    "fragments": "This appears to be an archaeological fragment.",
    "beads": "This looks like ancient beads.",
}

# ================= MODEL =================
def create_model(num_classes):
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

# ================= PREDICT =================
def predict(image_path):
    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError("Model checkpoint not found")

    if not os.path.exists(CLASSES_FILE):
        raise FileNotFoundError("Classes file not found")

    with open(CLASSES_FILE, "r") as f:
        classes = json.load(f)

    model = create_model(len(classes))
    model.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    cls = classes[pred.item()]
    confidence = conf.item()

    text = RESPONSES.get(cls.lower(), f"Predicted class: {cls}")

    return cls, confidence, probs[0]

# ================= CLI =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    args = parser.parse_args()

    cls, conf, probs = predict(args.image)

    print(f"\nClass: {cls}")
    print(f"Confidence: {conf:.1%}")
    print("\nAll probabilities:")
    for i, p in enumerate(probs):
        print(f"{i:2d} {p.item():.3f}")
