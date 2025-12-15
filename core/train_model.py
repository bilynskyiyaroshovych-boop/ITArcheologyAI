import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34, ResNet34_Weights
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

DATA_DIR = os.environ.get(
    "DATA_DIR", r"C:\python\Archeological_Dataset"
)

MODEL_DIR = "models"
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "artifact_classifier.pth")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.json")

IMAGE_SIZE = 160
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(num_classes: int):
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def get_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = ImageFolder(DATA_DIR, transform=transform)

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(CLASSES_FILE, "w") as f:
        json.dump(dataset.classes, f, indent=2)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, len(dataset.classes)


def train(num_epochs=10, log_callback=None):
   
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    try:
        log("Loading data...")
        train_loader, val_loader, num_classes = get_loaders()
        log(f"Data loaded. Classes: {num_classes}")
        
        model = create_model(num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

        best_val_acc = 0.0

        log(f"Starting training for {num_epochs} epochs on {DEVICE}...")

        for epoch in range(num_epochs):
            log(f"Epoch {epoch + 1}/{num_epochs}")

            model.train()
            correct, total = 0, 0

            # Training loop
           
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total if total > 0 else 0

            # Validation loop
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total if total > 0 else 0

            log(f"  Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), CHECKPOINT_FILE)
                log("  -> Best model saved!")

        log(f"Training finished. Best Val Acc: {best_val_acc:.1%}")
    
    except Exception as e:
        log(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    train()