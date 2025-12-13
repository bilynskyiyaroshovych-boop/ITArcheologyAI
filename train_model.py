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

# ================= PERFORMANCE =================
torch.backends.cudnn.benchmark = True

# ================= CONFIG =================
DATA_DIR = os.environ.get("DATA_DIR", r"C:\python\Archeological_Dataset")
MODEL_DIR = "models"
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "artifact_classifier.pth")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.json")

IMAGE_SIZE = 160          # ⬅ швидше ніж 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10           # ⬅ як ти просив
NUM_WORKERS = 4           # 2–4 для Windows

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ================= MODEL =================
def create_model(num_classes):
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)

    # 🔒 Freeze ALL backbone
    for param in model.parameters():
        param.requires_grad = False

    # 🔁 New classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(DEVICE)

# ================= DATA =================
def get_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ImageFolder(DATA_DIR, transform=transform)

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(CLASSES_FILE, "w") as f:
        json.dump(dataset.classes, f)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Total images: {len(dataset)}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Classes: {dataset.classes}")

    return train_loader, val_loader, len(dataset.classes)

# ================= TRAIN / VAL =================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    correct, total, loss_sum = 0, 0, 0.0

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return loss_sum / len(loader), correct / total

def validate(model, loader, criterion):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss_sum / len(loader), correct / total

# ================= MAIN =================
def train():
    train_loader, val_loader, num_classes = get_loaders()
    model = create_model(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion
        )

        print(
            f"Train acc: {train_acc:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_FILE)
            print("✔ Best model saved")

    print("\nTraining finished")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

# ================= RUN =================
if __name__ == "__main__":
    train()
