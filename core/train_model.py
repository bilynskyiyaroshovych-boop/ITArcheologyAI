import os
import json
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34, ResNet34_Weights

# --- PIL ---

Image.MAX_IMAGE_PIXELS = None

warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes")
warnings.filterwarnings("ignore", "Corrupt EXIF", UserWarning)
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# cuda benchmark 
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

DATA_DIR = os.environ.get("DATA_DIR", r"C:\python\Archeological_Dataset")
MODEL_DIR = "models"
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "artifact_classifier.pth")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.json")

IMAGE_SIZE = 160
BATCH_SIZE = 32
LEARNING_RATE = 0.001
# Use an environment override or a reasonable default based on CPU count
DEFAULT_WORKERS = max(0, (os.cpu_count() or 1) // 2)
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", str(DEFAULT_WORKERS)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_to_rgb(image):
    """Конвертує зображення в RGB, обробляючи палітри."""
    if image.mode == 'P':
        image = image.convert('RGBA')
    return image.convert('RGB')


class SafeImageFolder(ImageFolder):
    
    def __getitem__(self, index):
        try:
            path, target = self.samples[index]
            # regular image loading
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target
        except (OSError, FileNotFoundError, UnidentifiedImageError) as e:
            
            print(f"\n[WARNING] Skipping bad file: {self.samples[index][0]} -> {e}")
            return None

# --- filtring collate function to skip None items ---
def filter_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    
    if not batch:
        return []
    return default_collate(batch)


def create_model(num_classes: int):
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def get_loaders():
    use_pin_memory = (DEVICE.type == "cuda")

    transform = transforms.Compose([
        transforms.Lambda(safe_to_rgb),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Directory not found: {DATA_DIR}")

    
    dataset = SafeImageFolder(DATA_DIR, transform=transform)

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
        pin_memory=use_pin_memory,
        collate_fn=filter_collate 
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        collate_fn=filter_collate  
    )

    return train_loader, val_loader, len(dataset.classes)


def train(num_epochs=10, log_callback=None, progress_callback=None):
    """Train the model.

    log_callback: callable(str) for textual logs
    progress_callback: callable(dict) for structured progress updates
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def progress(info: dict):
        if progress_callback:
            try:
                progress_callback(info)
            except Exception:
                
                pass

    try:
        log(f"Using device: {DEVICE}")
        log(f"Workers: {NUM_WORKERS}")
        log("Loading data...")

        train_loader, val_loader, num_classes = get_loaders()
        log(f"Data loaded. Classes: {num_classes}")

        model = create_model(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
        best_val_acc = 0.0

        log(f"Starting training for {num_epochs} epochs...")

        total_train_batches = len(train_loader) if len(train_loader) > 0 else 1

        for epoch in range(num_epochs):
            log(f"Epoch {epoch + 1}/{num_epochs}")

            model.train()
            correct, total = 0, 0

            for i, batch in enumerate(train_loader):
                
                if not batch:
                    continue

                images, labels = batch
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

                # emit per-batch progress
                progress({
                    "phase": "train",
                    "epoch": epoch + 1,
                    "batch": i + 1,
                    "total_batches": total_train_batches,
                    "loss": loss.item(),
                })

            train_acc = correct / total if total > 0 else 0

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                total_val_batches = len(val_loader) if len(val_loader) > 0 else 1
                for j, batch in enumerate(val_loader):
                    if not batch:
                        continue

                    images, labels = batch
                    images = images.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    
                    progress({
                        "phase": "val",
                        "epoch": epoch + 1,
                        "batch": j + 1,
                        "total_batches": total_val_batches,
                    })

            val_acc = correct / total if total > 0 else 0
            log(f"  Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%}")

            # epochprogress
            progress({
                "phase": "epoch_summary",
                "epoch": epoch + 1,
                "train_acc": train_acc,
                "val_acc": val_acc,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), CHECKPOINT_FILE)
                log("  -> Best model saved!")

        log(f"Training finished. Best Val Acc: {best_val_acc:.1%}")
        progress({"phase": "finished", "best_val_acc": best_val_acc})

    except Exception as e:
        log(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    train()