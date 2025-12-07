import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import json
from datetime import datetime

# Configuration
DATA_DIR = os.environ.get("DATA_DIR", r"C:\python\Archeological_Dataset")
MODEL_DIR = "models"
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "artifact_classifier.pth")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.json")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Data directory: {DATA_DIR}")


def create_model(num_classes):
    """
    Load pretrained ResNet18 and adapt for artifact classification.
    """
    model = models.resnet34(pretrained=True)
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model.to(DEVICE)


def get_data_loaders():
    """
    Create training and validation data loaders.
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset from Archaeological_Dataset folder
    dataset = ImageFolder(root=DATA_DIR, transform=transform)
    
    if len(dataset) == 0:
        print(f"Error: No images found in {DATA_DIR}")
        return None, None, None
    
    # Save class names
    os.makedirs(MODEL_DIR, exist_ok=True)
    classes = dataset.classes
    with open(CLASSES_FILE, 'w') as f:
        json.dump(classes, f)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Total images: {len(dataset)}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Classes: {classes}")
    
    return train_loader, val_loader, len(classes)


def train_epoch(model, train_loader, criterion, optimizer):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'acc': correct / total
        })
    
    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion):
    """
    Validate the model.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': correct / total
            })
    
    return total_loss / len(val_loader), correct / total


def train_model():
    """
    Full training pipeline.
    """
    print("\n=== Archaeological Artifact Classifier Training ===\n")
    
    # Load data
    train_loader, val_loader, num_classes = get_data_loaders()
    if train_loader is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create model
    model = create_model(num_classes)
    print(f"\nModel: ResNet18 with {num_classes} output classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    print(f"\nTraining for {NUM_EPOCHS} epochs...\n")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_FILE)
            print(f"Checkpoint saved! (Best Val Acc: {best_val_acc:.4f})")
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {CHECKPOINT_FILE}")


def predict(image_path, model_path=CHECKPOINT_FILE, classes_path=CLASSES_FILE):
    """
    Predict class for a single image.
    """
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        print(f"Model or classes file not found. Please train the model first.")
        return None
    
    # Load classes
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    
    # Load model
    model = create_model(len(classes))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Load and preprocess image
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item()
    
    return {
        'class': predicted_class,
        'confidence': confidence_score,
        'all_probabilities': {classes[i]: probabilities[0][i].item() for i in range(len(classes))}
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train artifact classifier")
    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                       help='Mode: train or predict')
    parser.add_argument('--image', type=str, help='Image path for prediction')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        if not args.image:
            print("Please provide --image for prediction mode")
        else:
            result = predict(args.image)
            if result:
                print(f"\nPrediction: {result['class']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"\nAll probabilities:")
                for cls, prob in result['all_probabilities'].items():
                    print(f"  {cls}: {prob:.4f}")
