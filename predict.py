import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import argparse
from pathlib import Path

# Configuration
MODEL_DIR = "models"
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "artifact_classifier.pth")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.json")
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Response templates for each artifact type
RESPONSES = {
    "ceramics": "This looks like an ancient ceramic artifact. Ceramics are among the most common archaeological finds, often used for storage, cooking, and ritual purposes in ancient civilizations.",
    "jewelry": "This appears to be an ancient piece of jewelry. Jewelry artifacts reveal information about the craftsmanship, trade routes, and aesthetic preferences of ancient societies.",
    "tools": "This looks like an ancient tool. Tools are crucial archaeological finds that help us understand the technological development and daily activities of past civilizations.",
    "fragments": "This appears to be an archaeological fragment or broken piece. Fragments often provide important clues about larger artifacts and ancient manufacturing techniques.",
    "beads": "This looks like ancient beads. Beads were used for decoration, trade, and possibly spiritual purposes in ancient cultures and reveal much about social structure and commerce.",
}


def create_model(num_classes):
    """
    Load pretrained ResNet18 and adapt for artifact classification.
    """
    model = models.resnet18(pretrained=True)
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model.to(DEVICE)


def predict(image_path, model_path=CHECKPOINT_FILE, classes_path=CLASSES_FILE):
    """
    Predict artifact class for an image and return descriptive text response.
    """
    # Check if model exists
    if not os.path.exists(model_path):
        return {
            'success': False,
            'error': f"Model not found at {model_path}. Please train the model first using: python train_model.py --mode train",
            'text': None,
            'class': None,
            'confidence': None
        }
    
    if not os.path.exists(classes_path):
        return {
            'success': False,
            'error': f"Classes file not found at {classes_path}.",
            'text': None,
            'class': None,
            'confidence': None
        }
    
    # Check if image exists
    if not os.path.exists(image_path):
        return {
            'success': False,
            'error': f"Image not found: {image_path}",
            'text': None,
            'class': None,
            'confidence': None
        }
    
    try:
        # Load classes
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        
        # Load model
        model = create_model(len(classes))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = classes[predicted.item()]
        confidence_score = confidence.item()
        
        # Generate descriptive response
        response_text = RESPONSES.get(predicted_class.lower(), 
                                     f"This appears to be a {predicted_class} artifact (confidence: {confidence_score:.1%}).")
        
        # Add confidence info
        if confidence_score < 0.6:
            response_text += f"\n[Low confidence: {confidence_score:.1%} - classification may not be accurate]"
        elif confidence_score >= 0.8:
            response_text += f"\n[High confidence: {confidence_score:.1%}]"
        
        return {
            'success': True,
            'error': None,
            'class': predicted_class,
            'confidence': confidence_score,
            'text': response_text,
            'all_probabilities': {classes[i]: probabilities[0][i].item() for i in range(len(classes))}
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Error during prediction: {str(e)}",
            'text': None,
            'class': None,
            'confidence': None
        }


def main():
    parser = argparse.ArgumentParser(description="Classify archaeological artifact from image")
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed probabilities')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Archaeological Artifact Classifier")
    print("="*60 + "\n")
    
    result = predict(args.image)
    
    if not result['success']:
        print(f" Error: {result['error']}")
        return
    
    print(f"Image: {args.image}")
    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']:.1%}\n")
    print(f"Description:\n{result['text']}\n")
    
    if args.verbose:
        print("All probabilities:")
        for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  {cls:15s} | {bar} | {prob:.1%}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
