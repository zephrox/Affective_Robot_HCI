import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys

# Add parent directory to path to import our model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import LightweightEmotionCNN

def test_accuracy():
    # SETUP: Expecting data in 'data/fer2013/test'
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data', 'fer2013')
    test_dir = os.path.join(data_dir, 'test')
    weights_path = os.path.join(project_root, 'models', 'emotion_model.pth')
    
    if not os.path.exists(test_dir):
        print(f"ERROR: Test dataset not found at {test_dir}")
        return
    if not os.path.exists(weights_path):
        print(f"ERROR: Model weights not found at {weights_path}. Train the model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Standard testing transforms (NO data augmentation here, only format matching)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = LightweightEmotionCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # Crucial: disables dropout and locks batchnorm layers for evaluation

    correct = 0
    total = 0

    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Complete: Model Accuracy on FER2013 Test Set: {accuracy:.2f}%")

if __name__ == "__main__":
    test_accuracy()