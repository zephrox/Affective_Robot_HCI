import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys

# Dynamically set the base directory to the folder containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.model import LightweightEmotionCNN

def train():
    print("--- STARTING SYSTEM HEALTH CHECK ---")
    data_dir = os.path.join(BASE_DIR, 'data', 'fer2013')
    train_dir = os.path.join(data_dir, 'train')
    
    if not os.path.exists(train_dir):
        print(f"ERROR: Directory missing -> {train_dir}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print(f"Scanning dataset in: {train_dir}...")
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        print(f"SUCCESS: Found {len(train_dataset)} images across {len(train_dataset.classes)} classes.")
        print(f"Detected Classes: {train_dataset.classes}")
    except Exception as e:
        print(f"\nDATASET ERROR: {e}")
        print("Your folder structure must look exactly like this:")
        print("data/fer2013/train/happy/image1.jpg")
        print("data/fer2013/train/sad/image2.jpg")
        return

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = LightweightEmotionCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    print("--- HEALTH CHECK PASSED. BEGINNING EPOCH 1 ---")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 100 batches to prove it hasn't frozen
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
        print(f"Epoch {epoch+1}/{epochs} Completed - Avg Loss: {running_loss/len(train_loader):.4f}")

    save_path = os.path.join(BASE_DIR, 'models', 'emotion_model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model saved to {save_path}")

if __name__ == "__main__":
    train()