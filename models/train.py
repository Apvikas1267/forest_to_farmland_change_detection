import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add project root to sys.path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.data_prep import get_dataloaders
from models.unet import UNet

def train_model(data_dir, epochs=10, batch_size=32, learning_rate=1e-3, device='cpu'):
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Preparing datasets...")
    train_loader, val_loader = get_dataloaders(root_dir=data_dir, batch_size=batch_size)
    
    # 2. Initialize Model, Loss, Optimizer
    # We have 3 classes: 0 (Other), 1 (Forest), 2 (Farmland)
    model = UNet(n_channels=3, n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    saved_models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(saved_models_dir, exist_ok=True)
    model_save_path = os.path.join(saved_models_dir, 'unet_eurosat.pth')

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            loop.set_postfix(loss=loss.item())
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation loss improved. Saving model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
            
    print("Training complete!")

if __name__ == '__main__':
    # Determine appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'eurosat_data')
    
    # Train for a few epochs for demonstration
    train_model(data_dir=dataset_path, epochs=5, batch_size=16, device=device)
