import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np

class EuroSATSegmentation(Dataset):
    """
    A custom PyTorch Dataset that wraps the EuroSAT image classification dataset 
    and generates simulated segmentation masks so it can be used to train a U-Net.
    
    EuroSAT has classes like: Forest, AnnualCrop, Pasture, PermanentCrop, etc.
    We map these to: 0 (Background), 1 (Forest), 2 (Farmland).
    """
    def __init__(self, root_dir, download=True, transform=None):
        self.root_dir = root_dir
        # EuroSAT handles its own download. We use the RGB version.
        self.eurosat = datasets.EuroSAT(root=root_dir, download=download, transform=transforms.ToTensor())
        self.transform = transform
        
        # EuroSAT class indices mapping (approximate, based on standard EuroSAT RGB):
        # We will map them dynamically based on class names.
        self.class_to_idx = self.eurosat.class_to_idx
        
        # Mapping EuroSAT classes to our 3 classes (0: Other, 1: Forest, 2: Farmland)
        self.segmentation_mapping = {}
        for class_name, idx in self.class_to_idx.items():
            class_name_lower = class_name.lower()
            if 'forest' in class_name_lower:
                self.segmentation_mapping[idx] = 1 # Forest
            elif 'crop' in class_name_lower or 'pasture' in class_name_lower:
                self.segmentation_mapping[idx] = 2 # Farmland
            else:
                self.segmentation_mapping[idx] = 0 # Other (River, Highway, Residential, etc.)

    def __len__(self):
        return len(self.eurosat)

    def __getitem__(self, idx):
        # Get image (tensor) and classification label
        image, label = self.eurosat[idx]
        
        # Determine the target segmentation class
        seg_class = self.segmentation_mapping[label]
        
        # Create a 64x64 segmentation mask where all pixels belong to the class
        # (Since EuroSAT patches are 64x64 and mostly homogeneous)
        _, h, w = image.shape
        mask = torch.full((h, w), seg_class, dtype=torch.long)
        
        if self.transform:
            # Note: Complex transforms requiring joint image/mask ops would need custom logic.
            pass
            
        return image, mask

def get_dataloaders(root_dir, batch_size=32, train_split=0.8):
    """
    Downloads EuroSAT (if needed), creates train/val splits, and returns DataLoaders.
    """
    dataset = EuroSATSegmentation(root_dir=root_dir, download=True)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    print("Testing Dataset Download and Preparation...")
    dataset_path = os.path.join(os.path.dirname(__file__), 'eurosat_data')
    train_loader, val_loader = get_dataloaders(root_dir=dataset_path, batch_size=4)
    images, masks = next(iter(train_loader))
    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Masks Shape: {masks.shape}")
    print("Dataset setup successful!")
