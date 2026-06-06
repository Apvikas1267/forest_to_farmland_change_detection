import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Adjust sys.path to run locally or from another module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet import UNet

def load_model(model_path=None, device='cpu'):
    """
    Loads the trained U-Net model. 
    If model_path is not found, returns an untrained U-Net for demonstration.
    """
    model = UNet(n_channels=3, n_classes=3).to(device)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: Trained model not found. Using initialized (untrained) U-Net.")
        
    model.eval()
    return model

def predict_image(model, image, device='cpu'):
    """
    Runs inference on a single image.
    Returns a 2D numpy array containing class predictions.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)), # U-Net trained on 64x64 patches
        transforms.ToTensor()
    ])
    
    if isinstance(image, str):
        try:
            image = Image.open(image)
        except Exception as e:
            print(f"Error loading image {image}: {e}")
            return None
            
    image = image.convert('RGB')
        
    original_size = image.size
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        
    # Get predictions (argmax over channels)
    # Output shape: [1, 3, 64, 64] -> [64, 64]
    prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Resize back to original image size for display mapping
    # prediction contains: 0 (Other), 1 (Forest), 2 (Farmland)
    prediction_resized = np.array(Image.fromarray(prediction.astype(np.uint8)).resize(original_size, resample=Image.NEAREST))
    
    return prediction_resized

if __name__ == "__main__":
    # Test inference
    dummy_model = load_model()
    print("Model loaded successfully.")
