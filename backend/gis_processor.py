import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

def overlay_mask_on_image(image, mask, alpha=0.5, color=(255, 0, 0)):
    """
    Overlays a binary mask onto an image.
    
    Args:
        image (PIL.Image or str): Original image object or path.
        mask (numpy.ndarray): 2D binary mask.
        alpha (float): Transparency of the overlay.
        color (tuple): RGB color for the overlay.
        
    Returns:
        PIL.Image: The blended image.
    """
    if isinstance(image, str):
        image = Image.open(image)
    base_image = image.convert('RGBA')
    
    # Create an RGBA image for the mask
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    
    # Where mask is 1, set color and alpha
    mask_rgba[mask == 1, 0] = color[0] # R
    mask_rgba[mask == 1, 1] = color[1] # G
    mask_rgba[mask == 1, 2] = color[2] # B
    mask_rgba[mask == 1, 3] = int(255 * alpha) # A
    
    mask_image = Image.fromarray(mask_rgba, mode='RGBA')
    
    # Resize mask to base image size if needed
    if mask_image.size != base_image.size:
        mask_image = mask_image.resize(base_image.size, resample=Image.NEAREST)
        
    # Composite the images
    combined = Image.alpha_composite(base_image, mask_image)
    return combined.convert('RGB')

def colorize_prediction(prediction):
    """
    Colorizes a class prediction array.
    0: Other (Gray)
    1: Forest (Green)
    2: Farmland (Yellow)
    """
    h, w = prediction.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    color_map[prediction == 0] = [200, 200, 200] # Gray
    color_map[prediction == 1] = [34, 139, 34]   # Forest Green
    color_map[prediction == 2] = [255, 215, 0]   # Gold/Yellow
    
    return Image.fromarray(color_map)
