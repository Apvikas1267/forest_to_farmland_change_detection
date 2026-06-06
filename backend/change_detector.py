import numpy as np

def detect_forest_to_farmland(pred_t1, pred_t2):
    """
    Detects pixels where land cover changed from Forest (1) to Farmland (2).
    
    Args:
        pred_t1 (numpy.ndarray): 2D array of class predictions for Time 1.
        pred_t2 (numpy.ndarray): 2D array of class predictions for Time 2.
        
    Returns:
        numpy.ndarray: 2D binary mask where 1 indicates Forest -> Farmland conversion.
        dict: Statistics about the conversion.
    """
    # Verify same shape
    if pred_t1.shape != pred_t2.shape:
        raise ValueError("T1 and T2 predictions must have the same dimensions.")
        
    # Logic: T1 == 1 (Forest) AND T2 == 2 (Farmland)
    conversion_mask = np.logical_and(pred_t1 == 1, pred_t2 == 2).astype(np.uint8)
    
    # Calculate statistics
    total_pixels = pred_t1.size
    forest_pixels_t1 = np.sum(pred_t1 == 1)
    forest_pixels_t2 = np.sum(pred_t2 == 1)
    farmland_pixels_t1 = np.sum(pred_t1 == 2)
    farmland_pixels_t2 = np.sum(pred_t2 == 2)
    converted_pixels = np.sum(conversion_mask == 1)
    
    # Area calculation
    # Assuming 10m spatial resolution for Sentinel-2 (1 pixel = 100 m^2 = 0.01 hectares)
    pixel_area_m2 = 100
    pixel_area_ha = 0.01
    
    stats = {
        "Forest Area T1 (ha)": forest_pixels_t1 * pixel_area_ha,
        "Forest Area T2 (ha)": forest_pixels_t2 * pixel_area_ha,
        "Farmland Area T1 (ha)": farmland_pixels_t1 * pixel_area_ha,
        "Farmland Area T2 (ha)": farmland_pixels_t2 * pixel_area_ha,
        "Converted Area (m2)": converted_pixels * pixel_area_m2,
        "Converted Area (ha)": converted_pixels * pixel_area_ha,
        "Percentage Converted (%)": (converted_pixels / forest_pixels_t1 * 100) if forest_pixels_t1 > 0 else 0
    }
    
    return conversion_mask, stats
