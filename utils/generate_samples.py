import os
import random
from PIL import Image

def generate_sample_images():
    # Paths to the downloaded EuroSAT dataset
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'eurosat_data', 'eurosat', '2750')
    forest_dir = os.path.join(dataset_dir, 'Forest')
    crop_dir = os.path.join(dataset_dir, 'AnnualCrop')
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'sample_images')
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(forest_dir) or not os.path.exists(crop_dir):
        print("EuroSAT dataset not found. Please ensure training completed successfully.")
        return

    # Grab a random Forest image and a random Crop image
    forest_files = [f for f in os.listdir(forest_dir) if f.endswith('.jpg')]
    crop_files = [f for f in os.listdir(crop_dir) if f.endswith('.jpg')]
    
    forest_img_path = os.path.join(forest_dir, random.choice(forest_files))
    crop_img_path = os.path.join(crop_dir, random.choice(crop_files))
    
    # Open images (EuroSAT patches are 64x64)
    base_forest = Image.open(forest_img_path).convert('RGB')
    crop_patch = Image.open(crop_img_path).convert('RGB')
    
    # Resize to 512x512 so they look high-quality on the dashboard
    base_forest = base_forest.resize((512, 512), Image.BICUBIC)
    crop_patch = crop_patch.resize((256, 256), Image.BICUBIC)
    
    # T1: Before (Pure Forest)
    t1_image = base_forest.copy()
    t1_path = os.path.join(output_dir, 'T1_Before_Forest.jpg')
    t1_image.save(t1_path)
    
    # T2: After (Forest with Farmland cut into the middle)
    t2_image = base_forest.copy()
    # Paste the crop patch in the center to simulate deforestation for agriculture
    paste_x = (512 - 256) // 2
    paste_y = (512 - 256) // 2
    t2_image.paste(crop_patch, (paste_x, paste_y))
    
    t2_path = os.path.join(output_dir, 'T2_After_Farmland.jpg')
    t2_image.save(t2_path)
    
    print(f"Sample images successfully generated in:\n{output_dir}")
    print(f"1. {t1_path}")
    print(f"2. {t2_path}")

if __name__ == "__main__":
    generate_sample_images()
