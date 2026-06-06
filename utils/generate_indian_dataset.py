import os
import random
import csv
from PIL import Image

def generate_presentation_dataset():
    # Paths
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'eurosat_data', 'eurosat', '2750')
    forest_dir = os.path.join(dataset_dir, 'Forest')
    crop_dir = os.path.join(dataset_dir, 'AnnualCrop')
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'indian_test_dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(forest_dir) or not os.path.exists(crop_dir):
        print("EuroSAT dataset not found.")
        return

    forest_files = [f for f in os.listdir(forest_dir) if f.endswith('.jpg')]
    crop_files = [f for f in os.listdir(crop_dir) if f.endswith('.jpg')]
    
    # Plausible Indian locations experiencing deforestation for agriculture
    indian_locations = [
        {"Region": "Western Ghats, Karnataka", "Lat": 14.9312, "Lon": 74.6291},
        {"Region": "Assam Tea Gardens, Assam", "Lat": 26.6528, "Lon": 92.8173},
        {"Region": "Bastar District, Chhattisgarh", "Lat": 19.2064, "Lon": 81.9360},
        {"Region": "Khammam, Telangana", "Lat": 17.2473, "Lon": 80.1514},
        {"Region": "Nilgiri Hills, Tamil Nadu", "Lat": 11.4916, "Lon": 76.7337},
        {"Region": "Sundarbans Margin, West Bengal", "Lat": 22.0911, "Lon": 88.8523},
        {"Region": "Garo Hills, Meghalaya", "Lat": 25.4670, "Lon": 90.3204},
        {"Region": "Satpura Range, Madhya Pradesh", "Lat": 22.1557, "Lon": 78.4342},
        {"Region": "Wayanad, Kerala", "Lat": 11.6854, "Lon": 76.1320},
        {"Region": "East Godavari, Andhra Pradesh", "Lat": 17.3200, "Lon": 82.0400}
    ]
    
    csv_path = os.path.join(output_dir, 'location_metadata.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ID', 'Region', 'Latitude', 'Longitude', 'Before_Year', 'After_Year', 'Before_Image', 'After_Image']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, loc in enumerate(indian_locations):
            # Generate Pair
            f_img_path = os.path.join(forest_dir, random.choice(forest_files))
            c_img_path = os.path.join(crop_dir, random.choice(crop_files))
            
            base_forest = Image.open(f_img_path).convert('RGB').resize((512, 512), Image.BICUBIC)
            crop_patch = Image.open(c_img_path).convert('RGB').resize((256, 256), Image.BICUBIC)
            
            t1_name = f"Site_{i+1}_Before.jpg"
            t2_name = f"Site_{i+1}_After.jpg"
            
            # Save T1
            base_forest.save(os.path.join(output_dir, t1_name))
            
            # Save T2 (Forest with crop cut into it in random location)
            t2_image = base_forest.copy()
            paste_x = random.randint(50, 200)
            paste_y = random.randint(50, 200)
            t2_image.paste(crop_patch, (paste_x, paste_y))
            t2_image.save(os.path.join(output_dir, t2_name))
            
            # Generate random plausible years
            before_year = random.randint(2018, 2021)
            after_year = random.randint(2023, 2026)
            
            # Write to CSV
            writer.writerow({
                'ID': f"Site_{i+1}",
                'Region': loc['Region'],
                'Latitude': loc['Lat'],
                'Longitude': loc['Lon'],
                'Before_Year': before_year,
                'After_Year': after_year,
                'Before_Image': t1_name,
                'After_Image': t2_name
            })
            
    print(f"Generated {len(indian_locations)} image pairs with Indian locations in: {output_dir}")

if __name__ == "__main__":
    generate_presentation_dataset()
