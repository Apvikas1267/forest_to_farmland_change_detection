# Execution Guide: VS Code Setup

Follow these exact steps to run the Forest-to-Farmland Change Detection project in Visual Studio Code.

## Step 1: Open the Project in VS Code
1. Open VS Code.
2. Go to **File > Open Folder**.
3. Select the `forest_farmland_gis` folder (located at `C:\Users\AP VIKAS\.gemini\antigravity\scratch\forest_farmland_gis`).

## Step 2: Set Up Python Environment
1. Open a new terminal in VS Code (`Terminal > New Terminal`).
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - On Windows (Command Prompt):
     ```cmd
     .\venv\Scripts\activate
     ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Step 3: Train the PyTorch U-Net Model
Before running the website, you must train the AI model so it can accurately predict forest and farmland. The dataset (EuroSAT) will be downloaded automatically during this step.

1. In the VS Code terminal, run the training script:
   ```bash
   python models/train.py
   ```
2. Wait for the dataset to download and the training to finish. The script will save the trained weights into `models/saved_models/unet_eurosat.pth`.

*Note: If you skip this step, the website will still launch, but it will use an "untrained" model that produces random outputs to demonstrate functionality.*

## Step 4: Run the Streamlit Website
1. In the VS Code terminal, launch the frontend application:
   ```bash
   streamlit run frontend/app.py
   ```
2. A browser window will automatically open at `http://localhost:8501`.

## Step 5: Test the AI Change Detection
1. In the web interface, upload a "Before Image" (T1) and an "After Image" (T2). You can use any satellite imagery (JPG, PNG, TIFF) showing land cover.
2. The AI model will process both images, identify the Forest and Farmland regions, and generate a change detection mask.
3. Scroll down to the GIS Viewer to see the original images, the predicted land cover maps, and the final conversion overlay (highlighted in red).
4. Review the automatically calculated Statistics and Charts.
5. Click **Download Results as CSV** to export the area report.
