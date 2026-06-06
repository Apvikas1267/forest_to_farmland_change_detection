# AI-Based Forest-to-Farmland Change Detection Using GIS

This project is a complete, end-to-end deployable web-based GIS platform that automatically detects locations where forest land has been converted into agricultural land using satellite imagery and Deep Learning (PyTorch U-Net).

## Features
- **Deep Learning Model**: PyTorch U-Net for Semantic Segmentation.
- **Data Pipeline**: Downloads and processes the EuroSAT dataset to train the model.
- **Change Detection**: Analyzes Time 1 (Before) and Time 2 (After) images to highlight precise areas of conversion.
- **Web Dashboard**: Built with Streamlit for an interactive, purely AI-driven user experience.
- **Metrics**: Calculates converted area in hectares and generates downloadable CSV reports.

## Structure
- `frontend/`: Streamlit web dashboard code (`app.py`).
- `backend/`: GIS processing (`gis_processor.py`) and change detection logic (`change_detector.py`).
- `models/`: PyTorch U-Net architecture (`unet.py`), training script (`train.py`), and inference script (`inference.py`).
- `datasets/`: Script to automatically download and wrap EuroSAT for segmentation training (`data_prep.py`).
- `outputs/`: Directory for generated exports.
- `requirements.txt`: Project dependencies.

See `execution_guide.md` for step-by-step instructions on running the project in VS Code.
