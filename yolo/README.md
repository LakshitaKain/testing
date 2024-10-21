# YOLO Model Training and Inference

This repository contains scripts and configurations to train and perform inference using a YOLO model.

## Directory Structure
- **data/**: Contains datasets and related files.
- **models/**: Contains the trained YOLO model(s) and checkpoints (e.g., best.pt).
- **scripts/**: Contains the training (`train_yolo.py`), inference (`inference.py`), and any other scripts.

## How to Use
1. Clone the repository.
2. Download your dataset using the Roboflow API or use an existing dataset.
3. Run the training script: `python3 scripts/train_yolo.py`.
4. Run inference on a test image: `python3 scripts/inference.py`.

## Environment Variables
- Ensure that your `.env` file contains the correct `ROBOFLOW_API` key.

## Requirements
Install the necessary packages:
```bash
pip install -r requirements.txt
