# PCB Defect Detection with YOLOv8

A deep learning pipeline for detecting and classifying defects in printed circuit boards (PCBs) using YOLOv8. Designed for accuracy, speed, and real-world deployment.

## Objective
Detect six types of PCB defects using a trained YOLOv8 model and serve predictions through a Flask-based web interface.

## Dataset
- Source: [Roboflow PCB Dataset](https://universe.roboflow.com/firsttry-4g52s/pcb-1lhk5)
- Classes: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper

## Model Training
- Model: yolov8s.pt
- Config: 100 epochs, batch size 16, image size 640, Adam optimizer
- Results:
  - Precision: 0.974
  - Recall: 0.985
  - mAP@50: 0.992
  - mAP@50–95: 0.726
- Artifacts: best.pt, last.pt, training logs, confusion matrix

## Inference & Deployment
- Web interface built with Flask (`app.py`)
- Upload images, run predictions, and view annotated results
- Fast inference (13–45ms per image)
- Centralized saving of outputs
  
## Requirements
- Python 3.8 or higher
- All dependencies listed in requirements.txt

## How to Use This Project

1. *Clone the repository*  
   ```bash
   git clone https://github.com/sumaiamustafa1997-gif/PCB-Defect-Detection-with-YOLOv8.git
   cd PCB-Defect-Detection-with-YOLOv8
   

2. *Install dependencies*  
   Make sure you have Python ≥3.8 and install required packages:
   ```bash
   pip install -r requirements.txt
   

3. *Download the dataset*  
   - Visit [Roboflow Universe – PCB Defects Dataset](https://universe.roboflow.com/university-2xdiy/pcb-defects-chi1b/dataset/6)
   - Choose YOLOv8 format and download the ZIP
   - Extract it into the project folder

4. *Run training (optional)*  
   Using the Notebook:
   - Open main.ipynb
   - Follow the training cells to retrain on your data
   - Adjust hyperparameters as needed
   

5. *Run inference*  
   Using the Notebook:
   - Open main.ipynb
   - Upload PCB images
   - Detect defects
   - View and save annotated results

6. *Explore the app structure*  
   - app.py: Flask app entry point (optional for deployment)
   - templates/: HTML templates for web interface
   - static/: Assets like CSS or images
   - utils.py: Helper functions for preprocessing and display

## Project Structure  
```bash
PCB Defect Detection/
├── model/              # YOLOv8 model (best.pt)
├── models_1/           # Training weights (best.pt, last.pt)
├── results/            # Output predictions
├── static/             # CSS, uploaded images, annotated results
│   ├── defects/
│   ├── results/
│   ├── results2/
│   ├── uploads/
│   └── style.css
├── templates/          # HTML templates
│   └── index.html
├── app.py              # Flask app entry point
├── main.ipynb          # Training and inference notebook
├── requirements.txt    # Python dependencies
└── utils.py            # Helper functions
