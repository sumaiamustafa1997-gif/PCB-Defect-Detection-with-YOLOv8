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
Install dependencies with:
```bash
pip install -r requirements.txt
