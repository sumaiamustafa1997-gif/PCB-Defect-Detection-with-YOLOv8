import cv2
import os
import glob

def is_pcb_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_ratio = green_mask.sum() / (img.shape[0] * img.shape[1] * 255)

    return green_ratio > 0.05

def find_saved_image(original_filename):
    base = os.path.splitext(os.path.basename(original_filename))[0]
    base_clean = base.replace(' ', '_').replace('(', '').replace(')', '')
    matches = glob.glob(f'static/results/{base_clean}*.jpg')
    return matches[0] if matches else None
