import cv2
import numpy as np

def apply_morphology(image):
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return processed
