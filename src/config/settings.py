import os

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OCR_OUTPUT_DIR = os.path.join(DATA_DIR, "ocr_output")
GT_DIR = os.path.join(DATA_DIR, "gt")

# OCR
OCR_LANGUAGES = ['en']  # adapter si nécessaire ['fr','en']
USE_GPU = os.getenv("USE_GPU", "false").lower() in {"1", "true", "yes", "on"}

# Image preprocessing parameters
RESIZE_WIDTH = 1500
