import os

from src.config.settings import RAW_DIR, PROCESSED_DIR, OCR_OUTPUT_DIR, RESIZE_WIDTH
from src.preprocessing.image_loader import load_image
from src.preprocessing.resize import resize_image
from src.preprocessing.denoise import denoise_image
from src.preprocessing.binarization import binarize_image
from src.preprocessing.morphology import apply_morphology
from src.ocr.easyocr_engine import EasyOCREngine
from src.utils.file_utils import save_text

def process_image(image_filename):
    image_path = os.path.join(RAW_DIR, image_filename)

    print(f"[INFO] Loading {image_filename}")
    image = load_image(image_path)

    image = resize_image(image, RESIZE_WIDTH)
    image = denoise_image(image)
    image = binarize_image(image)
    image = apply_morphology(image)

    ocr_engine = EasyOCREngine()
    text = ocr_engine.extract_text(image)

    output_file = os.path.join(OCR_OUTPUT_DIR, image_filename + ".txt")
    save_text(text, output_file)

    print(f"[SUCCESS] OCR saved to {output_file}")

if __name__ == "__main__":
    for file in os.listdir(RAW_DIR):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            process_image(file)
