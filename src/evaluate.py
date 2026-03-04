from src.evaluation.evaluate_ocr import evaluate_folder
from src.config.settings import GT_DIR, OCR_OUTPUT_DIR

results = evaluate_folder(GT_DIR, OCR_OUTPUT_DIR)
print("=== OCR Evaluation ===")
print(results)