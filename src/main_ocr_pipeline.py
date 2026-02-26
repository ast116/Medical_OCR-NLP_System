import os

from src.config.settings import RAW_DIR, PROCESSED_DIR, OCR_OUTPUT_DIR, RESIZE_WIDTH
from src.preprocessing.image_loader import load_image
from src.preprocessing.resize import resize_image
from src.preprocessing.denoise import denoise_image
from src.preprocessing.deskew import deskew_image
from src.preprocessing.binarization import binarize_image
from src.preprocessing.morphology import apply_morphology
from src.ocr.easyocr_engine import EasyOCREngine
from src.utils.file_utils import save_text
from src.ocr.table_segmentation import detect_columns, assign_boxes_to_columns
from src.ocr.box_utils import sort_boxes_top_to_bottom_left_to_right, group_boxes_by_line, reconstruct_text_from_lines, refine_lines
from src.ocr.postprocess_text import postprocess_medical_text


def process_image(image_filename):
    image_path = os.path.join(RAW_DIR, image_filename)

    print(f"[INFO] Loading {image_filename}")
    image = load_image(image_path)

    # Preprocessing
    image = resize_image(image, RESIZE_WIDTH)
    image = denoise_image(image)
    image = deskew_image(image)
    image = binarize_image(image)
    image = apply_morphology(image)

    # OCR avec Bounding Boxes
    ocr_engine = EasyOCREngine()
    ocr_results = ocr_engine.extract_with_boxes(image)

    # Tri Spatial des Boxes
    sorted_results = sort_boxes_top_to_bottom_left_to_right(ocr_results)

    # Regroupement par Lignes
    lines = group_boxes_by_line(sorted_results)
    lines = refine_lines(lines)

    # Texte Reconstruit à Partir des Lignes
    reconstructed_text = reconstruct_text_from_lines(lines)
    final_text = postprocess_medical_text(reconstructed_text)

    """
    # Segmentation des Colonnes
    columns = detect_columns(lines)
    table = assign_boxes_to_columns(lines, columns)
    """
    # Sauvegarde du texte reconstruit
    output_file = os.path.join(OCR_OUTPUT_DIR, image_filename + ".txt")
    save_text(final_text, output_file)

    """
    # Sauvegarde sous forme text lisible
    table_text = "\n".join(["\t".join(row) for row in table])
    """

    print(f"[SUCCESS] OCR saved to {output_file}")

if __name__ == "__main__":
    for file in os.listdir(RAW_DIR):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            process_image(file)
