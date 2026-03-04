from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_dir, dpi=300):
    os.makedirs(output_dir, exist_ok=True)

    images = convert_from_path(pdf_path, dpi=dpi)

    image_paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"page_{i+1}.png")
        img.save(path, "PNG")
        image_paths.append(path)

    return image_paths