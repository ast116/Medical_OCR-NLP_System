import cv2

def binarize_image(image, upscale_factor=1.5):
    """
    Binarisation avancée avec upscaling pour OCR
    """
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale (important pour petits caractères)
    if upscale_factor != 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=upscale_factor,
            fy=upscale_factor,
            interpolation=cv2.INTER_CUBIC
        )

    # Adaptive Threshold (meilleur que Otsu pour scans)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return binary
