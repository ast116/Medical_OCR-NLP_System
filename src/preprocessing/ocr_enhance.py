import cv2


def enhance_for_ocr(image):
    """
    Improve local contrast and text edges before OCR while keeping the document layout intact.
    """
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
