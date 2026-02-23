import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_boxes(image, ocr_results):
    """
    Dessine les bounding boxes EasyOCR
    """
    img = image.copy()

    for bbox, text, conf in ocr_results:
        pts = [(int(x), int(y)) for x, y in bbox]

        # Rectangle
        cv2.polylines(img, [np.array(pts)], True, (0, 255, 0), 2)

        # Text (Optionnel)
        cv2.putText(
            img,
            text,
            pts[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()