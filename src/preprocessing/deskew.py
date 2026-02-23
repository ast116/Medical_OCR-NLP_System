import cv2
import numpy as np

def deskew_image(image):
    """
    Corrige l'inclinaison d'une image scannée (deskew)
    """
    # Conversion en niveaux de gris si nécessaire
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Binarisation (inversion nécessaire pour détecter le texte)
    _, thresh = cv2.threshold(
           gray, 0, 255,
           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # Récupération des coordonnées des pixels texte
    coords = np.column_stack(np.where(thresh > 0))
    
    # Sécurité : image vide
    if coords.shape[0] == 0:
        return image

    # Calcul de l'angle minimal
    angle = cv2.minAreaRect(coords)[-1]

    # Ajustement de l'angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Rotation de l'image pour corriger l'inclinaison (Ou Rotation Inverse)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
         image, M, (w, h),
         flags = cv2.INTER_CUBIC,
         borderMode=cv2.BORDER_REPLICATE
    )
    return rotated