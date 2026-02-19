import cv2

def resize_image(image, width):
    h, w = image.shape[:2]
    ratio = width / w
    new_dim = (width, int(h * ratio))
    resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    return resized
