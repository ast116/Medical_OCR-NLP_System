import easyocr
from src.config.settings import OCR_LANGUAGES, USE_GPU

class EasyOCREngine:
    def __init__(self):
        self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=USE_GPU)

    def extract_text(self, image) -> str:
        results = self.reader.readtext(image)
        text = "\n".join([text for _, text, _ in results])
        return text
    
    def extract_with_boxes(self, image):
        """
        Retourne texte + bounding boxes + confidence
        """
        return self.reader.readtext(image, detail=1)
    
