import easyocr
from src.config.settings import OCR_LANGUAGES, USE_GPU

class EasyOCREngine:
    def __init__(self):
        self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=USE_GPU)

    def extract_text(self, image):
        results = self.reader.readtext(image)
        text = "\n".join([res[1] for res in results])
        return text
