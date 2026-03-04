import re
from .ocr_metrics import cer

def extract_medical_tokens(text):
    return " ".join(
        re.findall(r"\d+(?:[.,]\d+)?|g/dl|fL|pg|%|/µl|mg/dl|IU/L|Gm/dL|g/dl|gm/dl|mill/cmm|cells/cumm|mg/L", text, flags=re.I)
    )

def medical_cer(reference, hypothesis):
    ref = extract_medical_tokens(reference)
    hyp = extract_medical_tokens(hypothesis)
    return cer(ref, hyp)