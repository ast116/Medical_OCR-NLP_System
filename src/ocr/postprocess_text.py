import re

def basic_cleaning(text: str) -> str:
    """
    Nettoyage basique du texte OCR
    (préserve STRICTEMENT les retours à la ligne)
    """

    cleaned_lines = []

    for line in text.splitlines():
        # Supprimer caractères parasites
        line = re.sub(r"[{#@_]", "", line)

        # Supprimer apostrophes parasites
        line = line.replace("'", "")

        # Virgule décimale → point
        line = re.sub(r"(\d),(\d)", r"\1.\2", line)

        # Réduction des espaces (ESPACES SEULEMENT)
        line = re.sub(r"[ \t]+", " ", line)

        # Nettoyage espaces autour de %
        line = re.sub(r"\s*%\s*", " % ", line)

        cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines)

def normalize_units(text: str) -> str:
    """
    Normalisation des unités biologiques courantes
    """

    unit_map = {
        r"\bgmldl\b|\bgldl\b|\bgmidl\b": "g/dL",
        r"\bfl\b": "fL",
        r"\bpg\b": "pg",
        r"\biul\b|/ul\b|lul\b": "/µL",
        r"\bmilllcmm\b|\bmil/cumm\b": "mill/cmm",
    }

    for pattern, replacement in unit_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

def correct_medical_terms(text: str) -> str:
    """
    Corrections de termes médicaux fréquemment mal reconnus
    """

    corrections = {
        r"\bWAB\.?C\.?\b": "WBC",
        r"\bDWFFERENTIALCQUNT\b": "DIFFERENTIAL COUNT",
        r"\bMG;V:?\b": "MCV",
        r"\bMCiH\b": "MCH",
        r"\bMC\.?H\.?C\.?\b": "MCHC",
        r"\bPCVIHCT\b": "PCV/HCT",
    }

    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

def postprocess_medical_text(text: str) -> str:
    """
    Pipeline complet de post-processing médical
    """

    text = basic_cleaning(text)
    text = normalize_units(text)
    text = correct_medical_terms(text)

    return text

