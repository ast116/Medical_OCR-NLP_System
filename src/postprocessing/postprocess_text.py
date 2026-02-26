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

"""
def infer_missing_unit(line: str) -> str:
    lower = line.lower()

    for analyte, units in EXPECTED_UNITS.items():
        if analyte in lower:
            # valeur détectée mais aucune unité connue
            if re.search(r"\d+(\.\d+)?", line):
                if not any(u in lower for u in units):
                    return line + f" {units[0]}"

    return line
"""

def fix_missing_decimal(line: str) -> str:
    patterns = [
        # RBC : 3.5 – 6.0
        ("rbc", 10, 1),
        # Hb : 5 – 20
        ("haemoglobin", 30, 1),
        # WBC : 3 000 – 30 000
        ("wbc", 100000, 3),
    ]

    for analyte, max_val, decimals in patterns:
        if analyte in line.lower():
            match = re.search(r"\b(\d{2,5})\b", line)
            if match:
                value = int(match.group(1))
                if value > max_val:
                    corrected = str(value / (10 ** decimals))
                    return line.replace(match.group(1), corrected)

    return line

def postprocess_medical_text(text: str) -> str:
    processed = []

    for line in text.splitlines():
        line = basic_cleaning(line)
        line = normalize_units(line)
        line = correct_medical_terms(line)
        line = fix_missing_decimal(line)
       # line = infer_missing_unit(line)

        if line.strip():
            processed.append(line)

    return "\n".join(processed)

