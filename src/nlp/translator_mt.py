from transformers import AutoModelForSeq2SeqLM, MarianTokenizer
import torch
import re

model_name = "Helsinki-NLP/opus-mt-en-fr"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.eval()


ENGLISH_MARKERS = {
    "the", "is", "and", "of", "may", "suggest", "within", "normal", "range",
    "value", "elevated", "high", "low", "abnormal", "results", "possible",
    "infection", "clinical", "correlation", "within normal limits"
}

FRENCH_MARKERS = {
    "dans", "limites", "normales", "valeur", "élevé", "élevée", "bas", "basse",
    "résultats", "possible", "infection", "corrélation", "clinique"
}

EN_PHRASE_MAP = {
    "within normal range.": "Dans les limites normales.",
    "within normal limits.": "Toutes les valeurs biologiques sont dans les limites normales.",
    "all laboratory values are within normal limits.": "Toutes les valeurs biologiques sont dans les limites normales.",
}

_TRANSLATION_CACHE = {}

"""
ALERTS_MAP = {
    "possible infection or inflammation": "Possible infection ou inflammation",
    "possible anemia": "Possible anémie",
    "possible polycythemia": "Possible polyglobulie",
    "possible thrombocytopenia": "Possible thrombopénie",
    "possible thrombocytosis": "Possible thrombocytose",
    "possible renal impairment": "Possible atteinte rénale",
    "possible liver dysfunction": "Possible dysfonction hépatique",
    "possible hyperglycemia / diabetes": "Possible hyperglycémie / diabète",
    "possible hypoglycemia": "Possible hypoglycémie",
    "possible electrolyte imbalance": "Possible déséquilibre électrolytique",
    "possible low albumin": "Possible hypoalbuminémie",
    "possible malaria infection": "Possible infection palustre",
}

NORMAL_FR = "Toutes les valeurs biologiques sont dans les limites normales."
NORMAL_EN = "All laboratory values are within normal limits."

"""

def _normalize(text):
    return re.sub(r"\s+", " ", text.strip())


# Détection simple anglais (robuste)
def is_english(text):
    text = _normalize(text).lower()

    if not text:
        return False

    # Phrase connue
    if text in EN_PHRASE_MAP:
        return True

    english_hits = sum(1 for w in ENGLISH_MARKERS if w in text)
    french_hits = sum(1 for w in FRENCH_MARKERS if w in text)

    return english_hits > 0 and english_hits >= french_hits


# Traduction EN → FR
def translate_en_to_fr(text):

    normalized = _normalize(text)
    lower = normalized.lower()

    if lower in EN_PHRASE_MAP:
        return EN_PHRASE_MAP[lower]

    if normalized in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[normalized]

    inputs = tokenizer(normalized, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(**inputs)

    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    _TRANSLATION_CACHE[normalized] = translated
    return translated

"""
def _translate_alerts(alerts):
    if not alerts:
        return []

    translated = []
    seen = set()
    for alert in alerts:
        if not alert:
            continue
        key = alert.strip().lower()
        fr = ALERTS_MAP.get(key)
        if fr is None:
            try:
                fr = translate_en_to_fr(alert) if is_english(alert) else alert
            except Exception:
                fr = alert
        if fr.lower() in seen:
            continue
        seen.add(fr.lower())
        translated.append(fr)
    return translated


def _combine_summary_with_alerts(summary_fr, alerts_fr):
    if not alerts_fr:
        return summary_fr

    summary_norm = _normalize(summary_fr)
    alerts_text = "; ".join(alerts_fr)

    if not summary_norm:
        return f"Alertes : {alerts_text}."

    if summary_norm.lower() in {NORMAL_FR.lower(), NORMAL_EN.lower()}:
        return f"{NORMAL_FR} Alertes : {alerts_text}."

    if alerts_text.lower() in summary_norm.lower():
        return summary_norm

    return f"{summary_norm} Alertes : {alerts_text}."

"""

# Traduction ciblée JSON
def translate_medical_json(data):

    # SUMMARY
    if "summary" in data:
        summary = data["summary"]

        if is_english(summary):
            try:
                data["summary_fr"] = translate_en_to_fr(summary)
            except Exception:
                data["summary_fr"] = summary
        else:
            data["summary_fr"] = summary

    # INTERPRETATIONS
    for entry in data.get("lab_results", []):

        interp = entry.get("interpretation")

        if not interp:
            continue

        if is_english(interp):
            try:
                entry["interpretation_fr"] = translate_en_to_fr(interp)
            except Exception:
                entry["interpretation_fr"] = interp
        else:
            entry["interpretation_fr"] = interp

    # ALERTES (traduction + combinaison dans le résumé) - désactivé pour l'instant
    # alerts = data.get("alerts", [])
    # alerts_fr = _translate_alerts(alerts)
    # if alerts_fr:
    #     data["alerts_fr"] = alerts_fr
    #
    # if "summary_fr" in data:
    #     data["summary_fr_combined"] = _combine_summary_with_alerts(data.get("summary_fr", ""), alerts_fr)

    return data
