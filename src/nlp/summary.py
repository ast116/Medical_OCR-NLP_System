import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Même modèle (réutilisation logique)
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.eval()


# Construire résumé brut à partir du JSON
def build_summary_input(data):

    lab_results = data.get("lab_results", [])

    # On ne prend que les anomalies (important)
    abnormal = []
    seen = set()

    for entry in lab_results:
        status = entry.get("status")
        test = entry.get("test_name")

        if status in ["high", "low"] and test:
            key = (test.strip().lower(), status)
            if key in seen:
                continue
            seen.add(key)
            abnormal.append(f"{test} is {status}")

    if not abnormal:
        return "All laboratory values are within normal limits.", []

    # Limiter pour éviter prompt trop long
    abnormal = abnormal[:15]

    return " , ".join(abnormal), abnormal


def _clean_summary(text):
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split())
    return text


def _dedupe_segments(text):
    segments = [seg.strip() for seg in re.split(r"[;\\.]", text) if seg.strip()]
    if not segments:
        return text
    seen = set()
    deduped = []
    for seg in segments:
        key = seg.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(seg)
    if len(deduped) == len(segments):
        return text
    return ". ".join(deduped) + "."


def _is_repetitive(text):
    words = re.findall(r"\w+", text.lower())
    if len(words) < 8:
        return False
    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.4:
        return True
    segments = [seg.strip().lower() for seg in re.split(r"[;\\.]", text) if seg.strip()]
    if len(segments) >= 3:
        seg_ratio = len(set(segments)) / max(len(segments), 1)
        if seg_ratio < 0.6:
            return True
    return False


def _fallback_summary(abnormal):
    if not abnormal:
        return "All laboratory values are within normal limits."
    return "Abnormal results: " + "; ".join(abnormal) + "."


# Générer résumé
def generate_summary(text):

    prompt = f"""
        You are a medical assistant.

        Summarize the following lab abnormalities in 1 or 2 short clinical sentences.

        {text}

        Give a concise medical summary.
        """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=80,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# Fonction principale
def add_global_summary(data):

    summary_input, abnormal = build_summary_input(data)

    # Si tout est normal → pas besoin du modèle
    if "within normal limits" in summary_input.lower():
        data["summary"] = summary_input
        return data

    try:
        summary = generate_summary(summary_input)
    except Exception:
        summary = "Unable to generate summary."

    summary = _clean_summary(summary)
    summary = _dedupe_segments(summary)
    if _is_repetitive(summary):
        summary = _fallback_summary(abnormal)

    data["summary"] = summary

    return data
