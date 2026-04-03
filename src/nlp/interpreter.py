from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Chargement du modèle
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Mode évaluation (important)
model.eval()

# Calcul du status
def compute_status(value, ref, flags=None):
    if flags:
        if "H" in flags:
            return "high"
        if "L" in flags:
            return "low"

    if not ref:
        return "unknown"

    low = ref.get("low")
    high = ref.get("high")

    if low is not None and value < low:
        return "low"
    elif high is not None and value > high:
        return "high"

    return "normal"


# Prompt
def build_prompt(entry, status):
    ref = entry.get("reference_range")

    ref_text = ""
    if ref and ref.get("low") is not None and ref.get("high") is not None:
        ref_text = f"(normal range: {ref['low']} - {ref['high']})"

    return f"""
        You are a medical assistant.

        Interpret this laboratory result in one short and clear sentence.

        Test: {entry.get('test_name', '')}
        Value: {entry.get('value', '')} {entry.get('unit', '')}
        Status: {status}
        {ref_text}

        Give a concise clinical interpretation.
        """


# Génération optimisée
def generate_interpretation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():  # 🔥 très important pour perf
        outputs = model.generate(
            **inputs,
            max_length=100,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# Fonction principale
def enrich_lab_results(data):
    lab_results = data.get("lab_results", [])

    for entry in lab_results:

        if "value" not in entry:
            continue

        # Status
        status = compute_status(
            entry.get("value"),
            entry.get("reference_range"),
            entry.get("flags")
        )

        entry["status"] = status

        # Skip si normal
        if status == "normal":
            entry["interpretation"] = "Within normal range."
            continue

        prompt = build_prompt(entry, status)

        try:
            interpretation = generate_interpretation(prompt)
        except Exception:
            interpretation = "Unable to interpret result."

        entry["interpretation"] = interpretation

    return data