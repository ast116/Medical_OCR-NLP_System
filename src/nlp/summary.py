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

    for entry in lab_results:
        status = entry.get("status")
        test = entry.get("test_name")

        if status in ["high", "low"]:
            abnormal.append(f"{test} is {status}")

    if not abnormal:
        return "All laboratory values are within normal limits."

    # Limiter pour éviter prompt trop long
    abnormal = abnormal[:15]

    return " ; ".join(abnormal)


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

    summary_input = build_summary_input(data)

    # Si tout est normal → pas besoin du modèle
    if "within normal limits" in summary_input.lower():
        data["summary"] = summary_input
        return data

    try:
        summary = generate_summary(summary_input)
    except Exception:
        summary = "Unable to generate summary."

    data["summary"] = summary

    return data