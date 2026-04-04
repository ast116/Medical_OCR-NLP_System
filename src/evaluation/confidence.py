def compute_ocr_confidence(ocr_results):
    """
    ocr_results = [(bbox, text, confidence), ...]
    """

    if not ocr_results:
        return 0

    confidences = [res[2] for res in ocr_results if len(res) > 2]

    if not confidences:
        return 0

    return round(sum(confidences) / len(confidences), 2)


# Score extraction
def compute_extraction_confidence(lab_results):

    if not lab_results:
        return 0

    score = 0
    total = len(lab_results)

    for entry in lab_results:

        entry_score = 0

        # valeur présente
        if entry.get("value") is not None:
            entry_score += 0.4

        # unité présente
        if entry.get("unit"):
            entry_score += 0.2

        # nom test présent
        if entry.get("test_name"):
            entry_score += 0.2

        # référence présente
        if entry.get("reference_range"):
            entry_score += 0.2

        score += entry_score

    return round(score / total, 2)


# Score global
def compute_pipeline_confidence(ocr_conf, extraction_conf, severity_score):

    # pénalité si trop critique
    severity_penalty = severity_score * 0.3

    pipeline_score = (ocr_conf + extraction_conf + (1 - severity_penalty)) / 3

    return round(pipeline_score, 2)


# Fonction principale
def add_confidence_scores(data, ocr_results):

    lab_results = data.get("lab_results", [])

    ocr_conf = compute_ocr_confidence(ocr_results)
    extraction_conf = compute_extraction_confidence(lab_results)
    severity_score = data.get("severity_score", 0)

    pipeline_conf = compute_pipeline_confidence(
        ocr_conf,
        extraction_conf,
        severity_score
    )

    data["ocr_confidence"] = ocr_conf
    data["extraction_confidence"] = extraction_conf
    data["pipeline_confidence"] = pipeline_conf

    return data