def detect_conditions(lab_results):
    conditions = []

    # Flags simples
    has_high_wbc = False
    has_low_hemoglobin = False

    for entry in lab_results:
        name = entry.get("test_name", "").lower()
        status = entry.get("status")

        # Infection (WBC élevé)
        if "wbc" in name or "leukocyte" in name:
            if status == "high":
                has_high_wbc = True

        # Anémie (Hb faible)
        if "hemoglobin" in name or "hb" in name:
            if status == "low":
                has_low_hemoglobin = True

    if has_high_wbc:
        conditions.append("possible infection")

    if has_low_hemoglobin:
        conditions.append("possible anemia")

    return conditions


# Score de gravité
def compute_severity_score(lab_results):
    abnormal_count = 0
    total = len(lab_results)

    weight = 0

    for entry in lab_results:
        status = entry.get("status")

        if status in ["high", "low"]:
            abnormal_count += 1

            # pondération simple
            if status == "high":
                weight += 1.2
            else:
                weight += 1

    if total == 0:
        return 0

    score = (weight / total)

    # normalisation (max ~2 → ramené à 1)
    return min(score / 2, 1.0)


# Priorité clinique
def compute_priority(severity_score, conditions):

    if severity_score > 0.7:
        return "urgent"

    if "possible infection" in conditions and severity_score > 0.5:
        return "urgent"

    return "normal"


# Fonction principale
def add_clinical_analysis(data):

    lab_results = data.get("lab_results", [])

    # Détection conditions
    conditions = detect_conditions(lab_results)

    # Score
    severity_score = compute_severity_score(lab_results)

    # Priorité
    priority = compute_priority(severity_score, conditions)

    # Injection dans JSON
    data["alerts"] = conditions
    data["severity_score"] = round(severity_score, 2)
    data["clinical_priority"] = priority

    return data