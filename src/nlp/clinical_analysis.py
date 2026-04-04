def _name_matches(name, keywords):
    name = name.lower()
    return any(keyword in name for keyword in keywords)


def detect_conditions(lab_results):
    conditions = []

    flags = {
        "infection": False,
        "anemia": False,
        "polycythemia": False,
        "thrombocytopenia": False,
        "thrombocytosis": False,
        "renal": False,
        "liver": False,
        "hyperglycemia": False,
        "hypoglycemia": False,
        "electrolyte": False,
        "hypoalbuminemia": False,
        "possible_malaria": False,
    }

    for entry in lab_results:
        name = entry.get("test_name", "").lower()
        status = entry.get("status")
        value = entry.get("value")

        if status not in ["high", "low", "normal", "unknown"]:
            continue

        # Infection / inflammation
        if _name_matches(name, ["wbc", "leukocyte", "leucocyte", "tlc", "neutrophil", "crp"]):
            if status == "high":
                flags["infection"] = True

        # Anemia / polycythemia
        if _name_matches(name, ["hemoglobin", "haemoglobin", "hb", "hgb", "hct", "hematocrit", "haematocrit", "pcv", "rbc"]):
            if status == "low":
                flags["anemia"] = True
            if status == "high" and _name_matches(name, ["hemoglobin", "haemoglobin", "hb", "hgb", "hct", "hematocrit", "haematocrit"]):
                flags["polycythemia"] = True

        # Platelets
        if _name_matches(name, ["platelet", "plt"]):
            if status == "low":
                flags["thrombocytopenia"] = True
            if status == "high":
                flags["thrombocytosis"] = True

        # Renal function
        if _name_matches(name, ["creatinine", "urea", "bun", "egfr"]):
            if status == "high":
                flags["renal"] = True

        # Liver function
        if _name_matches(name, ["bilirubin", "sgot", "ast", "sgpt", "alt", "alp", "alkaline phosphatase"]):
            if status == "high":
                flags["liver"] = True

        # Glucose / diabetes
        if _name_matches(name, ["glucose", "blood sugar", "glycemia", "random glucose", "fasting glucose"]):
            if status == "high":
                flags["hyperglycemia"] = True
            if status == "low":
                flags["hypoglycemia"] = True

        # Electrolytes
        if _name_matches(name, ["sodium", "na+", "potassium", "k+", "chloride", "cl-"]):
            if status in ["high", "low"]:
                flags["electrolyte"] = True

        # Proteins
        if _name_matches(name, ["albumin"]):
            if status == "low":
                flags["hypoalbuminemia"] = True

        # Malaria / goutte épaisse (heuristique)
        if _name_matches(name, ["goutte", "malaria", "plasmodium", "parasite", "pf", "pan"]):
            if status == "high" or (isinstance(value, (int, float)) and value > 0):
                flags["possible_malaria"] = True

    if flags["infection"]:
        conditions.append("possible infection or inflammation")
    if flags["anemia"]:
        conditions.append("possible anemia")
    if flags["polycythemia"]:
        conditions.append("possible polycythemia")
    if flags["thrombocytopenia"]:
        conditions.append("possible thrombocytopenia")
    if flags["thrombocytosis"]:
        conditions.append("possible thrombocytosis")
    if flags["renal"]:
        conditions.append("possible renal impairment")
    if flags["liver"]:
        conditions.append("possible liver dysfunction")
    if flags["hyperglycemia"]:
        conditions.append("possible hyperglycemia / diabetes")
    if flags["hypoglycemia"]:
        conditions.append("possible hypoglycemia")
    if flags["electrolyte"]:
        conditions.append("possible electrolyte imbalance")
    if flags["hypoalbuminemia"]:
        conditions.append("possible low albumin")
    if flags["possible_malaria"]:
        conditions.append("possible malaria infection")

    return conditions


# Score de gravité
def compute_severity_score(lab_results):
    abnormal_count = 0
    total = len(lab_results)

    weight = 0

    for entry in lab_results:
        status = entry.get("status")
        name = entry.get("test_name", "").lower()

        if status in ["high", "low"]:
            abnormal_count += 1

            # pondération simple
            if status == "high":
                weight += 1.2
            else:
                weight += 1

            # pondération des tests plus critiques
            if any(term in name for term in ["potassium", "sodium", "creatinine", "bilirubin", "glucose"]):
                weight += 0.3

    if total == 0:
        return 0

    score = (weight / total)

    # normalisation (max ~2 → ramené à 1)
    return min(score / 2, 1.0)


# Priorité clinique
def compute_priority(severity_score, conditions):

    if severity_score > 0.7:
        return "urgent"

    if "possible infection or inflammation" in conditions and severity_score > 0.5:
        return "urgent"

    if ("possible renal impairment" in conditions or "possible electrolyte imbalance" in conditions) and severity_score > 0.6:
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
