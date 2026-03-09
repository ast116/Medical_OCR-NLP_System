import re
from typing import Optional

VALUE_PATTERN = re.compile(r"(?<!\w)(?:\d+[.,;]\d+|\d+)(?!\w)")
UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:g\s*/\s*dL|gm\s*/\s*dL|mg\s*/\s*dL|mg\s*/\s*L|"
    r"fL|pg|%|/\s*ul|/\s*µl|IU\s*/\s*L|mill\s*/\s*cmm|cells\s*/\s*cumm|cumm)(?![A-Za-z])",
    re.I,
)

IGNORE_LINE_PATTERNS = [
    r"^page\b",
    r"\breport\b",
    r"\bcomment\b",
    r"\bstatus\b",
    r"\bdiagnostic\b",
    r"\bhospital\b",
    r"\bvalidate\b",
    r"\bend of report\b",
    r"\binfants?\b",
    r"\bdevelopment\b",
    r"\blevels are typically\b",
    r"\bconcentralion above\b",
    r"\bbetween days\b",
]

LAB_HINT_PATTERN = re.compile(
    r"\b("
    r"ha?emoglobin|hemoglobin|bil[il1]*rubin|hct|pcv|rbc|wbc|mcv|mch|mchc|rdw|"
    r"neutrophils?|lymphocytes?|eosinophils?|monocytes?|platelet|"
    r"bilirubin|crp|protein|albumin|globulin|glucose|urea|creatinine|"
    r"sodium|potassium|chloride|ast|alt|alp|count|cell|differential|aec"
    r")\b",
    re.I,
)

UNIT_CANONICAL_MAP = {
    "g/dl": "g/dL",
    "gm/dl": "g/dL",
    "mg/dl": "mg/dL",
    "mg/l": "mg/L",
    "fl": "fL",
    "pg": "pg",
    "%": "%",
    "/ul": "/µL",
    "/µl": "/µL",
    "iu/l": "IU/L",
    "mill/cmm": "mill/cmm",
    "cells/cumm": "cells/cumm",
    "cumm": "cumm",
}


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_numeric_token(token: str) -> float:
    token = token.replace(",", ".").replace(";", ".")
    return float(token)


def _canonical_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    cleaned = re.sub(r"\s+", "", unit.lower())
    return UNIT_CANONICAL_MAP.get(cleaned, unit)


def _is_noise_line(line: str) -> bool:
    lower = line.lower()
    return any(re.search(pattern, lower) for pattern in IGNORE_LINE_PATTERNS)


def _extract_reference_range(line: str):
    range_match = re.search(r"(\d+(?:[.,]\d+)?)\s*[-~]\s*(\d+(?:[.,]\d+)?)", line)
    if range_match:
        return {
            "type": "interval",
            "low": _normalize_numeric_token(range_match.group(1)),
            "high": _normalize_numeric_token(range_match.group(2)),
        }

    threshold_match = re.search(r"([<>])\s*(\d+(?:[.,]\d+)?)", line)
    if threshold_match:
        return {
            "type": "threshold",
            "operator": threshold_match.group(1),
            "value": _normalize_numeric_token(threshold_match.group(2)),
        }

    return None


def _extract_flags(line: str) -> list[str]:
    return re.findall(r"\[(H|L)\]", line, flags=re.I)


def _extract_value_unit_pairs(line: str):
    values = list(VALUE_PATTERN.finditer(line))
    units = list(UNIT_PATTERN.finditer(line))

    if not values:
        return [], values, units

    pairs = []
    for value in values:
        closest_unit = None
        closest_distance = None
        for unit in units:
            distance = unit.start() - value.end()
            if 0 <= distance <= 8 and (closest_distance is None or distance < closest_distance):
                closest_distance = distance
                closest_unit = unit
        pairs.append((value, closest_unit))

    return pairs, values, units


def _find_best_measurement(line: str):
    pairs, values, units = _extract_value_unit_pairs(line)
    if not values:
        return None, None, None

    # Prefer value directly followed by a medical unit.
    for value_match, unit_match in pairs:
        if unit_match is not None:
            return value_match, unit_match, values

    # Fallback: first numeric value (for some analytes where unit is omitted in OCR).
    return values[0], None, values


def _extract_test_name(line: str, value_start: int) -> str:
    test_name = line[:value_start]
    test_name = re.sub(r"[:;,\-\s]+$", "", test_name)
    return _normalize_spaces(test_name)


def _is_likely_lab_line(line: str, test_name: str, unit_match, values) -> bool:
    if _is_noise_line(line):
        return False
    if not test_name or len(test_name) < 3:
        return False

    # Metadata/header lines often include ID and very long integers.
    if re.search(r"\b(patient|doctor|sample|client|collected|report|status)\b", line, re.I):
        return False

    has_lab_hint = bool(LAB_HINT_PATTERN.search(line))

    # A strong positive condition: unit + medical term.
    if unit_match is not None and has_lab_hint:
        return True

    # Fallback for lines where OCR misses unit, but analyte and range are present.
    if has_lab_hint and len(values) >= 2:
        return True

    return False

def extract_lab_line(line):
    line = _normalize_spaces(line)
    if not line:
        return None

    value_match, unit_match, values = _find_best_measurement(line)
    if not value_match:
        return None

    test_name = _extract_test_name(line, value_match.start())
    if not _is_likely_lab_line(line, test_name, unit_match, values):
        return None

    result = {
        "test_name": test_name,
        "value": _normalize_numeric_token(value_match.group()),
        "unit": _canonical_unit(unit_match.group()) if unit_match else None,
    }
    reference = _extract_reference_range(line)
    flags = _extract_flags(line)

    if reference:
        result["reference_range"] = reference
    if flags:
        result["flags"] = [flag.upper() for flag in flags]

    return result


def extract_metadata(text: str):
    lines = text.splitlines()
    joined = "\n".join(lines)

    metadata = {
        "patient_id": None,
        "sample_no": None,
        "doctor": None,
        "dates": [],
    }

    patient_match = re.search(r"\bpatient\s*id\b[^0-9]*([A-Za-z0-9\-]{5,})", joined, re.I)
    if patient_match:
        metadata["patient_id"] = patient_match.group(1)

    sample_match = re.search(
        r"\bsampl\w*\s*no\b[^A-Za-z0-9]*([A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)",
        joined,
        re.I,
    )
    if sample_match:
        metadata["sample_no"] = sample_match.group(1)

    for raw_line in lines:
        line = _normalize_spaces(raw_line)
        doctor_match = re.search(r"\bdr[:.\s]*([A-Za-z][A-Za-z\s\.-]{2,})", line, re.I)
        if doctor_match:
            doctor_name = doctor_match.group(1)
            doctor_name = re.split(
                r"\b(sampl\w*|patient|report|status|collected)\b",
                doctor_name,
                flags=re.I,
            )[0]
            metadata["doctor"] = _normalize_spaces(doctor_name)
            break

    date_pattern = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{8}|\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)\b",
        re.I,
    )
    metadata["dates"] = sorted(set(match.group(0) for match in date_pattern.finditer(joined)))

    return metadata

def extract_from_text(text):

    results = []

    lines = text.split("\n")

    for line in lines:

        parsed = extract_lab_line(line)

        if parsed:
            results.append(parsed)

    return {
        "metadata": extract_metadata(text),
        "lab_results": results,
    }
