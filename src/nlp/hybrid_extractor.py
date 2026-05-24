import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import BASE_DIR
from src.nlp.table_extractor import (
    IGNORE_LINE_PATTERNS,
    LAB_HINT_PATTERN,
    UNIT_CANONICAL_MAP,
    extract_from_text,
)

DEFAULT_SPACY_MODEL_DIR = Path(BASE_DIR) / "models" / "spacy_ner" / "model-best"
_NOISE_RE = re.compile("|".join(IGNORE_LINE_PATTERNS), re.IGNORECASE)
_ADDRESS_RE = re.compile(
    r"\b(hospital|diagnostic|sector|colony|road|market|near|gate|plot|town|state bank|ph\.?|phone|fax)\b",
    re.IGNORECASE,
)
_META_LINE_RE = re.compile(
    r"\b(patient|sample|report|verified|print|collection|doctor|consultant|pathologist|stationery)\b",
    re.IGNORECASE,
)
_DOCTOR_BAD_WORDS_RE = re.compile(
    r"\b(report|date|verified|sample|collection|accepted|on|pathologist|consultant)\b",
    re.IGNORECASE,
)
_DATE_STRICT_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)\b",
    re.IGNORECASE,
)
_TEST_BAD_WORDS_RE = re.compile(
    r"\b(hospital|diagnostic|sector|colony|road|market|near|gate|plot|town|state|bank|address|inside)\b",
    re.IGNORECASE,
)


def _normalize_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    compact = re.sub(r"\s+", "", unit).lower()
    return UNIT_CANONICAL_MAP.get(compact, unit)


def _parse_float(value_text: str) -> Optional[float]:
    cleaned = value_text.strip().replace(",", ".").replace(";", ".")
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_reference_range(range_text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not range_text:
        return None

    text = range_text.replace(",", ".").replace(";", ".")
    interval = re.search(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)", text)
    if interval:
        low = float(interval.group(1))
        high = float(interval.group(2))
        if low <= high:
            return {"type": "interval", "low": low, "high": high}

    threshold = re.search(r"([<>])\s*(\d+(?:\.\d+)?)", text)
    if threshold:
        return {
            "type": "threshold",
            "operator": threshold.group(1),
            "value": float(threshold.group(2)),
        }

    return None


def _normalize_test_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip(" :;,-|\t")).strip()


def _dedupe_lab_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in results:
        key = (
            (item.get("test_name") or "").lower(),
            item.get("value"),
            item.get("unit") or None,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _result_quality_score(row: Dict[str, Any]) -> int:
    score = 0
    if row.get("unit"):
        score += 3
    if row.get("reference_range"):
        score += 3
    if row.get("flags"):
        score += 1
    name = _normalize_test_name(str(row.get("test_name") or ""))
    if len(name) >= 4:
        score += 1
    return score


def _consolidate_similar_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge near-duplicate rows (same analyte + same numeric value) and keep the richest version.
    Example handled: one row with unit=None and another same row with unit present.
    """
    grouped: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (_normalize_test_name(str(row.get("test_name") or "")).lower(), row.get("value"))
        grouped.setdefault(key, []).append(row)

    output: List[Dict[str, Any]] = []
    for _, variants in grouped.items():
        best = max(variants, key=_result_quality_score)
        merged = dict(best)

        # Fill missing fields from weaker variants when useful.
        if not merged.get("unit"):
            for v in variants:
                if v.get("unit"):
                    merged["unit"] = v["unit"]
                    break
        if not merged.get("reference_range"):
            for v in variants:
                if v.get("reference_range"):
                    merged["reference_range"] = v["reference_range"]
                    break
        if not merged.get("flags"):
            for v in variants:
                if v.get("flags"):
                    merged["flags"] = v["flags"]
                    break

        output.append(merged)

    return output


def _is_noise_line(line: str) -> bool:
    lower = line.lower()
    if _NOISE_RE.search(lower):
        return True
    if _ADDRESS_RE.search(lower):
        return True
    return False


def _is_valid_test_name(test_name: str) -> bool:
    normalized = _normalize_test_name(test_name)
    if len(normalized) < 2:
        return False
    if not re.search(r"[A-Za-z]", normalized):
        return False
    if _TEST_BAD_WORDS_RE.search(normalized):
        return False
    return True


def _is_valid_lab_candidate(line: str, test_name: str, unit: Optional[str], has_ref: bool) -> bool:
    if _is_noise_line(line):
        return False

    if not _is_valid_test_name(test_name):
        return False

    has_lab_hint = bool(LAB_HINT_PATTERN.search(line))
    has_meta_tokens = bool(_META_LINE_RE.search(line))

    # Strict fallback policy:
    # - Accept lines with clear lab analyte hints and either unit/ref range.
    # - For no-unit OCR lines, keep only if analyte hint + reference range present.
    if has_lab_hint and (unit is not None or has_ref):
        return True
    if has_lab_hint and unit is None and has_ref:
        return True

    # Avoid metadata/header/footer pollution.
    if has_meta_tokens:
        return False

    # As last-resort, keep only if unit exists and test name looks clean.
    return unit is not None


def _load_spacy_model(model_dir: Path):
    try:
        import spacy
    except ImportError:
        return None

    if not model_dir.exists():
        return None

    try:
        return spacy.load(model_dir)
    except Exception:
        return None


def _extract_spacy_metadata(doc) -> Dict[str, Any]:
    metadata = {
        "patient_id": None,
        "sample_no": None,
        "doctor": None,
        "dates": [],
    }

    dates = []
    for ent in doc.ents:
        if ent.label_ == "PATIENT_ID" and metadata["patient_id"] is None:
            metadata["patient_id"] = ent.text.strip()
        elif ent.label_ == "SAMPLE_NO" and metadata["sample_no"] is None:
            metadata["sample_no"] = ent.text.strip()
        elif ent.label_ == "DOCTOR" and metadata["doctor"] is None:
            doctor_name = ent.text.strip()
            if doctor_name and not _DOCTOR_BAD_WORDS_RE.search(doctor_name):
                metadata["doctor"] = doctor_name
        elif ent.label_ == "DATE_TIME":
            value = ent.text.strip()
            if value and _DATE_STRICT_RE.search(value):
                dates.append(value)

    metadata["dates"] = sorted(set(dates))
    return metadata


def _find_nearest(entities, origin, label, before=False):
    candidates = [e for e in entities if e.label_ == label]
    if not candidates:
        return None

    best = None
    best_distance = None
    for ent in candidates:
        if before and ent.end_char > origin.start_char:
            continue
        if not before and ent.start_char < origin.end_char:
            continue

        distance = abs(origin.start_char - ent.end_char) if before else abs(ent.start_char - origin.end_char)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = ent

    return best


def _extract_spacy_measurements(text: str, nlp) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_noise_line(line):
            continue

        doc = nlp(line)
        ents = list(doc.ents)
        value_ents = [e for e in ents if e.label_ == "VALUE"]

        for value_ent in value_ents:
            value = _parse_float(value_ent.text)
            if value is None:
                continue

            test_ent = _find_nearest(ents, value_ent, "TEST_NAME", before=True)
            if test_ent is None:
                continue

            test_name = _normalize_test_name(test_ent.text)
            if len(test_name) < 2:
                continue

            unit_ent = _find_nearest(ents, value_ent, "UNIT", before=False)
            ref_ent = _find_nearest(ents, value_ent, "REFERENCE_RANGE", before=False)
            unit_value = _normalize_unit(unit_ent.text.strip()) if unit_ent is not None else None
            has_ref = ref_ent is not None

            if not _is_valid_lab_candidate(line, test_name, unit_value, has_ref):
                continue

            item = {
                "test_name": test_name,
                "value": value,
                "unit": unit_value,
            }

            reference = _parse_reference_range(ref_ent.text.strip()) if ref_ent is not None else None
            if reference:
                item["reference_range"] = reference

            rows.append(item)

    return _dedupe_lab_results(rows)


def _merge_metadata(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        "patient_id": primary.get("patient_id") or fallback.get("patient_id"),
        "sample_no": primary.get("sample_no") or fallback.get("sample_no"),
        "doctor": primary.get("doctor") or fallback.get("doctor"),
        "dates": sorted(set((primary.get("dates") or []) + (fallback.get("dates") or []))),
    }
    return merged


def _merge_lab_results(primary_results: List[Dict[str, Any]], fallback_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = list(primary_results)

    existing = {
        ((r.get("test_name") or "").lower(), r.get("value"), r.get("unit") or None)
        for r in primary_results
    }

    for row in fallback_results:
        key = ((row.get("test_name") or "").lower(), row.get("value"), row.get("unit") or None)
        if key in existing:
            continue
        existing.add(key)
        merged.append(row)

    return merged


def _is_plausible_result(row: Dict[str, Any]) -> bool:
    test_name = _normalize_test_name(str(row.get("test_name") or ""))
    if not _is_valid_test_name(test_name):
        return False

    has_lab_hint = bool(LAB_HINT_PATTERN.search(test_name))
    unit = row.get("unit")
    has_ref = bool(row.get("reference_range"))
    value = row.get("value")

    # Reject obvious context rows (address/header) that slipped through.
    if _ADDRESS_RE.search(test_name) or _META_LINE_RE.search(test_name):
        return False

    # Conservative rule:
    # Without unit and without reference range, keep only explicit lab-like analytes.
    if unit is None and not has_ref and not has_lab_hint:
        return False

    # Extremely large unitted-less values are usually IDs/postal codes.
    if unit is None and has_lab_hint is False:
        if isinstance(value, (int, float)) and value >= 1000:
            return False

    return True


def _sanitize_lab_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if _is_plausible_result(row)]
    consolidated = _consolidate_similar_results(filtered)
    return _dedupe_lab_results(consolidated)


def extract_hybrid_structured_data(
    text: str,
    spacy_model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    # 1) Primary extraction (existing regex/table pipeline)
    structured = extract_from_text(text)
    structured["lab_results"] = _sanitize_lab_results(structured.get("lab_results", []))

    # 2) spaCy fallback (optional)
    model_path = Path(spacy_model_dir) if spacy_model_dir else DEFAULT_SPACY_MODEL_DIR
    nlp = _load_spacy_model(model_path)
    if nlp is None:
        return structured

    doc = nlp(text)
    spacy_metadata = _extract_spacy_metadata(doc)
    spacy_results = _extract_spacy_measurements(text, nlp)

    structured["metadata"] = _merge_metadata(structured.get("metadata", {}), spacy_metadata)
    merged = _merge_lab_results(structured.get("lab_results", []), spacy_results)
    structured["lab_results"] = _sanitize_lab_results(merged)

    return structured
