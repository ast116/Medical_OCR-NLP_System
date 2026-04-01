import re
from typing import Any, Dict, List, Optional, Tuple

from .table_extractor import (
    IGNORE_LINE_PATTERNS,
    UNIT_CANONICAL_MAP,
    UNIT_PATTERN,
    VALUE_PATTERN,
)

_NOISE_RE = re.compile("|".join(IGNORE_LINE_PATTERNS), re.IGNORECASE)


def _normalize_number(token: str) -> float:
    token = token.replace(",", ".").replace(";", ".")
    return float(token)


def _normalize_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    cleaned = re.sub(r"\s+", "", unit.lower())
    return UNIT_CANONICAL_MAP.get(cleaned, unit)


def _extract_value_unit_pairs(line: str) -> List[Tuple[re.Match, re.Match]]:
    values = list(VALUE_PATTERN.finditer(line))
    units = list(UNIT_PATTERN.finditer(line))
    pairs: List[Tuple[re.Match, re.Match]] = []

    if not values or not units:
        return pairs

    for value in values:
        closest_unit = None
        closest_distance = None
        for unit in units:
            distance = unit.start() - value.end()
            if 0 <= distance <= 8 and (closest_distance is None or distance < closest_distance):
                closest_distance = distance
                closest_unit = unit
        if closest_unit is not None:
            pairs.append((value, closest_unit))

    return pairs


def _extract_measurements_from_line(line: str) -> List[Tuple[str, float, Optional[str]]]:
    pairs = _extract_value_unit_pairs(line)
    results: List[Tuple[str, float, Optional[str]]] = []
    for value_match, unit_match in pairs:
        test_name = line[: value_match.start()].strip()
        test_name = re.sub(r"[:;,\-\s]+$", "", test_name)
        test_name = re.sub(r"\s+", " ", test_name).strip()
        if not test_name:
            continue
        value = _normalize_number(value_match.group())
        unit = _normalize_unit(unit_match.group())
        results.append((test_name, value, unit))
    return results


def _format_reference_range(reference: Dict[str, Any]) -> str:
    if reference.get("type") == "interval":
        return f"{reference.get('low')}-{reference.get('high')}"
    if reference.get("type") == "threshold":
        return f"{reference.get('operator')}{reference.get('value')}"
    return ""


def _add_entity(entities: List[Dict[str, Any]], seen: set, entity: Dict[str, Any]) -> None:
    key = (
        entity.get("label"),
        entity.get("text"),
        entity.get("normalized"),
        entity.get("group_id"),
        entity.get("source"),
    )
    if key in seen:
        return
    seen.add(key)
    entities.append(entity)


def extract_ner_entities(text: str, structured_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    seen: set = set()

    lab_results = []
    if structured_data:
        metadata = structured_data.get("metadata", {})
        if metadata.get("patient_id"):
            _add_entity(
                entities,
                seen,
                {
                    "label": "PATIENT_ID",
                    "text": str(metadata["patient_id"]),
                    "group_id": "metadata",
                    "source": "structured",
                },
            )
        if metadata.get("sample_no"):
            _add_entity(
                entities,
                seen,
                {
                    "label": "SAMPLE_NO",
                    "text": str(metadata["sample_no"]),
                    "group_id": "metadata",
                    "source": "structured",
                },
            )
        if metadata.get("doctor"):
            _add_entity(
                entities,
                seen,
                {
                    "label": "DOCTOR",
                    "text": str(metadata["doctor"]),
                    "group_id": "metadata",
                    "source": "structured",
                },
            )
        for date_value in metadata.get("dates", []):
            _add_entity(
                entities,
                seen,
                {
                    "label": "DATE_TIME",
                    "text": str(date_value),
                    "group_id": "metadata",
                    "source": "structured",
                },
            )

        lab_results = structured_data.get("lab_results", [])
        for idx, result in enumerate(lab_results, start=1):
            group_id = f"lab_{idx}"
            test_name = result.get("test_name")
            if test_name:
                _add_entity(
                    entities,
                    seen,
                    {
                        "label": "TEST_NAME",
                        "text": str(test_name),
                        "group_id": group_id,
                        "source": "structured",
                    },
                )
            if result.get("value") is not None:
                _add_entity(
                    entities,
                    seen,
                    {
                        "label": "VALUE",
                        "text": str(result.get("value")),
                        "normalized": result.get("value"),
                        "group_id": group_id,
                        "source": "structured",
                    },
                )
            if result.get("unit"):
                _add_entity(
                    entities,
                    seen,
                    {
                        "label": "UNIT",
                        "text": str(result.get("unit")),
                        "group_id": group_id,
                        "source": "structured",
                    },
                )
            if result.get("reference_range"):
                _add_entity(
                    entities,
                    seen,
                    {
                        "label": "REFERENCE_RANGE",
                        "text": _format_reference_range(result["reference_range"]),
                        "group_id": group_id,
                        "source": "structured",
                    },
                )
            for flag in result.get("flags", []):
                _add_entity(
                    entities,
                    seen,
                    {
                        "label": "FLAG",
                        "text": str(flag),
                        "group_id": group_id,
                        "source": "structured",
                    },
                )

    existing_measurements = {
        (
            (item.get("test_name") or "").lower(),
            item.get("value"),
            str(item.get("unit")) if item.get("unit") is not None else None,
        )
        for item in lab_results
    }

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if _NOISE_RE.search(line.lower()):
            continue

        measurements = _extract_measurements_from_line(line)
        if not measurements:
            continue

        for idx, (test_name, value, unit) in enumerate(measurements, start=1):
            key = (test_name.lower(), value, unit)
            if key in existing_measurements:
                continue
            group_id = f"raw_{line_number}_{idx}"
            _add_entity(
                entities,
                seen,
                {
                    "label": "TEST_NAME",
                    "text": test_name,
                    "group_id": group_id,
                    "source": "raw_line",
                    "line_number": line_number,
                },
            )
            _add_entity(
                entities,
                seen,
                {
                    "label": "VALUE",
                    "text": str(value),
                    "normalized": value,
                    "group_id": group_id,
                    "source": "raw_line",
                    "line_number": line_number,
                },
            )
            if unit:
                _add_entity(
                    entities,
                    seen,
                    {
                        "label": "UNIT",
                        "text": unit,
                        "group_id": group_id,
                        "source": "raw_line",
                        "line_number": line_number,
                    },
                )

    return entities
