import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.nlp.hybrid_extractor import extract_hybrid_structured_data
from src.nlp.table_extractor import extract_from_text


def _key(row: Dict) -> Tuple[str, Optional[float], str]:
    # Normalize test name and unit; safely convert value to float when possible
    name = str(row.get("test_name") or "").strip().lower()
    raw_value = row.get("value")
    try:
        value = float(raw_value) if raw_value is not None and raw_value != "" else None
    except Exception:
        value = None
    unit = str(row.get("unit") or "").strip().lower()
    return (name, value, unit)


def _map_results(rows: List[Dict]) -> Dict[Tuple[str, float, str], Dict]:
    out = {}
    for row in rows:
        out[_key(row)] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare regex-only extraction vs hybrid (regex + spaCy fallback).")
    parser.add_argument("--ocr-text", required=True, help="Path to OCR text file")
    parser.add_argument("--out", default=None, help="Optional path to write diff JSON")
    args = parser.parse_args()

    text = Path(args.ocr_text).read_text(encoding="utf-8")

    regex_data = extract_from_text(text)
    hybrid_data = extract_hybrid_structured_data(text)

    regex_map = _map_results(regex_data.get("lab_results", []))
    hybrid_map = _map_results(hybrid_data.get("lab_results", []))

    regex_keys = set(regex_map.keys())
    hybrid_keys = set(hybrid_map.keys())

    added = [hybrid_map[k] for k in sorted(hybrid_keys - regex_keys)]
    removed = [regex_map[k] for k in sorted(regex_keys - hybrid_keys)]

    upgraded = []
    by_name_value_regex = {}
    by_name_value_hybrid = {}

    for r in regex_data.get("lab_results", []):
        by_name_value_regex[(str(r.get("test_name") or "").strip().lower(), r.get("value"))] = r
    for r in hybrid_data.get("lab_results", []):
        by_name_value_hybrid[(str(r.get("test_name") or "").strip().lower(), r.get("value"))] = r

    for kv, r in by_name_value_regex.items():
        h = by_name_value_hybrid.get(kv)
        if not h:
            continue
        if (not r.get("unit") and h.get("unit")) or (not r.get("reference_range") and h.get("reference_range")):
            upgraded.append({"before": r, "after": h})

    summary = {
        "regex_count": len(regex_data.get("lab_results", [])),
        "hybrid_count": len(hybrid_data.get("lab_results", [])),
        "added_by_hybrid": added,
        "removed_vs_regex": removed,
        "upgraded_rows": upgraded,
    }

    print("[INFO] regex_count:", summary["regex_count"])
    print("[INFO] hybrid_count:", summary["hybrid_count"])
    print("[INFO] added_by_hybrid:", len(added))
    print("[INFO] removed_vs_regex:", len(removed))
    print("[INFO] upgraded_rows:", len(upgraded))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] diff saved: {out}")


if __name__ == "__main__":
    main()
