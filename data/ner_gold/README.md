# NER Gold Set (Annotation Guide)

Goal: create a small annotated dataset to train or evaluate NER.

Format: JSONL (one JSON object per line).

Each record:
- "text": raw OCR line (or short paragraph).
- "entities": list of spans with "start", "end", "label".
- "notes" (optional): any ambiguity or OCR noise.

Entity labels (minimal set):
- TEST_NAME
- VALUE
- UNIT
- REFERENCE_RANGE
- FLAG
- PATIENT_ID
- DATE_TIME
- DOCTOR
- SAMPLE_NO

Example:
{
  "text": "Hb(Haemoglobin) 11.8 g/dL 12 - 15",
  "entities": [
    {"start": 0, "end": 15, "label": "TEST_NAME"},
    {"start": 16, "end": 20, "label": "VALUE"},
    {"start": 21, "end": 25, "label": "UNIT"},
    {"start": 26, "end": 33, "label": "REFERENCE_RANGE"}
  ]
}

Tips:
- Use the raw OCR text as-is.
- Keep spans exact (character indices).
- If unsure, add a note in "notes".
