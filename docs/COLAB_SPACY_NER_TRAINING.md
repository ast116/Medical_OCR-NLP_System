# Entraîner un modèle spaCy NER sur Google Colab (GPU) puis l'importer dans ce projet

Ce guide est fait pour ton architecture actuelle OCR -> texte (`data/ocr_output/*.txt`) -> NLP.

Réponse courte à ta question:
- Oui, pour spaCy NER, l'entraînement se fait sur du **texte + annotations d'entités (start/end/label)**.
- Le JSON structuré peut aider à pré-annoter (bootstrap), mais le vrai train spaCy attend des spans sur texte.

---

## Labels NER recommandés (cohérents avec ton projet)
`TEST_NAME`, `VALUE`, `UNIT`, `REFERENCE_RANGE`, `FLAG`, `PATIENT_ID`, `DATE_TIME`, `DOCTOR`, `SAMPLE_NO`

---

## Préparation avant Colab (sur ta machine locale)
1. Lance ton pipeline pour générer les OCR txt:
```bash
python -m src.main_ocr_pipeline
```
2. Copie dans Google Drive ton dossier projet (ou au minimum `data/ocr_output` et `data/ner_gold`).
3. Si tu as déjà un fichier annoté, place-le dans:
`data/ner_gold/annotated.jsonl`

Format attendu dans `annotated.jsonl` (une ligne JSON par exemple):
```json
{"text":"Hb 11.8 g/dL 12-15","entities":[{"start":0,"end":2,"label":"TEST_NAME"},{"start":3,"end":7,"label":"VALUE"},{"start":8,"end":12,"label":"UNIT"},{"start":13,"end":18,"label":"REFERENCE_RANGE"}],"source_file":"test10.png","line_number":12}
```

---

## Cellules Colab (copier-coller dans l'ordre)

### Cellule 1 - Vérifier GPU
```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

### Cellule 2 - Monter Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cellule 3 - Définir chemins
```python
from pathlib import Path

# Adapte ce chemin à ton dossier dans Drive
PROJECT_DIR = Path('/content/drive/MyDrive/medical_ocr_nlp_system')
DATA_DIR = PROJECT_DIR / 'data'
OCR_DIR = DATA_DIR / 'ocr_output'
NER_GOLD_DIR = DATA_DIR / 'ner_gold'

NER_GOLD_DIR.mkdir(parents=True, exist_ok=True)
print('PROJECT_DIR =', PROJECT_DIR)
print('OCR_DIR exists =', OCR_DIR.exists())
print('NER_GOLD_DIR =', NER_GOLD_DIR)
```

### Cellule 4 - Installer spaCy + dépendances
```python
!pip -q install -U pip
!pip -q install "spacy[cuda12x]>=3.7,<4.0" tqdm
!python -m spacy validate
```

Si erreur CUDA 12x, remplace par:
```python
# !pip -q install "spacy[cuda11x]>=3.7,<4.0" tqdm
```

### Cellule 5 - (Optionnel) Générer un bootstrap faible depuis OCR txt
Utilise cette cellule seulement si tu n'as pas encore `annotated.jsonl`.
```python
import json, re
from pathlib import Path

input_dir = OCR_DIR
output_path = NER_GOLD_DIR / 'bootstrap_weak.jsonl'

value_re = re.compile(r"(?<!\w)(?:\d+[.,]\d+|\d+)(?!\w)")
unit_re = re.compile(r"(?:g/dL|mg/dL|mg/L|mmol/L|fL|pg|%|IU/L|U/L|x10\^?3/µ?L|/µL|/uL)", re.I)
range_re = re.compile(r"\d+(?:[.,]\d+)?\s*[-~]\s*\d+(?:[.,]\d+)?")


def annotate_line(line: str):
    ents = []
    vm = value_re.search(line)
    if not vm:
        return ents

    if vm.start() > 1:
        test = line[:vm.start()].strip(' :;,-\t')
        if len(test) >= 2 and re.search(r"[A-Za-z]", test):
            start = line.find(test)
            ents.append({"start": start, "end": start+len(test), "label": "TEST_NAME"})

    ents.append({"start": vm.start(), "end": vm.end(), "label": "VALUE"})

    um = unit_re.search(line[vm.end():])
    if um:
        us = vm.end() + um.start()
        ue = vm.end() + um.end()
        ents.append({"start": us, "end": ue, "label": "UNIT"})

    rm = range_re.search(line)
    if rm:
        ents.append({"start": rm.start(), "end": rm.end(), "label": "REFERENCE_RANGE"})

    ents = sorted(ents, key=lambda x: (x['start'], x['end']))
    clean = []
    last_end = -1
    for e in ents:
        if e['start'] >= last_end and e['end'] > e['start']:
            clean.append(e)
            last_end = e['end']
    return clean

count = 0
with output_path.open('w', encoding='utf-8') as out:
    for txt_file in sorted(input_dir.glob('*.txt')):
        with txt_file.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                ents = annotate_line(line)
                if not ents:
                    continue
                rec = {
                    "text": line,
                    "entities": ents,
                    "source_file": txt_file.stem,
                    "line_number": i
                }
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                count += 1

print('bootstrap records =', count)
print('saved ->', output_path)
```

### Cellule 6 - Choisir le fichier d'annotation
Si tu as déjà un gold set manuel, garde `annotated.jsonl`.
```python
from pathlib import Path

annotated_path = NER_GOLD_DIR / 'annotated.jsonl'
bootstrap_path = NER_GOLD_DIR / 'bootstrap_weak.jsonl'

if annotated_path.exists():
    SOURCE_JSONL = annotated_path
    print('Using annotated.jsonl')
elif bootstrap_path.exists():
    SOURCE_JSONL = bootstrap_path
    print('Using bootstrap_weak.jsonl (a corriger manuellement conseillé)')
else:
    raise FileNotFoundError('Ni annotated.jsonl ni bootstrap_weak.jsonl trouvés.')

print('SOURCE_JSONL =', SOURCE_JSONL)
```

### Cellule 7 - Valider, split train/dev, convertir en `.spacy`
```python
import json, random
from collections import Counter, defaultdict
from pathlib import Path
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

ALLOWED_LABELS = {
    "TEST_NAME", "VALUE", "UNIT", "REFERENCE_RANGE", "FLAG",
    "PATIENT_ID", "DATE_TIME", "DOCTOR", "SAMPLE_NO"
}

train_jsonl = NER_GOLD_DIR / 'train.jsonl'
dev_jsonl = NER_GOLD_DIR / 'dev.jsonl'
train_spacy = NER_GOLD_DIR / 'train.spacy'
dev_spacy = NER_GOLD_DIR / 'dev.spacy'

records = []
with SOURCE_JSONL.open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

valid = []
dropped = 0
label_counter = Counter()
for r in records:
    text = r.get('text', '')
    ents = r.get('entities', [])
    if not text or not isinstance(ents, list):
        dropped += 1
        continue

    cleaned = []
    for e in ents:
        s, en, lb = e.get('start'), e.get('end'), e.get('label')
        if not isinstance(s, int) or not isinstance(en, int):
            continue
        if s < 0 or en > len(text) or en <= s:
            continue
        if lb not in ALLOWED_LABELS:
            continue
        if text[s:en].strip() == '':
            continue
        cleaned.append({'start': s, 'end': en, 'label': lb})

    cleaned = sorted(cleaned, key=lambda x: (x['start'], x['end']))
    non_overlap = []
    last_end = -1
    for e in cleaned:
        if e['start'] >= last_end:
            non_overlap.append(e)
            last_end = e['end']

    if not non_overlap:
        dropped += 1
        continue

    for e in non_overlap:
        label_counter[e['label']] += 1

    valid.append({
        'text': text,
        'entities': non_overlap,
        'source_file': r.get('source_file', 'unknown')
    })

# split par source_file pour éviter fuite train/dev
by_file = defaultdict(list)
for r in valid:
    by_file[r['source_file']].append(r)

files = list(by_file.keys())
random.seed(42)
random.shuffle(files)

if len(files) <= 1:
    train_records = valid
    dev_records = []
else:
    dev_count = max(1, round(0.2 * len(files)))
    dev_files = set(files[:dev_count])
    train_records = [r for f in files if f not in dev_files for r in by_file[f]]
    dev_records = [r for f in files if f in dev_files for r in by_file[f]]

with train_jsonl.open('w', encoding='utf-8') as f:
    for r in train_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

with dev_jsonl.open('w', encoding='utf-8') as f:
    for r in dev_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

nlp = spacy.blank('en')

def to_docbin(items, out_path):
    db = DocBin(store_user_data=True)
    dropped_spans = 0
    for r in items:
        doc = nlp.make_doc(r['text'])
        spans = []
        for e in r['entities']:
            sp = doc.char_span(e['start'], e['end'], label=e['label'], alignment_mode='contract')
            if sp is None:
                dropped_spans += 1
                continue
            spans.append(sp)
        doc.ents = filter_spans(spans)
        db.add(doc)
    db.to_disk(out_path)
    return dropped_spans

train_dropped = to_docbin(train_records, train_spacy)
dev_dropped = to_docbin(dev_records, dev_spacy)

print('raw records      =', len(records))
print('valid records    =', len(valid))
print('dropped records  =', dropped)
print('train records    =', len(train_records))
print('dev records      =', len(dev_records))
print('train drop spans =', train_dropped)
print('dev drop spans   =', dev_dropped)
print('label distribution:', dict(label_counter))
```

### Cellule 8 - Générer config spaCy NER
```python
CONFIG_PATH = Path('/content/config_spacy_ner.cfg')

!python -m spacy init config /content/config_spacy_ner.cfg \
  --lang en --pipeline tok2vec,ner --optimize efficiency --force

print('Config saved:', CONFIG_PATH)
```

### Cellule 9 - Vérifier les données (debug)
```python
!python -m spacy debug data /content/config_spacy_ner.cfg \
  --paths.train "$train_spacy" \
  --paths.dev "$dev_spacy"
```

### Cellule 10 - Entraîner sur GPU
```python
OUTPUT_DIR = Path('/content/spacy_ner_output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

!python -m spacy train /content/config_spacy_ner.cfg \
  --paths.train "$train_spacy" \
  --paths.dev "$dev_spacy" \
  --output "$OUTPUT_DIR" \
  --gpu-id 0
```

### Cellule 11 - Évaluer (precision / recall / F1 + per-label)
```python
EVAL_JSON = OUTPUT_DIR / 'eval_metrics.json'

!python -m spacy evaluate "$OUTPUT_DIR/model-best" "$dev_spacy" \
  --output "$EVAL_JSON"

import json
metrics = json.loads(EVAL_JSON.read_text(encoding='utf-8'))

print('=== Global NER Metrics ===')
print('ents_p (precision):', round(metrics.get('ents_p', 0), 4))
print('ents_r (recall)   :', round(metrics.get('ents_r', 0), 4))
print('ents_f (f1-score) :', round(metrics.get('ents_f', 0), 4))

print('\n=== Per Label Metrics ===')
for label, vals in metrics.get('ents_per_type', {}).items():
    p = round(vals.get('p', 0), 4)
    r = round(vals.get('r', 0), 4)
    f = round(vals.get('f', 0), 4)
    print(f'{label:16s} P={p} R={r} F1={f}')
```

### Cellule 12 - Tester rapidement le modèle
```python
import spacy
nlp = spacy.load(OUTPUT_DIR / 'model-best')

sample = "Hb 10.2 g/dL (12-15) WBC 13.4 x10^3/uL"
doc = nlp(sample)
for ent in doc.ents:
    print(ent.text, ent.label_, ent.start_char, ent.end_char)
```

### Cellule 13 - Exporter le modèle entraîné vers Drive
```python
EXPORT_DIR = PROJECT_DIR / 'models' / 'spacy_ner'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

!cp -r "$OUTPUT_DIR/model-best" "$EXPORT_DIR/model-best"
!cp -r "$OUTPUT_DIR/model-last" "$EXPORT_DIR/model-last"

print('Model exported to:', EXPORT_DIR)
```

---

## Retour dans ton projet local

Après téléchargement/synchronisation Drive -> machine locale:
- Mets le modèle dans `models/spacy_ner/model-best`
- Ensuite on pourra brancher un module d'inférence dans la pipeline pour comparer:
1. regex seul
2. spaCy seul
3. fusion regex + spaCy

---

## Ce qu'il faut éviter
- Entraîner uniquement sur annotations automatiques non relues.
- Mélanger des lignes du même document dans train/dev.
- Changer les labels en cours de route.
- Sur-interpréter les métriques si le dev set est trop petit.

## Seuil minimum conseillé avant intégration
- `ents_f >= 0.75` global (premier objectif)
- Per-label `TEST_NAME`, `VALUE`, `UNIT` au-dessus de `0.80` si possible

