# 🏥 Medical OCR & NLP System

> Pipeline OCR + NLP pour extraire, structurer, interpréter et exposer via API des résultats de laboratoire médicaux à partir d'images ou de PDF.

---

## ✨ Vue d'ensemble

Ce projet transforme des comptes rendus de laboratoire scannés en données JSON exploitables.

Le système combine:
- 🖼️ **OCR EasyOCR** pour lire les images/PDF
- 🧼 **Prétraitement image** pour améliorer la qualité OCR
- 🧠 **Extraction NLP hybride** avec regex + spaCy NER
- 🩺 **Interprétation clinique** des résultats biologiques
- 🌍 **Traduction anglais vers français**
- 🚀 **API FastAPI** pour l'intégration avec un frontend ou un système RAG
- 🐳 **Docker GPU** pour exécuter le projet sur une machine NVIDIA

---

## 📑 Sommaire

- [🎯 Fonctionnalités](#-fonctionnalités)
- [🏗️ Architecture globale](#️-architecture-globale)
- [📁 Structure du projet](#-structure-du-projet)
- [⚙️ Installation locale](#️-installation-locale)
- [🚀 Utilisation de la pipeline](#-utilisation-de-la-pipeline)
- [🌐 API FastAPI](#-api-fastapi)
- [🐳 Docker GPU](#-docker-gpu)
- [🧠 NLP hybride regex + spaCy](#-nlp-hybride-regex--spacy)
- [🎓 Réentraînement spaCy](#-réentraînement-spacy)
- [📋 Format JSON](#-format-json)
- [📊 Évaluation et comparaison](#-évaluation-et-comparaison)
- [⚠️ Limites connues](#️-limites-connues)

---

## 🎯 Fonctionnalités

| Module | Rôle |
|---|---|
| 🖼️ Prétraitement image | Resize, débruitage, deskew, amélioration contraste/netteté, binarisation, morphologie |
| 🔤 OCR EasyOCR | Extraction du texte avec bounding boxes |
| 📐 Reconstruction lignes | Tri spatial, regroupement des boxes, reconstruction du texte ligne par ligne |
| 🧼 Post-traitement OCR | Correction unités, décimales, plages de référence et termes médicaux fréquents |
| 🧪 Extraction regex | Extraction fiable des lignes de laboratoire connues |
| 🧠 spaCy NER | Fallback NLP pour compléter les résultats non captés par regex |
| 🔀 Fusion hybride | Regex prioritaire + spaCy fallback + filtres anti-bruit |
| 🩺 Interprétation | Statut normal/bas/élevé/inconnu et interprétation clinique |
| 🌍 Traduction FR | Traduction des interprétations et résumés vers le français |
| 📊 Confiance | Score OCR, score extraction et score global pipeline |
| 🚀 API | Endpoint FastAPI `/process` pour envoyer un ou plusieurs fichiers |
| 🐳 Docker GPU | Image prête pour machine NVIDIA, par exemple RTX 3090 |

---

## 🏗️ Architecture globale

```text
Image/PDF médical
      ↓
Prétraitement image
      ↓
EasyOCR + bounding boxes
      ↓
Reconstruction du texte OCR
      ↓
Post-traitement médical
      ↓
Extraction regex
      ↓
spaCy NER fallback
      ↓
Fusion hybride + filtres anti-bruit
      ↓
Interprétation clinique + résumé
      ↓
Traduction FR
      ↓
JSON structuré / API FastAPI / RAG
```

---

## 📁 Structure du projet

```text
medical_ocr&nlp_system/
├── data/
│   ├── raw/                 # Images/PDF à traiter
│   ├── ocr_output/          # Textes OCR générés
│   ├── structured/          # JSON structurés finaux
│   ├── ner_output/          # Sorties NER séparées
│   ├── ner_gold/            # Données d'annotation spaCy
│   └── gt/                  # Ground truth OCR pour évaluation
├── docs/
│   └── COLAB_SPACY_NER_TRAINING.md
├── models/
│   └── spacy_ner/           # Modèle spaCy entraîné
├── notebooks/
│   └── ocr_debug_pipeline.ipynb
├── src/
│   ├── api/                 # FastAPI
│   ├── config/              # Paramètres globaux
│   ├── evaluation/          # Scores et métriques
│   ├── nlp/                 # Regex, spaCy hybride, interprétation, traduction
│   ├── ocr/                 # EasyOCR, boxes, reconstruction
│   ├── postprocessing/      # Nettoyage texte OCR
│   ├── preprocessing/       # Traitements image
│   └── utils/               # Export JSON/Excel/PostgreSQL
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-docker-gpu.txt
```

---

## ⚙️ Installation locale

### Prérequis

- Python 3.12 recommandé
- `pip`
- Poppler si traitement PDF via `pdf2image`
- GPU optionnel en local

### Installation CPU locale

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Le mode local CPU utilise:

```python
USE_GPU=false
```

dans l'environnement, ou la valeur par défaut de [settings.py](src/config/settings.py).

---

## 🚀 Utilisation de la pipeline

Place les images ou PDF dans:

```text
data/raw/
```

Puis lance:

```bash
python -m src.main_ocr_pipeline
```

Sorties générées:

```text
data/ocr_output/*.txt       # texte OCR nettoyé
data/structured/*.json      # JSON structuré enrichi
```

---

## 🌐 API FastAPI

Lancer l'API:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Documentation interactive:

```text
http://localhost:8000/docs
```

Endpoint principal:

```text
POST /process
```

Avec Postman:
- Body: `form-data`
- Key: `files`
- Type: `File`
- Ajouter plusieurs lignes `files` pour plusieurs fichiers

Exemple `curl`:

```bash
curl -X POST "http://localhost:8000/process" \
  -F "files=@data/raw/test1.png" \
  -F "files=@data/raw/test2.png"
```

---

## 🐳 Docker GPU

Le projet contient une configuration Docker pour machine NVIDIA.

### Prérequis machine GPU

- Driver NVIDIA installé
- Docker installé
- Docker Compose installé
- NVIDIA Container Toolkit installé

Vérifier le GPU:

```bash
nvidia-smi
```

Tester Docker + GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Construire et lancer:

```bash
docker compose build
docker compose up
```

Tester PyTorch CUDA dans le conteneur:

```bash
docker compose run --rm medical-ocr-api python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Variables importantes:

| Variable | Rôle | Valeur Docker |
|---|---|---|
| `USE_GPU` | Active EasyOCR GPU | `true` |
| `ENABLE_POSTGRES_EXPORT` | Active l'export PostgreSQL | `false` par défaut |
| `HF_HOME` | Cache Hugging Face | `/app/.cache/huggingface` |
| `EASYOCR_MODULE_PATH` | Cache EasyOCR | `/app/.cache/easyocr` |

---

## 🧠 NLP hybride regex + spaCy

Le module principal est:

```text
src/nlp/hybrid_extractor.py
```

Stratégie:

```text
1. Extraction regex/table_extractor
2. Chargement du modèle spaCy si disponible
3. Extraction NER sur le texte OCR
4. Fusion regex + spaCy
5. Suppression des faux positifs
6. Consolidation des doublons
```

Le modèle spaCy attendu est:

```text
models/spacy_ner/model-best
```

Si spaCy ou le modèle n'est pas disponible, le système continue avec regex uniquement.

---

## 🎓 Réentraînement spaCy

Les annotations se trouvent dans:

```text
data/ner_gold/annotated.jsonl
```

Format d'une ligne:

```json
{"text":"Hb 10.2 g/dL 12-15","entities":[{"start":0,"end":2,"label":"TEST_NAME"},{"start":3,"end":7,"label":"VALUE"},{"start":8,"end":12,"label":"UNIT"},{"start":13,"end":18,"label":"REFERENCE_RANGE"}],"source_file":"test.png","line_number":1}
```

Labels utilisés:

```text
TEST_NAME
VALUE
UNIT
REFERENCE_RANGE
FLAG
PATIENT_ID
DATE_TIME
DOCTOR
SAMPLE_NO
```

Guide Colab:

```text
docs/COLAB_SPACY_NER_TRAINING.md
```

---

## 📋 Format JSON

Exemple simplifié d'une sortie réelle:

```json
{
  "metadata": {
    "patient_id": null,
    "sample_no": "7",
    "doctor": null,
    "dates": []
  },
  "lab_results": [
    {
      "test_name": "S.Creatinine",
      "value": 0.8,
      "unit": "mg/dL",
      "reference_range": {
        "type": "interval",
        "low": 0.2,
        "high": 1.0
      },
      "status": "normal",
      "interpretation": "Within normal range.",
      "interpretation_fr": "Dans les limites normales."
    },
    {
      "test_name": "Alk phosphatase",
      "value": 205.0,
      "unit": "U/L",
      "reference_range": {
        "type": "interval",
        "low": 30.0,
        "high": 120.0
      },
      "status": "high",
      "interpretation": "The alk phosphatase value is 205.0 U/L.",
      "interpretation_fr": "La valeur alk phosphatase est de 205,0 U/L."
    }
  ],
  "summary": "A blotchy globulin.",
  "alerts": [],
  "severity_score": 0.11,
  "clinical_priority": "normal",
  "ocr_confidence": 0.72,
  "extraction_confidence": 0.98,
  "pipeline_confidence": 0.89,
  "summary_fr": "A blotchy globulin."
}
```

---

## 📊 Évaluation et comparaison

Comparer regex seul vs hybride:

```bash
python -m src.nlp.compare_regex_hybrid \
  --ocr-text data/ocr_output/test7.png.txt \
  --out /tmp/test7_diff.json
```

Sortie:

```text
regex_count       # nombre de résultats extraits par regex seul
hybrid_count      # nombre de résultats après fusion regex + spaCy
added_by_hybrid   # lignes ajoutées par spaCy
removed_vs_regex  # lignes regex supprimées par les filtres
upgraded_rows     # lignes complétées/améliorées
```

Évaluation OCR:

```bash
python -m src.evaluate
```

ou via les modules:

```text
src/evaluation/ocr_metrics.py
src/evaluation/medical_metrics.py
src/evaluation/confidence.py
```

---

## ⚠️ Limites connues

- La qualité OCR dépend fortement du scan d'origine.
- Les tableaux très déformés peuvent rester difficiles à reconstruire.
- spaCy améliore certains cas, mais le regex reste prioritaire pour limiter les faux positifs.
- Les interprétations générées doivent rester une aide à la lecture, pas un diagnostic médical.
- Les données anonymisées peuvent réduire la performance sur `DOCTOR`, `PATIENT_ID` ou `DATE_TIME`.
- PostgreSQL est optionnel dans Docker et désactivé par défaut.

---

## 🧭 Roadmap

- 🧪 Ajouter plus d'annotations spaCy pour améliorer `UNIT` et `REFERENCE_RANGE`
- 🧾 Export FHIR et Excel plus complet
- 🖥️ Interface frontend avec correction manuelle
- 🔎 Intégration RAG pour interrogation documentaire
- 📈 Tableau de bord de métriques OCR/NLP

---

## 📌 Statut

Projet en développement académique/prototypage avancé.

Objectif principal: fournir une brique OCR + NLP médicale capable d'alimenter une API, une interface de correction et un système RAG.
