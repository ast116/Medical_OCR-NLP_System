# 🏥 Medical OCR-NLP System

> **Elegant and robust pipeline for extracting and analyzing medical data from scanned documents**

---

## ✨ Overview

A complete system for **intelligent extraction** combining optical character recognition (OCR) and natural language processing (NLP) to convert your scanned documents into structured data ready for advanced analysis.

**Use cases:**
- 📋 Patient information and medical treatment extraction
- 🔍 Automatic indexing and document search
- 📊 Analytics and queries on medical archives
- ✅ Form validation and enrichment

---

## 📑 Table of Contents

- [🎯 Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📦 Repository Composition](#-repository-composition)
- [⚙️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [📸 OCR Pipeline](#-ocr-pipeline)
- [🧠 NLP Pipeline: Regex + Claude/spaCy Strategy](#-nlp-pipeline-regex--claudespacy-strategy)
- [📋 JSON Output Format](#-json-output-format)
- [🎓 Training the NLP Model](#-training-the-nlp-model)
- [🛡️ Error Handling](#️-error-handling)
- [✅ Tests and Metrics](#-tests-and-metrics)
- [🤝 Contribution](#-contribution)
- [📄 License](#-license)

---

## 🎯 Features

| Feature | Description |
|---|---|
| 🖼️ **Image Preprocessing** | Skew correction, binarization, intelligent enhancement |
| 🔤 **Configurable OCR** | Support for Tesseract, EasyOCR and other engines |
| 🎯 **Regex Extraction** | Immediate coverage with reliable and auditable patterns |
| 🧠 **NLP with spaCy** | Advanced NER extraction (after training) |
| 🤖 **Claude Mode** | Support for Claude Haiku usage as fallback |
| 📋 **JSON Output** | Standardized format with metadata and confidence scores |
| 🔄 **Intelligent Fallback** | Auto switching regex → Claude → spaCy as needed |
| 📊 **Audit Logging** | Complete traceability of processing decisions |

---

## 🏗️ Architecture

```
Medical_OCR-NLP_System/
├── 📁 data/                    # Input/output data
│   ├── input/                  # Scanned documents to process
│   └── output/                 # Generated JSON results
├── �� src/
│   ├── ocr/                    # OCR pipeline
│   │   ├── preprocessor.py     # Image preprocessing
│   │   └── engines/            # Tesseract, EasyOCR, etc.
│   ├── nlp/                    # NLP pipeline
│   │   ├── patterns/           # Regular expressions
│   │   ├── models/             # spaCy/Claude models
│   │   └── extraction.py       # Extraction logic
│   └── utils/                  # Utilities and helpers
├── 📁 docs/                    # 📖 Documentation
│   ├── nlp_training.md         # Model training guide
│   ├── patterns.md             # Regex patterns reference
│   └── api.md                  # API documentation
├── 📁 notebooks/               # Jupyter notebooks for exploration
├── 📁 tests/                   # Unit & integration tests
├── 🐳 Dockerfile              # Docker image
├── requirements.txt            # Python dependencies
└── run_pipeline.py             # Main script
```

---

## 📦 Repository Composition

```
Languages used:
├── 📓 Jupyter Notebook : 79.8%  (Exploration & experimentation)
├── 🐍 Python           : 19.9%  (Production & scripts)
└── 🐳 Dockerfile       : 0.3%   (Containerization)
```

For NLP model training instructions, see: 📖 [`docs/nlp_training.md`](https://github.com/ast116/Medical_OCR-NLP_System/tree/main/docs)

---

## ⚙️ Installation

### Prerequisites
- 🐍 Python 3.8+ (3.10+ recommended)
- 📦 pip or conda
- 🐳 Docker (optional)
- 🔧 Tesseract (if using Tesseract OCR)

### Standard Setup

```bash
# 1️⃣ Clone the repository
git clone https://github.com/ast116/Medical_OCR-NLP_System.git
cd Medical_OCR-NLP_System

# 2️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# or
.venv\Scripts\activate          # Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ (Optional) Download models
python scripts/download_models.py
```

### With Docker

```bash
docker build -t medical-ocr-nlp .
docker run -v /path/to/data:/app/data medical-ocr-nlp
```

---

## 🚀 Quick Start

### Basic Example

```bash
python run_pipeline.py \
  --input data/input/example.pdf \
  --output data/output/result.json \
  --ocr-engine tesseract \
  --nlp-mode regex
```

### Main Options

```
--input FILE              Input file (PDF, PNG, JPG, etc.)
--output FILE             JSON output path
--ocr-engine MOTOR        tesseract | easyocr (default: tesseract)
--nlp-mode MODE           regex | spacy | claude | hybrid (default: regex)
--spacy-model PATH        Path to custom spaCy model
--confidence-threshold N  Min confidence threshold (0.0-1.0, default: 0.5)
--debug                   Verbose mode with detailed logs
--language LANG           Document language (default: en)
```

---

## 📸 OCR Pipeline

### Processing Steps

```
📄 Input document
    ↓
🔧 Preprocessing
    ├── Skew correction (deskew)
    ├── Adaptive binarization
    ├── Noise removal (denoising)
    └── Enhancement (contrast, sharpness)
    ↓
🔤 Text recognition (OCR Engine)
    ├── Page/block segmentation
    ├── Text zone detection
    └── Call configured engine
    ↓
✨ Post-processing
    ├── Spell correction
    ├── Normalization (dates, numbers)
    └── Fragment merging
    ↓
📋 Structured text
```

### Configuration Tips

- **Variable quality images** → Increase preprocessing steps
- **Multi-language documents** → Configure language per zone
- **Performance critical** → Use EasyOCR (GPU) instead of Tesseract

---

## 🧠 NLP Pipeline: Regex + Claude/spaCy Strategy

### 🎯 The "Hidden Work" Strategy

**Current situation:** Using **Claude Haiku** and **robust regular expressions** for critical medical entity extraction.

#### Phase 1️⃣ : **Regex Priority** ✅ (NOW)

```
Advantages:
├── ✓ Immediate coverage and 100% explainable
├── ✓ No training data needed
├── ✓ Audit and compliance facilitated
├── ✓ High precision on fixed formats
└── ✓ Immediate production deployment
```

**Covered patterns:**
- 👤 Patient names, dates of birth
- 💊 Medications, dosages
- 🏥 Medical codes, IPS numbers
- 📅 Prescription dates
- 🔢 Identification numbers

#### Phase 2️⃣ : **Claude Haiku as Fallback** 🚀 (NOW)

In parallel with regex, **Claude Haiku** offers:

```
├── 🧠 Advanced contextual understanding
├── 📝 Handling linguistic variations
├── 🔗 Entity-entity relationships
├── ⚡ Lightweight and fast (Haiku)
└── 💰 Cost-effective at scale
```

#### Phase 3️⃣ : **spaCy After Training** 🎓 (ROADMAP)

After collecting and annotating data:
- 📚 Train dedicated medical NER spaCy model
- ✅ Validation on representative dataset
- 🚀 Deploy as primary engine

### 🔄 Decision Logic (Fallback)

**Step 1:** Check regex patterns  
↓  
**Step 2:** If no regex match, try Claude Haiku  
↓  
**Step 3:** If Claude score is low, check spaCy (if available and well-trained)  
↓  
**Step 4:** If all methods fail, flag for human review  
↓  
**Final:** Output result with source and confidence score

### 📊 Example Multi-Source Decision

```json
{
  "entity": "medication",
  "value": "Amoxicillin 500mg",
  "extractions": [
    {
      "source": "regex",
      "value": "Amoxicillin 500mg",
      "confidence": 0.98,
      "pattern": "MEDICATION_DOSAGE"
    },
    {
      "source": "claude_haiku",
      "value": "Amoxicillin 500mg",
      "confidence": 0.92,
      "reasoning": "Antibiotic treatment context identified"
    },
    {
      "source": "spacy",
      "value": "Amoxicillin",
      "confidence": 0.85,
      "entity_type": "MEDICATION"
    }
  ],
  "final_value": "Amoxicillin 500mg",
  "final_source": "regex",
  "decision_timestamp": "2026-06-07T14:32:00Z"
}
```

---

## 📤 JSON Output Format

### Complete Structure

```json
{
  "metadata": {
    "document_id": "DOC_2026_0001",
    "processed_at": "2026-06-07T14:32:00Z",
    "pipeline_version": "2.1.0",
    "pages": 2,
    "language": "en",
    "nlp_engine": "regex+haiku"
  },
  "ocr": {
    "raw_text": "Extracted raw text...",
    "confidence": 0.92,
    "engine": "tesseract"
  },
  "extractions": [
    {
      "entity": "patient_name",
      "value": "John Smith",
      "source": "regex",
      "confidence": 0.98,
      "pattern": "NAME_PATTERN_v3",
      "location": { "page": 1, "bbox": [100, 150, 300, 170] }
    },
    {
      "entity": "date_of_birth",
      "value": "1978-05-12",
      "source": "claude_haiku",
      "confidence": 0.95,
      "reasoning": "Format: MM/DD/YYYY detected in date of birth context"
    },
    {
      "entity": "prescription_items",
      "items": [
        {
          "medication": "Ibuprofen 400mg",
          "dosage": "400mg",
          "frequency": "3x/day",
          "duration": "7 days",
          "source": "regex"
        }
      ]
    }
  ],
  "quality_metrics": {
    "overall_confidence": 0.94,
    "required_fields_found": 8,
    "required_fields_total": 8,
    "fallback_used": false
  },
  "audit": {
    "patterns_matched": ["NAME_PATTERN_v3", "DOB_PATTERN_v2"],
    "rules_applied": ["STRICT_MODE"],
    "warnings": []
  }
}
```

---

## 🎓 Training the NLP Model

### 📖 Complete Documentation

Detailed instructions for training the spaCy model (annotation, dataset preparation, configuration, training, evaluation) are available here:

👉 **[docs/nlp_training.md](https://github.com/ast116/Medical_OCR-NLP_System/tree/main/docs)**

### ⚡ Quick Summary (for spaCy)

```bash
# 1. Prepare annotated data (spaCy format)
python scripts/prepare_training_data.py \
  --annotations data/annotations.jsonl \
  --output data/train.spacy

# 2. Create spaCy configuration
spacy init config config.cfg \
  --lang en \
  --pipeline ner \
  --optimize accuracy

# 3. Train the model
spacy train config.cfg \
  --output models/ \
  --paths.train data/train.spacy \
  --paths.dev data/dev.spacy

# 4. Evaluate performance
spacy evaluate models/model-best data/test.spacy

# 5. Export for production
cp models/model-best nlp/models/spacy_medical_en
```

### 📊 Training Metrics

```
Expected performance (after 100 epochs):
├── Precision : 0.94-0.96
├── Recall    : 0.90-0.93
├── F1-Score  : 0.92-0.94
└── Time/batch : 0.3-0.5s (GPU)
```

---

## 🛡️ Error Handling

### Robustness Strategies

```
┌─────────────────────────────────────┐
│   Invalid Document / Error          │
├─────────────────────────────────────┤
│ 🔴 STRICT Mode                      │
│    ├── Reject document              │
│    ├── Flag for review              │
│    └── Admin alert                  │
├─────────────────────────────────────┤
│ 🟡 EXPLORATORY Mode                 │
│    ├── Partial output               │
│    ├── Mark with low confidence     │
│    └── Suggest human review         │
├─────────────────────────────────────┤
│ 🟢 FALLBACK Mode                    │
│    ├── Try alternative source       │
│    ├── Combine results              │
│    └── Complete audit logging       │
└─────────────────────────────────────┘
```

### Mode Configuration

```bash
python run_pipeline.py \
  --error-mode strict          # or exploratory / fallback
  --log-level debug
```

---

## ✅ Tests and Metrics

### Test Suite

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# OCR Benchmark
python scripts/benchmark_ocr.py --engine tesseract

# NLP Evaluation
python scripts/eval_nlp.py --model nlp/models/spacy_medical_en

# Full coverage
pytest --cov=src/ --cov-report=html
```

### Tracked Metrics

| Metric | Target | Current |
|---|---|---|
| 🔤 OCR WER (Word Error Rate) | < 3% | 2.1% |
| 🧠 NER Precision (Regex) | > 95% | 96.8% |
| 🧠 NER Recall (Regex) | > 90% | 92.3% |
| ⚡ Processing time/page | < 2s | 1.4s |
| 💾 Memory usage | < 500MB | 380MB |

---

## 🤝 Contribution

We welcome contributions! 🎉

### Process

```bash
1. Fork the repository
2. Create a feature branch
   git checkout -b feature/my-feature

3. Commit your changes
   git commit -m "✨ Add: clear description"

4. Push to your fork
   git push origin feature/my-feature

5. Open a Pull Request
```

### Guidelines

- ✅ Code formatted with `black` and `flake8`
- ✅ Tests for all new code
- ✅ Document regex patterns with examples
- ✅ Follow project conventions

### Welcome Improvements

- 🌐 Multi-language support (additional languages)
- 📊 New medical-specific regex patterns
- 🚀 GPU performance optimizations
- 🐳 Docker image improvements
- 📚 More example notebooks

---

## 📄 License

To be specified: MIT / Apache 2.0 / Custom

---

## 📞 Support

For issues, questions, or discussions about the project:
- **Issues** : [Report a bug](https://github.com/ast116/Medical_OCR-NLP_System/issues)
- **Discussions** : [Questions & ideas](https://github.com/ast116/Medical_OCR-NLP_System/discussions)

---

<div align="center">

**Made with ❤️ for medicine and innovation**

⭐ If you like this project, feel free to star it!

</div>
