# FRAG-MED: Federated Retrieval-Augmented Generation for Medical Diagnosis

<div align="center">

**A privacy-preserving federated RAG system enabling collaborative medical diagnosis across multiple healthcare institutions without centralizing sensitive patient data.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG%20framework-blue)](https://docs.llamaindex.ai/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-green)](https://www.trychroma.com/)
[![Transformers](https://img.shields.io/badge/Transformers-PubMedBERT-orange)](https://huggingface.co/neuml/pubmedbert-base-embeddings)
[![Ollama](https://img.shields.io/badge/Ollama-BioMistral--7B-red)](https://ollama.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-pink)](https://streamlit.io/)
[![Arize Phoenix](https://img.shields.io/badge/Arize%20Phoenix-Observability-lightgrey)](https://docs.arize.com/phoenix/)

</div>

---

## 🎯 Overview

Healthcare institutions face a critical challenge: they need to share knowledge to improve patient outcomes, but regulations like HIPAA and GDPR prevent them from centralizing sensitive patient data. FRAG-MED solves this by enabling multiple hospitals to collaboratively answer medical queries while keeping all patient data within their own walls.

### Key Features

- 🏥 **Federated Architecture**: 10 independent hospital nodes with local RAG systems
- 🔒 **Privacy-Preserving**: Comprehensive de-identification (names, dates, addresses removed)
- 🧠 **Medical AI**: BioMistral-7B LLM + PubMedBERT embeddings (768-dim)
- 📊 **Hierarchical Retrieval**: Parent-child document architecture for efficient search
- 💻 **Local Deployment**: No external API dependencies, complete data sovereignty
- 🎨 **Web Interface**: Interactive Streamlit UI for easy querying


## 🗺️ Architecture

![FRAG-MED System Architecture](Screenshots/architecture_diagram.png)

### System Flow

1. **Data Generation**: Synthea creates realistic synthetic patient records (11,202 patients, 604,688 encounters)
2. **Preprocessing**: Each hospital independently processes data through de-identification and vector indexing
3. **Query Processing**: User queries are distributed to all hospitals, each performing local RAG
4. **Aggregation**: Responses are combined into a unified answer while maintaining privacy


## 📋 Prerequisites

- **Python 3.12+**
- **16GB RAM** (8GB minimum)
- **~60GB storage**
- **Ollama** (for local LLM)
- **Java 11+** (for Synthea data generation)


## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/Yashraj-Muthyapwar/FRAG-MED.git
cd FRAG-MED

# Create virtual environment
python3.12 -m venv fl_env
source fl_env/bin/activate  # Windows: fl_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install Ollama & Models
```bash
# Install Ollama (macOS)
brew install ollama

# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Pull BioMistral model
ollama pull jsk/bio-mistral
```

### 3. Download Embeddings
```bash
python download.py
```

Downloads PubMedBERT (~420MB) to `models/embeddings/`.

### 4. Verify Setup
```bash
python verify_setup.py
```

Expected: ✅ READY TO RUN!


## 📊 Generate Patient Data

We use [Synthea](https://github.com/synthetichealth/synthea) for synthetic FHIR patient records.
```bash
# Clone Synthea
git clone https://github.com/synthetichealth/synthea.git
cd synthea

# Generate 1000 patients
./run_synthea -p 1000 Texas
```

Place generated files in `data/preprocessed/` or see our [data generation notebook](data_generation.ipynb).


## ⚙️ Preprocessing

### Centralized System
```bash
python src/main_preprocessing.py
```

### Federated System
```bash
# Process each hospital
python hospital_preprocessing.py hospital_A
python hospital_preprocessing.py hospital_B
# ... repeat for hospitals C-J
```
### 🔀 Dynamic Hospital Splitting (optional)

You can use **`hospital_splitting.ipynb`** to automatically create realistic federated silos:

- 📂 **Analyze raw patient JSON files** in `data/raw_patients/`
- 🏥 **Build specialization profiles** for each hospital based on top medical conditions
- 👥 **Assign patients** to the best-matching hospital
- 📤 **Export results** to `data/federated_hospitals/`

### ⚙️ Preprocessing Outputs

| **Category**            | **Centralized System**                          | **Federated System**                                      |
|--------------------------|------------------------------------------------|-----------------------------------------------------------|
| 🎉 Status               | PREPROCESSING COMPLETE!                         | FEDERATED PREPROCESSING COMPLETE!                         |
| Patients                 | 11,202                                         | 11,202 (distributed across hospitals)                     |
| Encounters               | 604,688                                        | 604,688 (distributed across hospitals)                    |
| Hospitals                | Single centralized repository                  | 10 hospitals (hospital_A … hospital_J)                    |
| Parent docs              | `data/parent_docs/`                            | Per-hospital dirs: `parent_docs/`                         |
| Vector index             | `data/chromadb/`                               | Per-hospital dirs: `chromadb/`                            |
| Preprocessed data        | Centralized in `data/preprocessed/`            | Per-hospital dirs: `preprocessed/`                        |
| Hospital silos root      | —                                              | `federated_hospitals/`                                    |


## 💻 Running Queries

### Web Interface
```bash
streamlit run app.py
```

Access at `http://localhost:8501`

**Features:**
- Switch between centralized/federated modes
- Try sample queries or write custom ones
- View source citations

## 🔒 Privacy Features

### De-identification

All patient data is automatically de-identified:

| Original | De-identified |
|----------|---------------|
| John Smith | `PATIENT_a47c0828` |
| 2023-03-15 | `2023-Q1` |
| Age 65 | `60-69` |
| Address | `[REDACTED]` |
| Phone | `[REDACTED]` |
| SSN | `[REDACTED]` |

### Architecture

- ✅ **Local processing**: All computation within hospital boundaries
- ✅ **No raw data sharing**: Only aggregated responses leave hospitals
- ✅ **HIPAA-compliant design**: De-identification before indexing

## 📁 Project Structure

```text
FRAG-MED/
├── app.py                          # Streamlit / web UI
├── config.py                       # Global configuration
├── custom_query.py                 # Custom RAG query runner
├── download.py                     # Download embedding model locally
├── hospital_splitting.ipynb        # Optional dynamic hospital splitter
├── requirements.txt                # To install all the libraries and dependencies
├── readme.md
├── verify_setup.py                 # Sanity checks for paths/models

├── models/
│   └── embeddings/
│       └── neuml_pubmedbert-base-embeddings/   # Local PubMedBERT embeddings

├── data/                           # Centralized (non-federated) artifacts
│   ├── preprocessed/               # Cleaned patient JSON files
│   ├── parent_docs/                # Long-form parent documents (batched)
│   ├── child_nodes/                # Chunked child nodes for retrieval
│   └── chromadb/                   # Centralized Chroma vector store

├── src/
│   ├── main_preprocessing.py       # End-to-end centralized preprocessing
│   ├── monitor_system.py           # System resource monitoring
│   ├── preprocessing/              # Preprocessing + indexing pipeline
│   │   ├── batch_processor.py
│   │   ├── parent_storage.py
│   │   └── child_indexer.py
│   ├── rag/                        # RAG query engine
│   │   └── query_engine.py
│   ├── utils/                      # Helpers & de-identification
│   │   ├── data_loader.py
│   │   ├── deidentification.py
│   │   └── node_generator.py
│   └── observability/              # LLM observability (Phoenix)
│       └── phoenix_setup.py

├── federated_hospitals/            # Federated hospital silos (A–J)
│   ├── hospital_A/
│   │   ├── preprocessed/           # Hospital-level preprocessed data
│   │   ├── parent_docs/            # Hospital-level parent docs
│   │   ├── child_nodes/            # Hospital-level chunks
│   │   ├── chromadb/               # Hospital-level vector DB
│   │   └── logs/                   # Local RAG logs
│   ├── hospital_B/
│   └── ... hospital_C ... hospital_J/

├── hospital_preprocessing.py       # Build per-hospital silos
├── hospital_rag_dp.py              # Hospital-side RAG with DP
├── federated_config.py             # Federated-specific config
├── federated_aggregation.py        # Aggregation + majority voting
├── federated_orchestrator_dp.py    # Federated coordinator (DP-aware)
└── outputs/
    ├── logs/
    │   └── federated/              # Federated run logs
    └── phoenix/                    # Arize Phoenix traces & artifacts
```

## 🛠️ Configuration

Edit `config.py` or `federated_config.py`:
```python
# Model paths
EMBEDDING_MODEL_PATH = "models/embeddings/neuml_pubmedbert-base-embeddings"
LLM_MODEL_NAME = "jsk/bio-mistral"

# RAG parameters
SIMILARITY_TOP_K = 3          # Documents to retrieve
LLM_TEMPERATURE = 0.3         # 0=deterministic, 1=creative
LLM_MAX_TOKENS = 3072         # Max response length
```

### 📚 Resources
- **[Synthea](https://github.com/synthetichealth/synthea)** - Patient data generator
- **[BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B)** - Medical LLM
- **[PubMedBERT](https://huggingface.co/neuml/pubmedbert-base-embeddings)** - Clinical embeddings
- **[LlamaIndex](https://docs.llamaindex.ai/)** - RAG framework

### ⚠️ Disclaimer

**Research prototype for educational purposes only.**
**Always consult qualified medical professionals for clinical decisions.**

### 📝 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

**⭐ Star this repo if you find it useful! ⭐**

Contributions welcome built with ❤️ to advance privacy‑preserving healthcare AI and federated medical insights.

