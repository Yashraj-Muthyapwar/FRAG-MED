# FRAG-MED: Federated Retrieval-Augmented Generation for Medical Diagnosis

<div align="center">

**A privacy-preserving federated RAG system enabling collaborative medical diagnosis across multiple healthcare institutions without centralizing sensitive patient data.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
python download_local_models.py
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

**Output:**
```
🎉 PREPROCESSING COMPLETE!
├─ Patients: 11,202
├─ Encounters: 604,688
├─ Parent docs: data/parent_docs/
└─ Vector index: data/chromadb/
```



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
```
FRAG-MED/
├── app.py              # Web UI
├── config.py                      # Configuration
├── src/
│   ├── preprocessing/             # Data pipeline
│   ├── rag/                       # Query engine
│   └── utils/                     # De-identification
├── federated_hospitals/           # Hospital data silos
│   ├── hospital_A/
│   │   ├── preprocessed/          # Patient files
│   │   ├── parent_docs/           # Full contexts
│   │   └── chromadb/              # Vector DB
│   └── hospital_B...J/
├── hospital_preprocessing.py      # Per-hospital setup
├── federated_orchestrator_dp.py   # Federated coordinator
├── run_federated_dp.py            # CLI queries
└── test_queries.py                # Sample queries
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
