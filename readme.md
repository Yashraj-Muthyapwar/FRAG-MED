# FRAG-MED: Federated Retrieval-Augmented Generation for Medical Diagnosis

A privacy-preserving, federated Retrieval-Augmented Generation (RAG) system for collaborative medical diagnosis across multiple healthcare institutions without centralizing sensitive patient data.

## 🎯 Project Overview

FRAG-MED addresses a critical challenge in healthcare AI: enabling collaborative diagnosis across hospitals while maintaining strict data privacy. By combining **Retrieval-Augmented Generation**, **Federated Learning**, and **Differential Privacy**, this system allows multiple healthcare institutions to benefit from collective knowledge without sharing raw patient data.

### Key Features

- **🏥 Federated Architecture**: Each hospital maintains complete data sovereignty
- **🔒 Privacy-Preserving**: Comprehensive de-identification and differential privacy
- **🧠 Medical-Specific Models**: BioMistral-7B LLM + PubMedBERT embeddings
- **📊 Observability**: Real-time monitoring with Arize Phoenix
- **⚡ Efficient Retrieval**: Parent-child document architecture for fast, context-rich responses
- **🔍 Hybrid Search**: Vector similarity + keyword matching for precise retrieval

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Components](#system-components)
- [Usage](#usage)
- [Dataset](#dataset)
- [Performance](#performance)
- [Privacy & Security](#privacy--security)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRAG-MED System                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Hospital A   │  │ Hospital B   │  │ Hospital C   │          │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤          │
│  │ Local Data   │  │ Local Data   │  │ Local Data   │          │
│  │ ChromaDB     │  │ ChromaDB     │  │ ChromaDB     │          │
│  │ BioMistral   │  │ BioMistral   │  │ BioMistral   │          │
│  │ PubMedBERT   │  │ PubMedBERT   │  │ PubMedBERT   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │ DP-Protected    │ Responses        │                  │
│         └─────────────────┼──────────────────┘                  │
│                           │                                     │
│                  ┌────────▼────────┐                            │
│                  │   Federated     │                            │
│                  │  Orchestrator   │                            │
│                  │  (Aggregation)  │                            │
│                  └─────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Components

1. **Preprocessing Pipeline**: De-identification, parent-child node generation
2. **Local RAG Systems**: Independent retrieval and generation per hospital
3. **Vector Database**: ChromaDB for efficient semantic search
4. **LLM**: BioMistral-7B for medical reasoning
5. **Embeddings**: PubMedBERT for clinical text representation
6. **Observability**: Phoenix for real-time monitoring and traces

## 🚀 Installation

### Prerequisites

- Python 3.12+
- 16GB+ RAM (recommended)
- Ollama (for local LLM serving)
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/FRAG-MED.git
cd FRAG-MED
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3.12 -m venv fl_env

# Activate (macOS/Linux)
source fl_env/bin/activate

# Activate (Windows)
fl_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install Ollama and Models

```bash
# Install Ollama (macOS)
brew install ollama

# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull BioMistral model
ollama pull jsk/bio-mistral

# Verify installation
ollama list
```

### Step 5: Download Embedding Model

```bash
# Download PubMedBERT embeddings
python download_local_models.py
```

### Step 6: Verify Setup

```bash
python verify_setup.py
```

## ⚡ Quick Start

### 1. Prepare Data

```bash
# Place preprocessed patient data in data/preprocessed/
# (See Dataset section for Synthea data generation)
```

### 2. Run Preprocessing Pipeline

```bash
python src/main_preprocessing.py
```

**Expected Output:**
```
🔄 PHASE 1: Processing patient files into parent-child nodes...
✅ Phase 1 Complete!
Parent docs saved: 10,847
Child nodes saved: 10,847

🔄 PHASE 2: Building ChromaDB index with LOCAL embeddings...
✅ Phase 2 Complete!
Child nodes indexed: 10,847

🎉 PREPROCESSING PIPELINE COMPLETE!
```

### 3. Start Phoenix Observability (Optional)

```bash
# In a separate terminal
python monitor_system.py
```

Access Phoenix Dashboard at: `http://127.0.0.1:6006`

### 4. Run Sample Queries

```bash
python test_queries.py
```

**Example Query:**
```
Query: "Find patients with acute bronchitis and describe their diagnostic procedures."

Response: 
Patient PATIENT_a47c0828 (60-69, Male, White) was diagnosed with acute 
bronchitis in 2018-Q2. Diagnostic procedures included chest X-ray showing 
bilateral infiltrates and pulmonary function tests. Patient was prescribed 
Albuterol inhaler and Azithromycin 500mg...

Retrieved Sources: 5 encounters
Latency: 4.2s
Tokens: ~450
```

### 5. Custom Queries

```bash
# Interactive query interface
python custom_query.py
```

## 🧩 System Components

### Preprocessing (`src/preprocessing/`)

- **BatchProcessor**: Memory-efficient patient file processing
- **NodeGenerator**: Parent-child document generation
- **DeIdentifier**: Comprehensive PII removal
- **ChildIndexer**: ChromaDB indexing with embeddings
- **ParentStorage**: Full-context document storage

### RAG System (`src/rag/`)

- **CentralRAGSystem**: Main query engine
- **ParentDocumentRetriever**: Custom retrieval across batch folders
- **Query Engine**: Integration with LlamaIndex

### Observability (`src/observability/`)

- **PhoenixObservability**: OpenTelemetry tracing
- **PerformanceMonitor**: Latency and token tracking

### Utilities (`src/utils/`)

- **PatientDataLoader**: JSON file loading and validation
- **DeIdentifier**: Privacy-preserving transformations
- **NodeGenerator**: Document structuring

## 📊 Usage

### Basic Query

```python
from src.rag import CentralRAGSystem

# Initialize system
rag = CentralRAGSystem(enable_phoenix=True, verbose=True)

# Run query
result = rag.query(
    "What are common treatments for Type 2 diabetes in elderly patients?",
    show_sources=True
)

print(result['response'])
print(f"Latency: {result['latency']:.2f}s")

# Shutdown
rag.shutdown()
```

### Advanced Configuration

```python
from config import config

# Customize settings
config.SIMILARITY_TOP_K = 10  # Retrieve top-10 instead of 5
config.LLM_TEMPERATURE = 0.2  # Increase creativity
config.LLM_MAX_TOKENS = 4096  # Longer responses

# Initialize with custom config
rag = CentralRAGSystem(enable_phoenix=True)
```

### Batch Processing

```python
queries = [
    "Find patients with hypertension over 60",
    "Common medications for diabetes",
    "Diagnostic procedures for respiratory conditions"
]

for query in queries:
    result = rag.query(query)
    # Process results...
```

## 📚 Dataset

### Synthea Synthetic Patient Data

FRAG-MED uses the [Synthea Patient Generator](https://synthetichealth.github.io/synthea/) to create realistic synthetic patient records.

**Dataset Statistics:**
- **Total Patients**: ~11,000
- **Time Period**: Q4 2017 - Q1 2023
- **Encounters**: ~10,847
- **Conditions**: 200+ unique diagnoses
- **Procedures**: 150+ documented procedures
- **Medications**: 300+ prescriptions
- **Format**: FHIR JSON → Preprocessed EHR JSON

### Generating Your Own Dataset

```bash
# Clone Synthea
git clone https://github.com/synthetichealth/synthea.git
cd synthea

# Generate 1000 patients (Massachusetts)
./run_synthea -p 1000 Massachusetts

# Output will be in output/fhir/
```

### Data Privacy

All patient data undergoes comprehensive de-identification:
- ✅ Names → Pseudonyms (PATIENT_XXXXXXXX)
- ✅ Dates → Quarters (YYYY-QN)
- ✅ Ages → Age bands (e.g., 60-69)
- ✅ Addresses → [ADDRESS-REDACTED]
- ✅ Phone numbers → [PHONE-REDACTED]
- ✅ SSNs → [SSN-REDACTED]
- ✅ Medical record numbers → [MRN-REDACTED]

## 📈 Performance

### Current Benchmarks (Centralized System)

| Metric | Value |
|--------|-------|
| **Query Latency** | 3-8s (optimal), 20-60s (worst case) |
| **Retrieval Time** | <1s (98% of latency from LLM) |
| **Tokens/Query** | ~500-1000 |
| **ChromaDB Vectors** | 10,847 |
| **Embedding Dimension** | 768 (PubMedBERT) |
| **Context Window** | 8192 tokens (BioMistral) |

### Optimization Strategies

- ✅ Parent document caching
- ✅ Token reduction via section extraction
- ✅ Batch embedding generation
- ✅ Adversarial prompt engineering
- 🔄 Hybrid search (in development)
- 🔄 Response streaming (planned)

## 🔒 Privacy & Security

### De-identification Pipeline

```python
from src.utils import DeIdentifier

deidentifier = DeIdentifier()

# Scrub clinical notes
clean_text = deidentifier.scrub_clinical_notes(
    text=raw_clinical_note,
    patient_name="John Smith",
    pseudonym="PATIENT_a47c0828",
    age=65,
    age_band="60-69",
    encounter_date="2023-03-15",
    quarter="2023-Q1"
)
```

### Differential Privacy (Federated Mode)

```python
# Add ε-differential privacy to embeddings
from src.privacy import DifferentialPrivacy

dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
noisy_embeddings = dp.add_noise_to_embeddings(embeddings)
```

### Security Best Practices

- 🔒 All models run locally (no external API calls)
- 🔒 Data never leaves hospital premises
- 🔒 Encrypted communication between nodes
- 🔒 Audit logging for all queries
- 🔒 HIPAA/GDPR compliance by design

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run performance benchmarks
python benchmarks/run_benchmarks.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linters
black src/
flake8 src/
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use FRAG-MED in your research, please cite:

```bibtex
@misc{fragmed2025,
  title={FRAG-MED: A Federated Retrieval-Augmented Framework for Secure and Collaborative Medical Diagnosis},
  author={Chamakura, Hemanth Kumar and Muthyapwar, Yash Raj and Pabbathi, Rohith and Kuruva, Shashank},
  year={2025},
  publisher={GitHub},
  url={https://github.com/YOUR_USERNAME/FRAG-MED}
}
```

## 👥 Team

**Team16 - University of North Texas**

- Hemanth Kumar Chamakura - Data Engineering, Privacy Coordination
- Yash Raj Muthyapwar - Data Engineering, Model Integration
- Rohith Pabbathi - Model Development, Evaluation
- Shashank Kuruva - Privacy Implementation, Analysis

**Advisor:** Dr. Stephen F. Wheeler

## 🔗 Related Resources

- [Synthea Patient Generator](https://synthetichealth.github.io/synthea/)
- [BioMistral Model](https://huggingface.co/BioMistral/BioMistral-7B)
- [PubMedBERT Embeddings](https://huggingface.co/neuml/pubmedbert-base-embeddings)
- [Arize Phoenix](https://docs.arize.com/phoenix)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)

## 🐛 Known Issues

See [GitHub Issues](https://github.com/YOUR_USERNAME/FRAG-MED/issues) for current bugs and feature requests.

## 📞 Contact

For questions or collaboration:
- Open an [issue](https://github.com/YOUR_USERNAME/FRAG-MED/issues)
- Email: yash.muthyapwar@unt.edu

---

**⚠️ Disclaimer**: This is a research prototype for educational purposes. It should not be used for actual clinical diagnosis without proper validation, regulatory approval, and medical oversight.
