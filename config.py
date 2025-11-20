"""
FRAG-MED Centralized System Configuration with Phoenix Observability
Configured for LOCAL models (jsk/bio-mistral + neuml/pubmedbert-base-embeddings)
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Config(BaseSettings):
    """Configuration for FRAG-MED preprocessing and indexing"""
    
    # Project Paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "preprocessed"
    PARENT_DOCS_DIR: Path = DATA_DIR / "parent_docs"
    CHILD_NODES_DIR: Path = DATA_DIR / "child_nodes"
    CHILD_NODES_FILE: Path = CHILD_NODES_DIR / "child_nodes.jsonl"
    CHROMADB_DIR: Path = DATA_DIR / "chromadb"
    
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
    LOGS_DIR: Path = OUTPUTS_DIR / "logs"
    
    # Phoenix Observability
    PHOENIX_DIR: Path = OUTPUTS_DIR / "phoenix"
    PHOENIX_HOST: str = "127.0.0.1"
    PHOENIX_PORT: int = 6006
    PHOENIX_COLLECTOR_ENDPOINT: str = f"http://{PHOENIX_HOST}:{PHOENIX_PORT}"
    ENABLE_PHOENIX: bool = True  # Set to False to disable Phoenix
    
    # Processing Parameters
    BATCH_SIZE: int = 50  # Process 50 patients at a time (memory-efficient)
    EMBEDDING_BATCH_SIZE: int = 32  # Embedding batch size
    MAX_WORKERS: int = 2  # Parallel workers (conservative for 16GB RAM)
    
    # ===== LOCAL MODEL CONFIGURATION =====
    
    # Embedding Model - neuml/pubmedbert-base-embeddings (LOCAL)
    USE_LOCAL_EMBEDDINGS: bool = True
    EMBEDDING_MODEL_PATH: Path = MODELS_DIR / "embeddings" / "neuml_pubmedbert-base-embeddings"
    EMBEDDING_MODEL_NAME: str = "neuml/pubmedbert-base-embeddings"  # For reference
    EMBEDDING_DIM: int = 768  # PubMedBERT dimension
    
    # LLM Model - jsk/bio-mistral (LOCAL via Ollama)
    USE_LOCAL_LLM: bool = True
    LLM_MODEL_NAME: str = "jsk/bio-mistral"  # Ollama model name
    LLM_CONTEXT_WINDOW: int = 8192  # BioMistral context window
    LLM_TIMEOUT: int = 240  # Timeout for LLM requests

    
    LLM_TEMPERATURE: float = 0.3  # Low temperature for factual medical responses
    LLM_MAX_TOKENS: int = 3072  # Max tokens in response
    
    # ChromaDB Configuration
    CHROMADB_COLLECTION: str = "frag_med_central"
    CHROMADB_DISTANCE_METRIC: str = "cosine"
    
    # RAG Configuration
    SIMILARITY_TOP_K: int = 3  # Retrieve top-5 encounters
    RESPONSE_MODE: str = "compact"
    CHUNK_OVERLAP: int = 50  # Overlap for text chunking

    PARENT_DOC_MAX_TOKENS: int = 2000  # Truncate to 2000 tokens
    EXTRACT_SECTIONS_ONLY: bool = True  # Extract only === SECTIONS === (faster + more accurate)
    
    # Parent Document Configuration
    INCLUDE_DEVICES: bool = True  # Include patient devices in parent docs
    MAX_DEVICES_DISPLAY: int = 20  # Max devices to show in parent doc
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Memory Management (16GB RAM)
    MEMORY_THRESHOLD_GB: float = 12.0  # Pause if RAM usage > 12GB
    GC_FREQUENCY: int = 10  # Run garbage collection every N batches
    
    # Performance Monitoring
    TRACK_LATENCY: bool = True
    TRACK_TOKEN_USAGE: bool = True
    TRACK_RETRIEVAL_QUALITY: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def create_directories(self):
        """Create all necessary directories"""
        for dir_path in [
            self.PARENT_DOCS_DIR,
            self.CHILD_NODES_DIR,
            self.CHROMADB_DIR,
            self.LOGS_DIR,
            self.MODELS_DIR / "embeddings",
            self.PHOENIX_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate_local_models(self) -> bool:
        """Validate that local models exist"""
        issues = []
        
        if self.USE_LOCAL_EMBEDDINGS:
            if not self.EMBEDDING_MODEL_PATH.exists():
                issues.append(f"Embedding model not found: {self.EMBEDDING_MODEL_PATH}")
        
        if issues:
            print("⚠️  Model Validation Issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        return True

# Global config instance
config = Config()
config.create_directories()
