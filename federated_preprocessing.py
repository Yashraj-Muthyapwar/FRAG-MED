"""
FRAG-MED Federated System Configuration
Manages configurations for multiple hospital nodes with local RAG systems
"""
from pathlib import Path
from typing import List, Dict
from pydantic_settings import BaseSettings


class HospitalConfig:
    """Configuration for a single hospital node"""
    
    def __init__(self, hospital_id: str, base_dir: Path):
        """
        Initialize hospital-specific configuration
        
        Args:
            hospital_id: Hospital identifier (e.g., 'hospital_A')
            base_dir: Base directory for this hospital
        """
        self.hospital_id = hospital_id
        self.base_dir = Path(base_dir)
        
        # Hospital-specific paths
        self.raw_data_dir = self.base_dir / "preprocessed"
        self.parent_docs_dir = self.base_dir / "parent_docs"
        self.child_nodes_dir = self.base_dir / "child_nodes"
        self.child_nodes_file = self.child_nodes_dir / "child_nodes.jsonl"
        self.chromadb_dir = self.base_dir / "chromadb"
        self.logs_dir = self.base_dir / "logs"
        
        # ChromaDB collection name (unique per hospital)
        self.chromadb_collection = f"frag_med_{hospital_id}"
    
    def create_directories(self):
        """Create all necessary directories for this hospital"""
        for dir_path in [
            self.raw_data_dir,
            self.parent_docs_dir,
            self.child_nodes_dir,
            self.chromadb_dir,
            self.logs_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_patient_count(self) -> int:
        """Get number of patient files in this hospital"""
        if not self.raw_data_dir.exists():
            return 0
        return len(list(self.raw_data_dir.glob("*.json")))
    
    def __repr__(self):
        return f"HospitalConfig(id={self.hospital_id}, patients={self.get_patient_count()})"


class FederatedConfig(BaseSettings):
    """Configuration for FRAG-MED federated system"""
    
    # Project Paths
    PROJECT_ROOT: Path = Path(__file__).parent
    FEDERATED_DIR: Path = PROJECT_ROOT / "federated_hospitals"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
    LOGS_DIR: Path = OUTPUTS_DIR / "logs" / "federated"
    
    # Hospital Configuration
    HOSPITAL_IDS: List[str] = [
        "hospital_A", "hospital_B", "hospital_C", "hospital_D", "hospital_E",
        "hospital_F", "hospital_G", "hospital_H", "hospital_I", "hospital_J"
    ]
    
    # Processing Parameters (same as centralized)
    BATCH_SIZE: int = 50
    EMBEDDING_BATCH_SIZE: int = 32
    MAX_WORKERS: int = 2
    
    # Embedding Model Configuration (shared across hospitals)
    USE_LOCAL_EMBEDDINGS: bool = True
    EMBEDDING_MODEL_PATH: Path = MODELS_DIR / "embeddings" / "neuml_pubmedbert-base-embeddings"
    EMBEDDING_MODEL_NAME: str = "neuml/pubmedbert-base-embeddings"
    EMBEDDING_DIM: int = 768
    
    # LLM Model Configuration (shared across hospitals)
    USE_LOCAL_LLM: bool = True
    LLM_MODEL_NAME: str = "jsk/bio-mistral"
    LLM_CONTEXT_WINDOW: int = 8192
    LLM_TIMEOUT: int = 240
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 512
    
    # ChromaDB Configuration
    CHROMADB_DISTANCE_METRIC: str = "cosine"
    
    # RAG Configuration
    SIMILARITY_TOP_K: int = 3
    RESPONSE_MODE: str = "compact"
    CHUNK_OVERLAP: int = 50
    PARENT_DOC_MAX_TOKENS: int = 2000
    EXTRACT_SECTIONS_ONLY: bool = True
    
    # Parent Document Configuration
    INCLUDE_DEVICES: bool = True
    MAX_DEVICES_DISPLAY: int = 20
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Memory Management
    MEMORY_THRESHOLD_GB: float = 12.0
    GC_FREQUENCY: int = 10
    
    # Performance Monitoring
    TRACK_LATENCY: bool = True
    TRACK_TOKEN_USAGE: bool = True
    TRACK_RETRIEVAL_QUALITY: bool = True
    
    # Phoenix Observability (shared)
    PHOENIX_DIR: Path = OUTPUTS_DIR / "phoenix"
    PHOENIX_HOST: str = "127.0.0.1"
    PHOENIX_PORT: int = 6006
    ENABLE_PHOENIX: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def get_hospital_config(self, hospital_id: str) -> HospitalConfig:
        """
        Get configuration for a specific hospital
        
        Args:
            hospital_id: Hospital identifier
            
        Returns:
            HospitalConfig instance
        """
        if hospital_id not in self.HOSPITAL_IDS:
            raise ValueError(f"Unknown hospital ID: {hospital_id}")
        
        hospital_dir = self.FEDERATED_DIR / hospital_id
        return HospitalConfig(hospital_id, hospital_dir)
    
    def get_all_hospital_configs(self) -> List[HospitalConfig]:
        """Get configurations for all hospitals"""
        return [self.get_hospital_config(hid) for hid in self.HOSPITAL_IDS]
    
    def create_all_directories(self):
        """Create directories for all hospitals and shared resources"""
        # Create shared directories
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.PHOENIX_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create hospital-specific directories
        for hospital_config in self.get_all_hospital_configs():
            hospital_config.create_directories()
    
    def get_system_summary(self) -> Dict:
        """Get summary of federated system"""
        hospitals = self.get_all_hospital_configs()
        
        return {
            "total_hospitals": len(hospitals),
            "hospitals_with_data": sum(1 for h in hospitals if h.get_patient_count() > 0),
            "total_patients": sum(h.get_patient_count() for h in hospitals),
            "patient_distribution": {
                h.hospital_id: h.get_patient_count() 
                for h in hospitals
            }
        }
    
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


# Global federated config instance
federated_config = FederatedConfig()
federated_config.create_all_directories()
