"""
Child Node Indexer - Manages ChromaDB indexing of child nodes
Updated to use LOCAL embedding model
"""
import json
import logging
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChildNodeIndexer:
    """Indexes child nodes into ChromaDB with embeddings"""
    
    def __init__(
        self,
        chromadb_dir: Path,
        child_nodes_file: Path,
        embedding_model_path: Path,
        collection_name: str = "frag_med_central",
        distance_metric: str = "cosine",
        embedding_batch_size: int = 32
    ):
        """
        Initialize child node indexer
        
        Args:
            chromadb_dir: ChromaDB storage directory
            child_nodes_file: JSONL file to store child nodes temporarily
            embedding_model_path: Path to LOCAL embedding model
            collection_name: ChromaDB collection name
            distance_metric: Distance metric for similarity search
            embedding_batch_size: Batch size for embedding generation
        """
        self.chromadb_dir = Path(chromadb_dir)
        self.child_nodes_file = Path(child_nodes_file)
        self.embedding_model_path = Path(embedding_model_path)
        self.embedding_batch_size = embedding_batch_size
        
        # Ensure directories exist
        self.chromadb_dir.mkdir(parents=True, exist_ok=True)
        self.child_nodes_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Open child nodes file for writing
        self.child_file = None
        
        # ChromaDB client and collection
        self.client = None
        self.collection = None
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        
        # Embedding model
        self.embedder = None
        
        self.stats = {
            'child_nodes_saved': 0,
            'child_nodes_indexed': 0,
            'batches_indexed': 0
        }
    
    def initialize_jsonl(self):
        """Open JSONL file for writing child nodes"""
        self.child_file = open(self.child_nodes_file, 'w', encoding='utf-8')
        logger.info(f"Opened child nodes file: {self.child_nodes_file}")
    
    def save_child_batch(self, child_nodes: List[Dict]):
        """
        Save child nodes to JSONL file
        
        Args:
            child_nodes: List of child node dictionaries
        """
        if self.child_file is None:
            raise RuntimeError("JSONL file not initialized. Call initialize_jsonl() first")
        
        for child_node in child_nodes:
            self.child_file.write(json.dumps(child_node) + '\n')
            self.stats['child_nodes_saved'] += 1
    
    def close_jsonl(self):
        """Close JSONL file"""
        if self.child_file:
            self.child_file.close()
            logger.info(
                f"Closed child nodes file. "
                f"Total nodes saved: {self.stats['child_nodes_saved']}"
            )
    
    def build_index(self):
        """Build ChromaDB index from saved child nodes using LOCAL embedding model"""
        
        logger.info("="*60)
        logger.info("BUILDING CHROMADB INDEX WITH LOCAL EMBEDDINGS")
        logger.info("="*60)
        
        # Initialize ChromaDB
        logger.info(f"Initializing ChromaDB at {self.chromadb_dir}")
        self.client = chromadb.PersistentClient(
            path=str(self.chromadb_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        logger.info(f"Collection '{self.collection_name}' ready")
        
        # Load LOCAL embedding model
        logger.info(f"Loading LOCAL embedding model from: {self.embedding_model_path}")
        
        if not self.embedding_model_path.exists():
            raise FileNotFoundError(
                f"Embedding model not found at {self.embedding_model_path}\n"
                f"Please ensure neuml/pubmedbert-base-embeddings is downloaded to this location."
            )
        
        self.embedder = SentenceTransformer(str(self.embedding_model_path))
        logger.info("✓ Local embedding model loaded successfully")
        
        # Load child nodes from JSONL
        logger.info(f"Loading child nodes from {self.child_nodes_file}")
        child_nodes = []
        with open(self.child_nodes_file, 'r', encoding='utf-8') as f:
            for line in f:
                child_nodes.append(json.loads(line))
        
        total_nodes = len(child_nodes)
        logger.info(f"Loaded {total_nodes:,} child nodes")
        
        # Index in batches
        num_batches = (total_nodes + self.embedding_batch_size - 1) // self.embedding_batch_size
        
        for batch_idx in tqdm(
            range(0, total_nodes, self.embedding_batch_size),
            total=num_batches,
            desc="Indexing child nodes"
        ):
            batch = child_nodes[batch_idx:batch_idx + self.embedding_batch_size]
            
            # Extract data
            ids = [node['id'] for node in batch]
            summaries = [node['summary'] for node in batch]
            metadatas = [node['metadata'] for node in batch]
            
            # Generate embeddings using LOCAL model
            embeddings = self.embedder.encode(
                summaries,
                batch_size=self.embedding_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                documents=summaries,
                metadatas=metadatas,
                embeddings=embeddings.tolist()
            )
            
            self.stats['child_nodes_indexed'] += len(batch)
            self.stats['batches_indexed'] += 1
        
        logger.info("="*60)
        logger.info("INDEXING COMPLETE")
        logger.info(f"Total vectors in ChromaDB: {self.collection.count():,}")
        logger.info(f"ChromaDB location: {self.chromadb_dir}")
        logger.info(f"Embedding model: {self.embedding_model_path}")
        logger.info("="*60)
    
    def get_stats(self) -> Dict:
        """Get indexing statistics"""
        return self.stats.copy()
