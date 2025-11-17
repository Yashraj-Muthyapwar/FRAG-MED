"""
Parent Document Storage - Manages storage of full encounter documents
"""
import json
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class ParentDocumentStorage:
    """Stores parent documents to disk"""
    
    def __init__(self, storage_dir: Path):
        """
        Initialize parent document storage
        
        Args:
            storage_dir: Directory to store parent documents
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_saved': 0,
            'batches_processed': 0
        }
    
    def save_batch(self, parent_nodes: List[Dict], batch_idx: int):
        """
        Save a batch of parent documents
        
        Args:
            parent_nodes: List of parent node dictionaries
            batch_idx: Batch index for organization
        """
        if not parent_nodes:
            logger.warning(f"Batch {batch_idx} has no parent nodes to save")
            return
        
        # Create batch directory
        batch_dir = self.storage_dir / f"batch_{batch_idx:04d}"
        batch_dir.mkdir(exist_ok=True)
        
        # Save each parent document
        for parent_node in parent_nodes:
            try:
                file_path = batch_dir / f"{parent_node['id']}.json"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(parent_node, f, indent=2, ensure_ascii=False)
                
                self.stats['total_saved'] += 1
                
            except Exception as e:
                logger.error(
                    f"Error saving parent doc {parent_node['id']}: {e}"
                )
        
        self.stats['batches_processed'] += 1
        
        if self.stats['batches_processed'] % 10 == 0:
            logger.info(
                f"Saved {self.stats['total_saved']} parent documents "
                f"({self.stats['batches_processed']} batches)"
            )
    
    def get_stats(self) -> Dict:
        """Get storage statistics"""
        return self.stats.copy()
    
    def load_parent_doc(self, doc_id: str) -> Dict:
        """
        Load a parent document by ID
        
        Args:
            doc_id: Parent document ID
            
        Returns:
            Parent document dictionary
        """
        # Search through batch directories
        for batch_dir in sorted(self.storage_dir.glob("batch_*")):
            file_path = batch_dir / f"{doc_id}.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        raise FileNotFoundError(f"Parent document not found: {doc_id}")
