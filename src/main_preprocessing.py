#!/usr/bin/env python3
"""
FRAG-MED Main Preprocessing Script
Orchestrates the entire preprocessing pipeline for centralized RAG system
Configured for LOCAL models and includes patient devices
"""
import logging
import sys
from pathlib import Path
import coloredlogs
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from src.preprocessing import BatchProcessor, ParentDocumentStorage, ChildNodeIndexer


def setup_logging():
    """Configure logging"""
    log_file = config.LOGS_DIR / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add colored output for console
    coloredlogs.install(
        level=config.LOG_LEVEL,
        fmt=config.LOG_FORMAT
    )
    
    return logging.getLogger(__name__)


def main():
    """Main preprocessing pipeline"""
    
    logger = setup_logging()
    
    logger.info("="*80)
    logger.info("FRAG-MED CENTRALIZED PREPROCESSING PIPELINE (LOCAL MODELS)")
    logger.info("="*80)
    logger.info(f"Raw data directory: {config.RAW_DATA_DIR}")
    logger.info(f"Parent docs directory: {config.PARENT_DOCS_DIR}")
    logger.info(f"Child nodes file: {config.CHILD_NODES_FILE}")
    logger.info(f"ChromaDB directory: {config.CHROMADB_DIR}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Embedding model: {config.EMBEDDING_MODEL_PATH}")
    logger.info(f"Include devices: {config.INCLUDE_DEVICES}")
    logger.info("="*80)
    
    # Validate local models
    if not config.validate_local_models():
        logger.error("Local model validation failed. Exiting.")
        sys.exit(1)
    
    # --- PHASE 1: BATCH PROCESSING ---
    logger.info("\n🔄 PHASE 1: Processing patient files into parent-child nodes...")
    
    processor = BatchProcessor(
        data_dir=config.RAW_DATA_DIR,
        batch_size=config.BATCH_SIZE,
        memory_threshold_gb=config.MEMORY_THRESHOLD_GB,
        gc_frequency=config.GC_FREQUENCY,
        include_devices=config.INCLUDE_DEVICES,
        max_devices_display=config.MAX_DEVICES_DISPLAY
    )
    
    parent_storage = ParentDocumentStorage(
        storage_dir=config.PARENT_DOCS_DIR
    )
    
    child_indexer = ChildNodeIndexer(
        chromadb_dir=config.CHROMADB_DIR,
        child_nodes_file=config.CHILD_NODES_FILE,
        embedding_model_path=config.EMBEDDING_MODEL_PATH,
        collection_name=config.CHROMADB_COLLECTION,
        distance_metric=config.CHROMADB_DISTANCE_METRIC,
        embedding_batch_size=config.EMBEDDING_BATCH_SIZE
    )
    
    # Initialize child nodes JSONL file
    child_indexer.initialize_jsonl()
    
    # Process all patients in batches
    try:
        for parent_nodes, child_nodes, batch_idx in processor.process_all_patients():
            # Save parent documents to disk
            parent_storage.save_batch(parent_nodes, batch_idx)
            
            # Save child nodes to JSONL
            child_indexer.save_child_batch(child_nodes)
        
        # Close child nodes file
        child_indexer.close_jsonl()
        
        logger.info("\n✅ Phase 1 Complete!")
        logger.info(f"Parent docs saved: {parent_storage.get_stats()['total_saved']}")
        logger.info(f"Child nodes saved: {child_indexer.get_stats()['child_nodes_saved']}")
        
    except Exception as e:
        logger.error(f"Error in Phase 1: {e}", exc_info=True)
        sys.exit(1)
    
    # --- PHASE 2: CHROMADB INDEXING ---
    logger.info("\n🔄 PHASE 2: Building ChromaDB index with LOCAL embeddings...")
    
    try:
        child_indexer.build_index()
        
        logger.info("\n✅ Phase 2 Complete!")
        logger.info(f"Child nodes indexed: {child_indexer.get_stats()['child_nodes_indexed']}")
        
    except Exception as e:
        logger.error(f"Error in Phase 2: {e}", exc_info=True)
        sys.exit(1)
    
    # --- FINAL SUMMARY ---
    logger.info("\n" + "="*80)
    logger.info("🎉 PREPROCESSING PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info("\nFINAL STATISTICS:")
    logger.info(f"  Patients processed: {processor.stats['total_patients']}")
    logger.info(f"  Encounters processed: {processor.stats['total_encounters']}")
    logger.info(f"  Parent docs saved: {parent_storage.get_stats()['total_saved']}")
    logger.info(f"  Child nodes indexed: {child_indexer.get_stats()['child_nodes_indexed']}")
    logger.info(f"  Devices included: {config.INCLUDE_DEVICES}")
    logger.info(f"\n📁 Output Locations:")
    logger.info(f"  Parent documents: {config.PARENT_DOCS_DIR}")
    logger.info(f"  ChromaDB index: {config.CHROMADB_DIR}")
    logger.info(f"  Child nodes JSONL: {config.CHILD_NODES_FILE}")
    logger.info(f"\n🤖 Model Configuration:")
    logger.info(f"  Embedding model: {config.EMBEDDING_MODEL_PATH}")
    logger.info(f"  LLM model: {config.LLM_MODEL_NAME}")
    logger.info("="*80)
    
    logger.info("\n✨ Ready to build RAG query system!")


if __name__ == "__main__":
    main()
