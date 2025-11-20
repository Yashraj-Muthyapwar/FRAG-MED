#!/usr/bin/env python3
"""
FRAG-MED Single Hospital Preprocessing Script
Process data for a specific hospital in the federated system
"""
import sys
import argparse
from pathlib import Path
import logging
import coloredlogs
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from federated_config import federated_config
from src.preprocessing import BatchProcessor, ParentDocumentStorage, ChildNodeIndexer


def setup_logging(hospital_id: str) -> logging.Logger:
    """Configure logging for single hospital preprocessing"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = federated_config.LOGS_DIR / f"preprocessing_{hospital_id}_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, federated_config.LOG_LEVEL),
        format=federated_config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    coloredlogs.install(
        level=federated_config.LOG_LEVEL,
        fmt=federated_config.LOG_FORMAT
    )
    
    return logging.getLogger(f"hospital.{hospital_id}")


def main():
    """Main entry point for single hospital preprocessing"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process data for a specific hospital in FRAG-MED federated system"
    )
    parser.add_argument(
        "hospital_id",
        type=str,
        help="Hospital ID to process (e.g., hospital_A, hospital_B, ...)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if data already exists"
    )
    
    args = parser.parse_args()
    
    # Validate hospital ID
    if args.hospital_id not in federated_config.HOSPITAL_IDS:
        print(f"❌ Error: Unknown hospital ID '{args.hospital_id}'")
        print(f"\nValid hospital IDs:")
        for hid in federated_config.HOSPITAL_IDS:
            print(f"  - {hid}")
        sys.exit(1)
    
    # Get hospital configuration
    hospital_config = federated_config.get_hospital_config(args.hospital_id)
    
    # Setup logging
    logger = setup_logging(args.hospital_id)
    
    logger.info("="*80)
    logger.info(f"🏥 FRAG-MED SINGLE HOSPITAL PREPROCESSING")
    logger.info("="*80)
    logger.info(f"Hospital ID: {hospital_config.hospital_id}")
    logger.info(f"Raw data directory: {hospital_config.raw_data_dir}")
    logger.info(f"Parent docs directory: {hospital_config.parent_docs_dir}")
    logger.info(f"Child nodes file: {hospital_config.child_nodes_file}")
    logger.info(f"ChromaDB directory: {hospital_config.chromadb_dir}")
    logger.info(f"Collection name: {hospital_config.chromadb_collection}")
    logger.info("="*80)
    
    # Check if hospital has data
    patient_count = hospital_config.get_patient_count()
    if patient_count == 0:
        logger.error(f"❌ No patient files found in {hospital_config.raw_data_dir}")
        logger.error(f"Please ensure patient data is placed in this directory")
        sys.exit(1)
    
    logger.info(f"📊 Found {patient_count} patient files")
    
    # Check if already processed
    if not args.force and hospital_config.chromadb_dir.exists():
        existing_files = list(hospital_config.chromadb_dir.glob("*"))
        if existing_files:
            logger.warning(f"⚠️  ChromaDB directory already exists with {len(existing_files)} files")
            logger.warning(f"Use --force flag to reprocess")
            
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                logger.info("Preprocessing cancelled")
                sys.exit(0)
    
    # Validate models
    if not federated_config.validate_local_models():
        logger.error("Local model validation failed. Exiting.")
        sys.exit(1)
    
    try:
        # --- PHASE 1: BATCH PROCESSING ---
        logger.info(f"\n🔄 PHASE 1: Processing patient files into parent-child nodes...")
        
        processor = BatchProcessor(
            data_dir=hospital_config.raw_data_dir,
            batch_size=federated_config.BATCH_SIZE,
            memory_threshold_gb=federated_config.MEMORY_THRESHOLD_GB,
            gc_frequency=federated_config.GC_FREQUENCY,
            include_devices=federated_config.INCLUDE_DEVICES,
            max_devices_display=federated_config.MAX_DEVICES_DISPLAY
        )
        
        parent_storage = ParentDocumentStorage(
            storage_dir=hospital_config.parent_docs_dir
        )
        
        child_indexer = ChildNodeIndexer(
            chromadb_dir=hospital_config.chromadb_dir,
            child_nodes_file=hospital_config.child_nodes_file,
            embedding_model_path=federated_config.EMBEDDING_MODEL_PATH,
            collection_name=hospital_config.chromadb_collection,
            distance_metric=federated_config.CHROMADB_DISTANCE_METRIC,
            embedding_batch_size=federated_config.EMBEDDING_BATCH_SIZE
        )
        
        # Initialize child nodes JSONL file
        child_indexer.initialize_jsonl()
        
        # Process all patients in batches
        for parent_nodes, child_nodes, batch_idx in processor.process_all_patients():
            # Save parent documents to disk
            parent_storage.save_batch(parent_nodes, batch_idx)
            
            # Save child nodes to JSONL
            child_indexer.save_child_batch(child_nodes)
        
        # Close child nodes file
        child_indexer.close_jsonl()
        
        logger.info(f"\n✅ Phase 1 Complete!")
        logger.info(f"Parent docs saved: {parent_storage.get_stats()['total_saved']}")
        logger.info(f"Child nodes saved: {child_indexer.get_stats()['child_nodes_saved']}")
        
        # --- PHASE 2: CHROMADB INDEXING ---
        logger.info(f"\n🔄 PHASE 2: Building ChromaDB index with LOCAL embeddings...")
        
        child_indexer.build_index()
        
        logger.info(f"\n✅ Phase 2 Complete!")
        logger.info(f"Child nodes indexed: {child_indexer.get_stats()['child_nodes_indexed']}")
        
        # --- FINAL SUMMARY ---
        logger.info("\n" + "="*80)
        logger.info(f"🎉 {hospital_config.hospital_id.upper()} PREPROCESSING COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nFINAL STATISTICS:")
        logger.info(f"  Patients processed: {processor.stats['total_patients']}")
        logger.info(f"  Encounters processed: {processor.stats['total_encounters']}")
        logger.info(f"  Parent docs saved: {parent_storage.get_stats()['total_saved']}")
        logger.info(f"  Child nodes indexed: {child_indexer.get_stats()['child_nodes_indexed']}")
        logger.info(f"  Devices included: {federated_config.INCLUDE_DEVICES}")
        logger.info(f"\n📁 Output Locations:")
        logger.info(f"  Parent documents: {hospital_config.parent_docs_dir}")
        logger.info(f"  ChromaDB index: {hospital_config.chromadb_dir}")
        logger.info(f"  Child nodes JSONL: {hospital_config.child_nodes_file}")
        logger.info(f"\n🤖 Model Configuration:")
        logger.info(f"  Embedding model: {federated_config.EMBEDDING_MODEL_PATH}")
        logger.info(f"  LLM model: {federated_config.LLM_MODEL_NAME}")
        logger.info("="*80)
        
        logger.info(f"\n✨ {hospital_config.hospital_id} ready for querying!")
        
    except Exception as e:
        logger.error(f"❌ Error processing {hospital_config.hospital_id}: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
