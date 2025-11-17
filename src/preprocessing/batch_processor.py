"""
Batch Processor - Memory-efficient processing of patient files
Updated to use config for device inclusion
"""
import gc
import logging
import psutil
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from ..utils import PatientDataLoader, NodeGenerator

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes patient files in memory-efficient batches"""
    
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 50,
        memory_threshold_gb: float = 12.0,
        gc_frequency: int = 10,
        include_devices: bool = True,
        max_devices_display: int = 20
    ):
        """
        Initialize batch processor
        
        Args:
            data_dir: Directory containing patient files
            batch_size: Number of patients per batch
            memory_threshold_gb: Pause processing if RAM exceeds this
            gc_frequency: Run garbage collection every N batches
            include_devices: Include patient devices in parent docs
            max_devices_display: Max devices to display
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.memory_threshold_gb = memory_threshold_gb
        self.gc_frequency = gc_frequency
        
        self.loader = PatientDataLoader(data_dir)
        self.generator = NodeGenerator(
            include_devices=include_devices,
            max_devices_display=max_devices_display
        )
        
        self.stats = {
            'total_patients': 0,
            'total_encounters': 0,
            'failed_patients': 0,
            'failed_encounters': 0
        }
    
    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    
    def check_memory(self):
        """Check memory usage and pause if necessary"""
        memory_gb = self.get_memory_usage_gb()
        if memory_gb > self.memory_threshold_gb:
            logger.warning(
                f"Memory usage ({memory_gb:.2f}GB) exceeds threshold "
                f"({self.memory_threshold_gb}GB). Running garbage collection..."
            )
            gc.collect()
            memory_gb = self.get_memory_usage_gb()
            logger.info(f"Memory after GC: {memory_gb:.2f}GB")
    
    def process_all_patients(self):
        """
        Process all patient files in batches
        
        Yields:
            Tuple of (parent_nodes, child_nodes, batch_idx) for each batch
        """
        # Get all patient files
        patient_files = self.loader.get_patient_files()
        total_files = len(patient_files)
        
        logger.info(f"Processing {total_files} patient files...")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Memory threshold: {self.memory_threshold_gb}GB")
        logger.info(f"Including devices: {self.generator.include_devices}")
        
        # Process in batches
        num_batches = (total_files + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(
            range(0, total_files, self.batch_size),
            total=num_batches,
            desc="Processing batches"
        ):
            # Get batch of files
            batch_files = patient_files[batch_idx:batch_idx + self.batch_size]
            
            # Load patients
            patients = self.loader.load_batch(batch_files)
            self.stats['total_patients'] += len(patients)
            self.stats['failed_patients'] += len(batch_files) - len(patients)
            
            # Process encounters
            parent_nodes = []
            child_nodes = []
            
            for patient_data in patients:
                try:
                    for encounter in patient_data.get('encounters', []):
                        try:
                            (
                                parent_id,
                                parent_content,
                                child_id,
                                child_summary,
                                child_metadata
                            ) = self.generator.process_encounter(
                                patient_data,
                                encounter
                            )
                            
                            parent_nodes.append({
                                'id': parent_id,
                                'content': parent_content,
                                'metadata': {
                                    'patient_id': patient_data['patient']['patient_id'],
                                    'encounter_id': encounter['encounter_id']
                                }
                            })
                            
                            child_nodes.append({
                                'id': child_id,
                                'summary': child_summary,
                                'metadata': child_metadata
                            })
                            
                            self.stats['total_encounters'] += 1
                            
                        except Exception as e:
                            logger.error(
                                f"Error processing encounter "
                                f"{encounter.get('encounter_id', 'unknown')}: {e}"
                            )
                            self.stats['failed_encounters'] += 1
                            continue
                
                except Exception as e:
                    logger.error(f"Error processing patient: {e}")
                    continue
            
            # Check memory and garbage collect if needed
            if batch_idx % (self.batch_size * self.gc_frequency) == 0:
                self.check_memory()
                gc.collect()
            
            # Yield batch results
            yield parent_nodes, child_nodes, batch_idx // self.batch_size
            
            # Clear batch data from memory
            del patients, parent_nodes, child_nodes
        
        # Final stats
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Total patients processed: {self.stats['total_patients']}")
        logger.info(f"Total encounters processed: {self.stats['total_encounters']}")
        logger.info(f"Failed patients: {self.stats['failed_patients']}")
        logger.info(f"Failed encounters: {self.stats['failed_encounters']}")
        logger.info("="*60)
