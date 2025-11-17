"""
Patient Data Loader - Handles loading and validation of patient JSON files
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PatientDataLoader:
    """Loads and validates patient data files"""
    
    def __init__(self, data_dir: Path):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing patient JSON files
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def get_patient_files(self, pattern: str = "*.json") -> List[Path]:
        """
        Get all patient files in directory
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of patient file paths
        """
        files = sorted(self.data_dir.glob(pattern))
        logger.info(f"Found {len(files)} patient files in {self.data_dir}")
        return files
    
    def load_patient(self, file_path: Path) -> Optional[Dict]:
        """
        Load a single patient file
        
        Args:
            file_path: Path to patient JSON file
            
        Returns:
            Patient data dictionary or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Basic validation
            if 'patient' not in data:
                logger.warning(f"Missing 'patient' key in {file_path.name}")
                return None
            
            if 'encounters' not in data or not data['encounters']:
                logger.warning(f"No encounters found in {file_path.name}")
                return None
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            return None
    
    def load_batch(self, file_paths: List[Path]) -> List[Dict]:
        """
        Load a batch of patient files
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of successfully loaded patient data
        """
        patients = []
        for file_path in file_paths:
            patient_data = self.load_patient(file_path)
            if patient_data:
                patients.append(patient_data)
        
        return patients
    
    def validate_patient_structure(self, patient_data: Dict) -> bool:
        """
        Validate patient data structure
        
        Args:
            patient_data: Patient data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_patient_fields = ['patient_id', 'birthDate', 'gender']
        required_encounter_fields = ['encounter_id', 'period', 'type']
        
        # Check patient fields
        patient = patient_data.get('patient', {})
        if not all(field in patient for field in required_patient_fields):
            return False
        
        # Check encounters
        encounters = patient_data.get('encounters', [])
        for encounter in encounters:
            if not all(field in encounter for field in required_encounter_fields):
                return False
        
        return True
