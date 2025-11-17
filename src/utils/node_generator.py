"""
Node Generator - Creates parent and child nodes from patient encounters
Updated to include patient_level_devices
"""
import logging
from typing import Dict, Tuple, List
from .deidentification import DeIdentifier

logger = logging.getLogger(__name__)


class NodeGenerator:
    """Generates parent and child nodes for RAG system"""
    
    def __init__(self, include_devices: bool = True, max_devices_display: int = 20):
        """
        Initialize node generator
        
        Args:
            include_devices: Whether to include patient devices in parent docs
            max_devices_display: Maximum number of devices to display
        """
        self.deidentifier = DeIdentifier()
        self.include_devices = include_devices
        self.max_devices_display = max_devices_display
    
    def process_encounter(
        self,
        patient_data: Dict,
        encounter_data: Dict
    ) -> Tuple[str, str, str, str, Dict]:
        """
        Process a single encounter into parent-child nodes
        
        Args:
            patient_data: Full patient data dictionary
            encounter_data: Single encounter dictionary
            
        Returns:
            Tuple of (parent_doc_id, parent_content, child_doc_id, 
                     child_summary, child_metadata)
        """
        try:
            # Extract patient info
            patient = patient_data['patient']
            patient_pseudonym = self.deidentifier.create_pseudonym(
                patient['patient_id']
            )
            
            # Calculate age and temporal info
            encounter_date = encounter_data['period']['start']
            age = self.deidentifier.calculate_age(
                patient['birthDate'],
                encounter_date
            )
            age_band = self.deidentifier.get_age_band(age)
            temporal_quarter = self.deidentifier.get_quarter(encounter_date)
            
            # De-identify clinical notes
            raw_notes = "\n".join(encounter_data.get('clinical_notes', []))
            clean_notes = self.deidentifier.scrub_clinical_notes(
                raw_notes,
                patient.get('name', ''),
                patient_pseudonym,
                age,
                age_band,
                encounter_date,
                temporal_quarter
            )
            
            # Extract structured data
            conditions = [
                c['display_text'] 
                for c in encounter_data.get('conditions', [])
            ]
            procedures = [
                p['display_text'] 
                for p in encounter_data.get('procedures', [])
            ]
            medications = [
                m['display_text'] 
                for m in encounter_data.get('medications', [])
            ]
            observations = encounter_data.get('observations', [])
            
            # Extract patient-level devices
            devices = patient_data.get('patient_level_devices', [])
            
            # Get encounter type
            encounter_types = encounter_data.get('type', [])
            encounter_type = (
                encounter_types[0]['display_text'] 
                if encounter_types else "Unknown"
            )
            
            # --- BUILD PARENT NODE (Full Context) ---
            parent_doc_id = (
                f"{patient_pseudonym}_ENC_"
                f"{encounter_data['encounter_id'].split(':')[-1]}"
            )
            
            parent_content = self._build_parent_content(
                patient_pseudonym=patient_pseudonym,
                age_band=age_band,
                gender=patient.get('gender', 'Unknown'),
                race=patient.get('race', 'Unknown'),
                ethnicity=patient.get('ethnicity', 'Unknown'),
                marital_status=patient.get('maritalStatus', 'Unknown'),
                encounter_type=encounter_type,
                temporal_quarter=temporal_quarter,
                status=encounter_data.get('status', 'Unknown'),
                clean_notes=clean_notes,
                conditions=conditions,
                procedures=procedures,
                medications=medications,
                observations=observations,
                devices=devices  # Include devices
            )
            
            # --- BUILD CHILD NODE (Dense Summary) ---
            child_doc_id = f"CHILD_{parent_doc_id}"
            
            child_summary = self._build_child_summary(
                patient_pseudonym=patient_pseudonym,
                age_band=age_band,
                gender=patient.get('gender', 'Unknown'),
                race=patient.get('race', 'Unknown'),
                temporal_quarter=temporal_quarter,
                encounter_type=encounter_type,
                conditions=conditions[:5],  # Top 5 conditions
                procedures=procedures[:3],   # Top 3 procedures
                devices=devices[:3]  # Top 3 devices for summary
            )
            
            # --- BUILD METADATA ---
            child_metadata = {
                "parent_doc_id": parent_doc_id,
                "doc_type": "encounter_summary",
                "patient_pseudonym": patient_pseudonym,
                "age_range": age_band,
                "gender": patient.get('gender', 'Unknown'),
                "race": patient.get('race', 'Unknown'),
                "ethnicity": patient.get('ethnicity', 'Unknown'),
                "temporal_quarter": temporal_quarter,
                "encounter_type": encounter_type,
                "num_conditions": len(conditions),
                "num_procedures": len(procedures),
                "num_medications": len(medications),
                "num_observations": len(observations),
                "num_devices": len(devices)
            }
            
            return (
                parent_doc_id,
                parent_content,
                child_doc_id,
                child_summary,
                child_metadata
            )
            
        except Exception as e:
            logger.error(f"Error processing encounter: {e}")
            raise
    
    def _build_parent_content(self, **kwargs) -> str:
        """Build formatted parent document content"""
        
        # Format observations (limit to first 20 for readability)
        obs_text = "\n".join(
            f"- {o['display_text']}: {o.get('value', 'N/A')} "
            f"{o.get('unit', '')}"
            for o in kwargs['observations'][:20]
        ) if kwargs['observations'] else "None"
        
        # Format devices (with deduplication)
        devices_section = ""
        if self.include_devices and kwargs.get('devices'):
            devices = kwargs['devices']
            
            # Deduplicate devices by display_text
            unique_devices = []
            seen = set()
            for device in devices:
                device_name = device.get('display_text', device.get('name', 'Unknown'))
                if device_name not in seen:
                    seen.add(device_name)
                    unique_devices.append(device)
            
            # Limit display
            devices_to_show = unique_devices[:self.max_devices_display]
            
            devices_text = "\n".join(
                f"- {d.get('display_text', d.get('name', 'Unknown'))} "
                f"(Code: {d.get('code', 'N/A')}, Status: {d.get('status', 'Unknown')})"
                for d in devices_to_show
            )
            
            if len(unique_devices) > self.max_devices_display:
                devices_text += f"\n... and {len(unique_devices) - self.max_devices_display} more devices"
            
            devices_section = f"""
=== PATIENT-LEVEL DEVICES ===
{devices_text}
"""
        
        return f"""=== PATIENT DEMOGRAPHICS ===
Age: {kwargs['age_band']}
Gender: {kwargs['gender']}
Race/Ethnicity: {kwargs['race']}, {kwargs['ethnicity']}
Marital Status: {kwargs['marital_status']}
{devices_section}
=== ENCOUNTER INFORMATION ===
Type: {kwargs['encounter_type']}
Date: {kwargs['temporal_quarter']}
Status: {kwargs['status']}

=== CLINICAL SUMMARY ===
{kwargs['clean_notes'] if kwargs['clean_notes'] else 'No clinical notes available.'}

=== CONDITIONS DOCUMENTED ===
{chr(10).join(f"- {c}" for c in kwargs['conditions']) if kwargs['conditions'] else "None"}

=== PROCEDURES PERFORMED ===
{chr(10).join(f"- {p}" for p in kwargs['procedures']) if kwargs['procedures'] else "None"}

=== MEDICATIONS ===
{chr(10).join(f"- {m}" for m in kwargs['medications']) if kwargs['medications'] else "None"}

=== OBSERVATIONS ===
{obs_text}
"""
    
    def _build_child_summary(self, **kwargs) -> str:
        """Build compact child node summary for embedding"""
        
        conditions_str = (
            ', '.join(kwargs['conditions']) 
            if kwargs['conditions'] 
            else 'None documented'
        )
        
        procedures_str = (
            ', '.join(kwargs['procedures']) 
            if kwargs['procedures'] 
            else 'None performed'
        )
        
        # Include top devices in summary
        devices_str = ""
        if kwargs.get('devices'):
            device_names = [
                d.get('display_text', d.get('name', 'Unknown'))
                for d in kwargs['devices']
            ]
            if device_names:
                devices_str = f" Medical devices: {', '.join(device_names)}."
        
        return (
            f"Patient: {kwargs['patient_pseudonym']}, "
            f"{kwargs['age_band']}, {kwargs['gender']}, "
            f"{kwargs['race']}. "
            f"Time: {kwargs['temporal_quarter']}. "
            f"Encounter type: {kwargs['encounter_type']}. "
            f"Primary conditions: {conditions_str}. "
            f"Key procedures: {procedures_str}."
            f"{devices_str}"
        )
