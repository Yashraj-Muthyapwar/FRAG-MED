"""
De-identification Module - Removes PII and applies privacy-preserving transformations
Enhanced to handle partial names and better pattern matching
"""
import re
import hashlib
from datetime import datetime
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class DeIdentifier:
    """Handles de-identification of patient data"""
    
    @staticmethod
    def get_age_band(age: int) -> str:
        """
        Convert exact age to age band
        
        Args:
            age: Exact age in years
            
        Returns:
            Age band string (e.g., "50-59")
        """
        if age < 18:
            return "0-17"
        elif 18 <= age <= 29:
            return "18-29"
        elif 30 <= age <= 39:
            return "30-39"
        elif 40 <= age <= 49:
            return "40-49"
        elif 50 <= age <= 59:
            return "50-59"
        elif 60 <= age <= 69:
            return "60-69"
        else:
            return "70+"
    
    @staticmethod
    def get_quarter(date_str: str) -> str:
        """
        Convert specific date to year-quarter
        
        Args:
            date_str: ISO format date string
            
        Returns:
            Quarter string (e.g., "2025-Q2")
        """
        try:
            # Handle ISO format with timezone
            date_str_clean = date_str.split('+')[0].split('T')[0]
            date_obj = datetime.strptime(date_str_clean, "%Y-%m-%d")
            quarter = (date_obj.month - 1) // 3 + 1
            return f"{date_obj.year}-Q{quarter}"
        except Exception as e:
            logger.warning(f"Error parsing date {date_str}: {e}")
            return "UNKNOWN-Q0"
    
    @staticmethod
    def create_pseudonym(patient_id: str) -> str:
        """
        Create deterministic pseudonym from patient ID
        
        Args:
            patient_id: Original patient identifier
            
        Returns:
            Pseudonymized identifier
        """
        # Use first 8 characters of patient ID for consistency
        return f"PATIENT_{patient_id[:8]}"
    
    @staticmethod
    def calculate_age(birth_date: str, encounter_date: str) -> int:
        """
        Calculate age at time of encounter
        
        Args:
            birth_date: Birth date string (YYYY-MM-DD)
            encounter_date: Encounter date string (ISO format)
            
        Returns:
            Age in years
        """
        try:
            dob = datetime.strptime(birth_date, "%Y-%m-%d")
            enc_date_str = encounter_date.split('+')[0].split('T')[0]
            enc_date = datetime.strptime(enc_date_str, "%Y-%m-%d")
            
            age = enc_date.year - dob.year
            if (enc_date.month, enc_date.day) < (dob.month, dob.day):
                age -= 1
            
            return age
        except Exception as e:
            logger.warning(f"Error calculating age: {e}")
            return 0
    
    @staticmethod
    def extract_name_parts(full_name: str) -> List[str]:
        """
        Extract all parts of a name for comprehensive scrubbing
        
        Args:
            full_name: Full patient name (e.g., "Adelia946 Roob298 Collier329")
            
        Returns:
            List of name parts to scrub
        """
        if not full_name:
            return []
        
        # Split by spaces and filter out empty strings
        parts = [part.strip() for part in full_name.split() if part.strip()]
        
        # Also create variations without numbers (e.g., "Adelia946" -> "Adelia")
        variations = []
        for part in parts:
            variations.append(part)
            # Remove trailing numbers
            clean_part = re.sub(r'\d+$', '', part)
            if clean_part and clean_part != part:
                variations.append(clean_part)
        
        return list(set(variations))  # Remove duplicates
    
    @staticmethod
    def scrub_clinical_notes(
        text: str,
        patient_name: str,
        pseudonym: str,
        age: int,
        age_band: str,
        encounter_date: str,
        quarter: str
    ) -> str:
        """
        Remove PII from clinical notes with comprehensive name scrubbing
        
        Args:
            text: Original clinical text
            patient_name: Patient's real name
            pseudonym: Patient pseudonym
            age: Exact age
            age_band: Age band
            encounter_date: Original date
            quarter: Quarter representation
            
        Returns:
            Scrubbed text
        """
        if not text:
            return ""
        
        # Extract all name parts for comprehensive replacement
        name_parts = DeIdentifier.extract_name_parts(patient_name)
        
        # Replace each name part (case-insensitive, word boundary)
        for name_part in name_parts:
            if len(name_part) >= 3:  # Only replace parts with 3+ characters
                # Use word boundary regex for exact matches
                pattern = r'\b' + re.escape(name_part) + r'\b'
                text = re.sub(pattern, pseudonym, text, flags=re.IGNORECASE)
        
        # Also replace the full name
        if patient_name:
            text = text.replace(patient_name, pseudonym)
        
        # Replace age mentions (multiple patterns)
        age_patterns = [
            rf'\b{age}\s*year-?old\b',
            rf'\b{age}\s*y/?o\b',
            rf'\b{age}\s*years?\b',
        ]
        for pattern in age_patterns:
            text = re.sub(pattern, f'{age_band} year-old', text, flags=re.IGNORECASE)
        
        # Replace specific dates with quarters
        # Handle multiple date formats
        date_short = encounter_date.split('T')[0]  # YYYY-MM-DD
        text = text.replace(date_short, quarter)
        
        # Also handle other common date formats
        try:
            date_obj = datetime.strptime(date_short, "%Y-%m-%d")
            # MM/DD/YYYY format
            us_format = date_obj.strftime("%m/%d/%Y")
            text = text.replace(us_format, quarter)
            # Month DD, YYYY format
            long_format = date_obj.strftime("%B %d, %Y")
            text = text.replace(long_format, quarter)
        except:
            pass
        
        # Additional PII patterns
        
        # SSN pattern
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN-REDACTED]', text)
        
        # Phone numbers (multiple formats)
        phone_patterns = [
            r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        ]
        for pattern in phone_patterns:
            text = re.sub(pattern, '[PHONE-REDACTED]', text)
        
        # Email addresses
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL-REDACTED]',
            text
        )
        
        # Addresses (comprehensive patterns)
        address_patterns = [
            # Street addresses
            r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Place|Pl)\b',
            # Apartment/Unit numbers
            r'\b(?:Apt|Apartment|Unit|Suite|Ste)\s*\.?\s*#?\s*\d+\w*\b',
            # PO Boxes
            r'\bP\.?O\.?\s*Box\s+\d+\b',
        ]
        for pattern in address_patterns:
            text = re.sub(pattern, '[ADDRESS-REDACTED]', text, flags=re.IGNORECASE)
        
        # Zip codes
        text = re.sub(r'\b\d{5}(?:-\d{4})?\b', '[ZIP-REDACTED]', text)
        
        # Long numeric IDs (10+ digits)
        text = re.sub(r'\b\d{10,}\b', '[ID-REDACTED]', text)
        
        # Medical Record Numbers (MRN)
        text = re.sub(r'\bMRN:?\s*\d+\b', '[MRN-REDACTED]', text, flags=re.IGNORECASE)
        
        return text
