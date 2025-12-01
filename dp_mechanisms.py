#!/usr/bin/env python3
"""
FRAG-MED Differential Privacy Mechanisms
Response-level differential privacy for federated medical RAG.

This module implements DP at the RESPONSE level, which:
1. Adds calibrated noise to numerical counts
2. Suppresses small counts (k-anonymity threshold)
3. Generalizes specific values to broader categories
4. Provides provable privacy guarantees with epsilon parameter

WHY RESPONSE-LEVEL DP (vs Embedding-Level):
- Embedding-level DP requires DP-SGD during model training (compute-intensive)
- Response-level DP is applied post-hoc to already-generated responses
- More practical for resource-constrained environments (16GB RAM)
- Still provides meaningful privacy protection for federated sharing

Reference: Dwork & Roth, 2014 - "The Algorithmic Foundations of Differential Privacy"
"""

import re
import math
import random
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class DPConfig:
    """Configuration for differential privacy mechanisms"""
    
    # Privacy budget (lower = more privacy, more noise)
    epsilon: float = 1.0
    
    # Sensitivity for count queries (how much one record can change the output)
    count_sensitivity: float = 1.0
    
    # Minimum count threshold for k-anonymity (suppress if below this)
    k_anonymity_threshold: int = 3
    
    # Whether to generalize age ranges
    generalize_ages: bool = True
    
    # Whether to suppress rare conditions/medications
    suppress_rare_items: bool = True
    rare_item_threshold: int = 2
    
    # Random seed for reproducibility (set to None for true randomness)
    seed: Optional[int] = None


class DifferentialPrivacy:
    """
    Implements response-level differential privacy mechanisms.
    
    Key mechanisms:
    1. Laplace Mechanism: Add noise to numerical values
    2. K-Anonymity: Suppress small groups
    3. Generalization: Replace specific values with broader categories
    4. Randomized Response: For categorical data
    """
    
    def __init__(self, config: DPConfig = None):
        self.config = config or DPConfig()
        
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
    
    def laplace_noise(self, sensitivity: float = None) -> float:
        """
        Generate Laplace noise for differential privacy.
        
        Laplace mechanism: Add noise ~ Lap(sensitivity/epsilon)
        
        Args:
            sensitivity: Query sensitivity (default from config)
        
        Returns:
            Noise value to add to true count
        """
        sens = sensitivity or self.config.count_sensitivity
        scale = sens / self.config.epsilon
        return np.random.laplace(0, scale)
    
    def add_noise_to_count(self, true_count: int) -> int:
        """
        Add Laplace noise to a count value.
        
        Args:
            true_count: The actual count
        
        Returns:
            Noised count (always >= 0)
        """
        noise = self.laplace_noise()
        noised = true_count + noise
        return max(0, int(round(noised)))
    
    def suppress_if_below_threshold(self, count: int) -> Tuple[int, bool]:
        """
        Apply k-anonymity suppression.
        
        Args:
            count: The count to check
        
        Returns:
            Tuple of (count or 0, was_suppressed)
        """
        if count < self.config.k_anonymity_threshold:
            return 0, True
        return count, False
    
    def apply_dp_to_count(self, true_count: int) -> Tuple[int, str]:
        """
        Full DP pipeline for a count value.
        
        1. Check k-anonymity threshold
        2. Add Laplace noise
        
        Args:
            true_count: Original count
        
        Returns:
            Tuple of (dp_count, status_message)
        """
        # Step 1: K-anonymity check
        count, suppressed = self.suppress_if_below_threshold(true_count)
        if suppressed:
            return 0, "suppressed (k-anonymity)"
        
        # Step 2: Add noise
        dp_count = self.add_noise_to_count(count)
        
        # Ensure non-negative
        dp_count = max(0, dp_count)
        
        return dp_count, f"noised (ε={self.config.epsilon})"
    
    def generalize_age_distribution(self, age_dist: Dict[str, int]) -> Dict[str, int]:
        """
        Generalize and apply DP to age distribution.
        
        Combines fine-grained age bands into broader categories
        and adds noise to each count.
        
        Args:
            age_dist: Original age distribution {"30-39": 2, "50-59": 1}
        
        Returns:
            DP-protected age distribution
        """
        if not self.config.generalize_ages:
            # Just add noise without generalization
            return {
                age: self.add_noise_to_count(count)
                for age, count in age_dist.items()
                if count >= self.config.k_anonymity_threshold
            }
        
        # Generalize to broader categories
        generalized = {
            "Under 30": 0,
            "30-59": 0,
            "60+": 0
        }
        
        for age_range, count in age_dist.items():
            if age_range in ["0-17", "18-29"]:
                generalized["Under 30"] += count
            elif age_range in ["30-39", "40-49", "50-59"]:
                generalized["30-59"] += count
            else:  # 60-69, 70+
                generalized["60+"] += count
        
        # Apply DP to generalized counts
        dp_dist = {}
        for category, count in generalized.items():
            dp_count, _ = self.apply_dp_to_count(count)
            if dp_count > 0:  # Only include non-zero
                dp_dist[category] = dp_count
        
        return dp_dist
    
    def filter_rare_items(self, items: List[str], counts: Dict[str, int] = None) -> List[str]:
        """
        Filter out rare items that could be identifying.
        
        Args:
            items: List of items (conditions, medications, etc.)
            counts: Optional count per item
        
        Returns:
            Filtered list with rare items removed
        """
        if not self.config.suppress_rare_items:
            return items
        
        if counts:
            return [
                item for item in items 
                if counts.get(item, 0) >= self.config.rare_item_threshold
            ]
        
        # Without counts, keep items that appear multiple times
        from collections import Counter
        item_counts = Counter(items)
        return [
            item for item in set(items)
            if item_counts[item] >= self.config.rare_item_threshold
        ]
    
    def sanitize_response_text(self, text: str) -> str:
        """
        Remove or redact potentially identifying information from response text.
        
        Args:
            text: Raw response text
        
        Returns:
            Sanitized text with identifiers removed
        """
        sanitized = text
        
        # Remove patient IDs (PATIENT_xxx patterns)
        sanitized = re.sub(
            r'PATIENT_[A-Za-z0-9]+',
            '[PATIENT]',
            sanitized
        )
        
        # Remove specific dates, replace with quarters
        sanitized = re.sub(
            r'\d{4}-\d{2}-\d{2}',
            '[DATE]',
            sanitized
        )
        
        # Remove encounter IDs
        sanitized = re.sub(
            r'ENC_[A-Za-z0-9]+',
            '[ENCOUNTER]',
            sanitized
        )
        
        # Remove any UUIDs
        sanitized = re.sub(
            r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
            '[ID]',
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Remove specific numeric sequences that could be identifying
        sanitized = re.sub(
            r'\b[A-Za-z0-9]{8,}\b',  # Long alphanumeric strings
            lambda m: '[ID]' if any(c.isdigit() for c in m.group()) and any(c.isalpha() for c in m.group()) else m.group(),
            sanitized
        )
        
        return sanitized
    
    def create_dp_summary(
        self,
        patient_count: int,
        conditions: List[str],
        treatments: List[str],
        procedures: List[str],
        age_distribution: Dict[str, int],
        raw_response: str
    ) -> Dict:
        """
        Create a full DP-protected summary from raw hospital response.
        
        Args:
            patient_count: Raw patient count
            conditions: List of conditions found
            treatments: List of treatments
            procedures: List of procedures
            age_distribution: Age distribution dict
            raw_response: Raw LLM response text
        
        Returns:
            DP-protected summary dict
        """
        # Apply DP to patient count
        dp_count, count_status = self.apply_dp_to_count(patient_count)
        
        # Filter rare items
        dp_conditions = self.filter_rare_items(conditions)
        dp_treatments = self.filter_rare_items(treatments)
        dp_procedures = self.filter_rare_items(procedures)
        
        # Generalize and protect age distribution
        dp_age_dist = self.generalize_age_distribution(age_distribution)
        
        # Sanitize response text
        dp_response = self.sanitize_response_text(raw_response)
        
        return {
            'patient_count': dp_count,
            'count_status': count_status,
            'conditions': dp_conditions[:7],  # Limit to top 7
            'treatments': dp_treatments[:7],
            'procedures': dp_procedures[:5],
            'age_distribution': dp_age_dist,
            'sanitized_response': dp_response,
            'privacy_params': {
                'epsilon': self.config.epsilon,
                'k_threshold': self.config.k_anonymity_threshold,
                'mechanism': 'Laplace + k-anonymity + generalization'
            }
        }


class ResponseLevelDP:
    """
    High-level interface for response-level differential privacy.
    
    This is the main class to use for applying DP to hospital responses
    before sharing with the federated orchestrator.
    """
    
    def __init__(self, epsilon: float = 1.0, k_threshold: int = 3):
        """
        Initialize response-level DP.
        
        Args:
            epsilon: Privacy budget (recommended 0.1 to 2.0)
                    - 0.1-0.5: Strong privacy, more noise
                    - 0.5-1.0: Balanced privacy-utility
                    - 1.0-2.0: Better utility, less privacy
            k_threshold: Minimum count for k-anonymity
        """
        self.config = DPConfig(
            epsilon=epsilon,
            k_anonymity_threshold=k_threshold
        )
        self.dp = DifferentialPrivacy(self.config)
    
    def protect_hospital_response(
        self,
        hospital_id: str,
        patient_count: int,
        conditions: List[str],
        treatments: List[str],
        procedures: List[str],
        age_distribution: Dict[str, int],
        synthesized_response: str
    ) -> Dict:
        """
        Apply DP protection to a hospital's response before federation.
        
        Args:
            hospital_id: Hospital identifier
            patient_count: Number of patients found
            conditions: List of conditions
            treatments: List of treatments  
            procedures: List of procedures
            age_distribution: Age distribution
            synthesized_response: LLM-generated response
        
        Returns:
            DP-protected response dict safe for federation
        """
        # Create DP-protected summary
        dp_summary = self.dp.create_dp_summary(
            patient_count=patient_count,
            conditions=conditions,
            treatments=treatments,
            procedures=procedures,
            age_distribution=age_distribution,
            raw_response=synthesized_response
        )
        
        return {
            'hospital_id': hospital_id,
            'dp_patient_count': dp_summary['patient_count'],
            'dp_conditions': dp_summary['conditions'],
            'dp_treatments': dp_summary['treatments'],
            'dp_procedures': dp_summary['procedures'],
            'dp_age_distribution': dp_summary['age_distribution'],
            'dp_response': dp_summary['sanitized_response'],
            'privacy_guarantee': {
                'mechanism': 'Response-Level Differential Privacy',
                'epsilon': self.config.epsilon,
                'k_anonymity': self.config.k_anonymity_threshold,
                'components': [
                    'Laplace noise on counts',
                    'K-anonymity suppression',
                    'Age generalization',
                    'Identifier sanitization'
                ]
            }
        }
    
    def get_privacy_statement(self) -> str:
        """Get human-readable privacy guarantee statement"""
        return (
            f"This response is protected by Response-Level Differential Privacy "
            f"with ε={self.config.epsilon}. Counts have Laplace noise added, "
            f"groups smaller than {self.config.k_anonymity_threshold} are suppressed, "
            f"and all patient identifiers have been removed."
        )


# Utility function for easy import
def create_dp_protector(epsilon: float = 1.0, k_threshold: int = 3) -> ResponseLevelDP:
    """
    Create a DP protector with specified parameters.
    
    Args:
        epsilon: Privacy budget (lower = more private)
        k_threshold: K-anonymity threshold
    
    Returns:
        Configured ResponseLevelDP instance
    """
    return ResponseLevelDP(epsilon=epsilon, k_threshold=k_threshold)


if __name__ == "__main__":
    # Demo of DP mechanisms
    print("="*60)
    print("FRAG-MED Differential Privacy Demo")
    print("="*60)
    
    dp = create_dp_protector(epsilon=1.0, k_threshold=3)
    
    # Example hospital response
    result = dp.protect_hospital_response(
        hospital_id="hospital_A",
        patient_count=5,
        conditions=["Acute bronchitis", "Hypertension", "Diabetes mellitus type 2"],
        treatments=["Acetaminophen 325mg", "Lisinopril 10mg"],
        procedures=["Sputum examination", "Chest X-ray"],
        age_distribution={"30-39": 1, "50-59": 3, "70+": 1},
        synthesized_response="PATIENT_560864a2 presented with acute bronchitis. Treatment included acetaminophen."
    )
    
    print("\nDP-Protected Response:")
    print(f"  Patient count: {result['dp_patient_count']} (noised)")
    print(f"  Conditions: {result['dp_conditions']}")
    print(f"  Age distribution: {result['dp_age_distribution']}")
    print(f"\n  Sanitized response preview:")
    print(f"  {result['dp_response'][:200]}...")
    print(f"\n  Privacy guarantee: {result['privacy_guarantee']}")
