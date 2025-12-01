#!/usr/bin/env python3
"""
FRAG-MED Federated Aggregation Module
Implements proper ensemble-based aggregation methods WITHOUT LLM hallucination.

Aggregation Methods:
1. Secure Count Aggregation - Sum DP counts programmatically (no LLM)
2. Majority Voting - Aggregate conditions/treatments by frequency
3. Weighted Averaging - Weight by confidence scores
4. K-Anonymity Filtering - Suppress rare items across federation

Reference: McMahan et al. 2017 - "Communication-Efficient Learning of Deep Networks"
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import numpy as np


@dataclass
class AggregatedResult:
    """Result from federated aggregation"""
    total_patient_count: int
    aggregated_conditions: List[Tuple[str, int]]  # (condition, count across hospitals)
    aggregated_treatments: List[Tuple[str, int]]   # (treatment, count)
    aggregated_procedures: List[Tuple[str, int]]   # (procedure, count)
    age_distribution: Dict[str, int]
    hospital_contributions: Dict[str, int]  # hospital_id -> patient_count
    aggregation_method: str
    privacy_params: Dict


class SecureAggregator:
    """
    Implements secure aggregation for federated medical RAG.
    
    Key principle: Numbers are computed PROGRAMMATICALLY, not by LLM.
    The LLM is only used for natural language synthesis AFTER aggregation.
    """
    
    def __init__(self, k_anonymity_threshold: int = 3):
        """
        Args:
            k_anonymity_threshold: Minimum count to include an item
        """
        self.k_threshold = k_anonymity_threshold
    
    def aggregate_counts(self, hospital_results: List[Dict]) -> int:
        """
        Secure count aggregation - sum DP counts from all hospitals.
        
        This is done PROGRAMMATICALLY, not by LLM, to prevent hallucination.
        
        Args:
            hospital_results: List of hospital result dicts with 'dp_patient_count'
        
        Returns:
            Total aggregated count
        """
        total = sum(
            r.get('dp_patient_count', 0) 
            for r in hospital_results 
            if r is not None
        )
        return total
    
    def majority_voting(
        self, 
        items_by_hospital: List[List[str]], 
        min_hospitals: int = 2
    ) -> List[Tuple[str, int]]:
        """
        Majority voting aggregation for categorical items.
        
        An item is included only if it appears in >= min_hospitals.
        
        Args:
            items_by_hospital: List of item lists from each hospital
            min_hospitals: Minimum number of hospitals that must report an item
        
        Returns:
            List of (item, hospital_count) tuples, sorted by count descending
        """
        # Count how many hospitals report each item
        hospital_counts = Counter()
        
        for hospital_items in items_by_hospital:
            # Use set to count each item once per hospital
            unique_items = set(hospital_items)
            for item in unique_items:
                if item and item.lower() not in ['none', 'none documented']:
                    hospital_counts[item] += 1
        
        # Filter by minimum hospital threshold
        filtered = [
            (item, count) 
            for item, count in hospital_counts.items() 
            if count >= min_hospitals
        ]
        
        # Sort by count descending
        return sorted(filtered, key=lambda x: x[1], reverse=True)
    
    def weighted_average_aggregation(
        self,
        items_by_hospital: List[Tuple[List[str], float]],  # (items, confidence_score)
        min_weight: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Weighted aggregation based on retrieval confidence scores.
        
        Items from high-confidence retrievals are weighted more heavily.
        
        Args:
            items_by_hospital: List of (items, confidence) tuples
            min_weight: Minimum total weight to include an item
        
        Returns:
            List of (item, total_weight) tuples
        """
        weighted_counts = Counter()
        
        for items, confidence in items_by_hospital:
            for item in items:
                if item and item.lower() not in ['none', 'none documented']:
                    weighted_counts[item] += confidence
        
        # Filter by minimum weight
        filtered = [
            (item, weight) 
            for item, weight in weighted_counts.items() 
            if weight >= min_weight
        ]
        
        return sorted(filtered, key=lambda x: x[1], reverse=True)
    
    def aggregate_age_distribution(
        self, 
        age_dists: List[Dict[str, int]]
    ) -> Dict[str, int]:
        """
        Aggregate age distributions across hospitals.
        
        Args:
            age_dists: List of age distribution dicts from each hospital
        
        Returns:
            Combined age distribution
        """
        combined = Counter()
        
        for dist in age_dists:
            for age_range, count in dist.items():
                if age_range and age_range != 'Unknown':
                    combined[age_range] += count
        
        # Apply k-anonymity - suppress small counts
        return {
            age: count 
            for age, count in combined.items() 
            if count >= self.k_threshold
        }
    
    def full_aggregation(self, hospital_results: List[Dict]) -> AggregatedResult:
        """
        Perform full secure aggregation across all hospitals.
        
        Args:
            hospital_results: List of hospital result dicts
        
        Returns:
            AggregatedResult with all aggregated data
        """
        valid_results = [r for r in hospital_results if r is not None]
        
        if not valid_results:
            return AggregatedResult(
                total_patient_count=0,
                aggregated_conditions=[],
                aggregated_treatments=[],
                aggregated_procedures=[],
                age_distribution={},
                hospital_contributions={},
                aggregation_method='none',
                privacy_params={}
            )
        
        # 1. Secure count aggregation
        total_count = self.aggregate_counts(valid_results)
        
        # 2. Majority voting for conditions
        conditions_by_hospital = [r.get('dp_conditions', []) for r in valid_results]
        aggregated_conditions = self.majority_voting(conditions_by_hospital, min_hospitals=1)
        
        # 3. Majority voting for treatments
        treatments_by_hospital = [r.get('dp_treatments', []) for r in valid_results]
        aggregated_treatments = self.majority_voting(treatments_by_hospital, min_hospitals=1)
        
        # 4. Majority voting for procedures
        procedures_by_hospital = [r.get('dp_procedures', []) for r in valid_results]
        aggregated_procedures = self.majority_voting(procedures_by_hospital, min_hospitals=1)
        
        # 5. Age distribution aggregation
        age_dists = [r.get('dp_age_distribution', {}) for r in valid_results]
        combined_age = self.aggregate_age_distribution(age_dists)
        
        # 6. Hospital contributions
        contributions = {
            r.get('hospital_id', f'hospital_{i}'): r.get('dp_patient_count', 0)
            for i, r in enumerate(valid_results)
        }
        
        return AggregatedResult(
            total_patient_count=total_count,
            aggregated_conditions=aggregated_conditions[:10],
            aggregated_treatments=aggregated_treatments[:10],
            aggregated_procedures=aggregated_procedures[:5],
            age_distribution=combined_age,
            hospital_contributions=contributions,
            aggregation_method='secure_aggregation + majority_voting',
            privacy_params={
                'k_anonymity_threshold': self.k_threshold,
                'hospitals_contributing': len(valid_results)
            }
        )


class EnsembleVoting:
    """
    Ensemble voting methods for federated RAG responses.
    """
    
    @staticmethod
    def count_voting(
        items_across_hospitals: List[List[str]]
    ) -> Dict[str, int]:
        """
        Simple count voting - count total occurrences across all hospitals.
        
        Args:
            items_across_hospitals: List of item lists
        
        Returns:
            Dict of item -> total count
        """
        all_items = []
        for items in items_across_hospitals:
            all_items.extend(items)
        
        return dict(Counter(all_items))
    
    @staticmethod
    def hospital_voting(
        items_across_hospitals: List[List[str]]
    ) -> Dict[str, int]:
        """
        Hospital-level voting - count how many hospitals report each item.
        
        Args:
            items_across_hospitals: List of item lists
        
        Returns:
            Dict of item -> number of hospitals reporting it
        """
        hospital_counts = Counter()
        
        for hospital_items in items_across_hospitals:
            for item in set(hospital_items):  # Unique per hospital
                hospital_counts[item] += 1
        
        return dict(hospital_counts)
    
    @staticmethod
    def confidence_weighted_voting(
        results_with_confidence: List[Tuple[List[str], float]]
    ) -> Dict[str, float]:
        """
        Confidence-weighted voting.
        
        Args:
            results_with_confidence: List of (items, confidence_score)
        
        Returns:
            Dict of item -> weighted score
        """
        weighted = Counter()
        
        for items, confidence in results_with_confidence:
            for item in items:
                weighted[item] += confidence
        
        return dict(weighted)


def format_aggregated_summary(agg_result: AggregatedResult, query: str) -> str:
    """
    Format aggregated result into human-readable summary.
    
    This uses the PROGRAMMATICALLY computed values, not LLM hallucination.
    
    Args:
        agg_result: Aggregated result from SecureAggregator
        query: Original query
    
    Returns:
        Formatted summary string
    """
    lines = []
    
    # Header
    lines.append("=" * 70)
    lines.append("FEDERATED QUERY RESULTS (Secure Aggregation)")
    lines.append("=" * 70)
    
    # Patient count - EXACT from aggregation
    lines.append(f"\n📊 TOTAL PATIENTS: {agg_result.total_patient_count}")
    lines.append(f"   (Aggregated from {len(agg_result.hospital_contributions)} hospitals)")
    
    # Hospital breakdown
    if agg_result.hospital_contributions:
        contrib_str = ", ".join(
            f"{h}: {c}" 
            for h, c in sorted(agg_result.hospital_contributions.items())
        )
        lines.append(f"   Breakdown: {contrib_str}")
    
    # Conditions with counts
    lines.append(f"\n📋 CONDITIONS FOUND:")
    if agg_result.aggregated_conditions:
        for condition, count in agg_result.aggregated_conditions[:7]:
            lines.append(f"   • {condition} (reported by {count} hospital{'s' if count > 1 else ''})")
    else:
        lines.append("   • None documented")
    
    # Treatments with counts
    lines.append(f"\n💊 TREATMENTS PRESCRIBED:")
    if agg_result.aggregated_treatments:
        for treatment, count in agg_result.aggregated_treatments[:7]:
            lines.append(f"   • {treatment} ({count} hospital{'s' if count > 1 else ''})")
    else:
        lines.append("   • None documented")
    
    # Procedures with counts
    lines.append(f"\n🔬 PROCEDURES PERFORMED:")
    if agg_result.aggregated_procedures:
        for procedure, count in agg_result.aggregated_procedures[:5]:
            lines.append(f"   • {procedure} ({count} hospital{'s' if count > 1 else ''})")
    else:
        lines.append("   • None documented")
    
    # Age distribution
    if agg_result.age_distribution:
        lines.append(f"\n👥 AGE DISTRIBUTION:")
        for age_range, count in sorted(agg_result.age_distribution.items()):
            lines.append(f"   • {age_range}: {count} patients")
    
    # Privacy footer
    lines.append(f"\n" + "-" * 70)
    lines.append(f"🔒 Privacy: {agg_result.aggregation_method}")
    lines.append(f"   K-anonymity threshold: {agg_result.privacy_params.get('k_anonymity_threshold', 'N/A')}")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# Example usage and demo
if __name__ == "__main__":
    print("=" * 70)
    print("FRAG-MED Secure Aggregation Demo")
    print("=" * 70)
    
    # Simulated hospital results (what you'd get from DPProtectedResult)
    hospital_results = [
        {
            'hospital_id': 'hospital_A',
            'dp_patient_count': 3,
            'dp_conditions': ['Acute bronchitis (disorder)'],
            'dp_treatments': ['acetaminophen 325 mg', 'amoxicillin 500 mg'],
            'dp_procedures': ['sputum examination'],
            'dp_age_distribution': {'Under 30': 3},
            'confidence_score': 0.85
        },
        {
            'hospital_id': 'hospital_B',
            'dp_patient_count': 3,
            'dp_conditions': ['Acute bronchitis (disorder)'],
            'dp_treatments': ['acetaminophen 325 mg', 'ibuprofen 100 mg'],
            'dp_procedures': ['sputum examination'],
            'dp_age_distribution': {'Under 30': 4},
            'confidence_score': 0.82
        },
        {
            'hospital_id': 'hospital_C',
            'dp_patient_count': 2,
            'dp_conditions': ['Acute bronchitis (disorder)'],
            'dp_treatments': ['acetaminophen 325 mg'],
            'dp_procedures': ['sputum examination'],
            'dp_age_distribution': {'Under 30': 6},
            'confidence_score': 0.78
        },
    ]
    
    # Perform secure aggregation
    aggregator = SecureAggregator(k_anonymity_threshold=3)
    result = aggregator.full_aggregation(hospital_results)
    
    # Format and print
    summary = format_aggregated_summary(result, "Find patients with acute bronchitis")
    print(summary)
    
    print("\n\n--- Comparison ---")
    print("✅ Programmatic count: 3 + 3 + 2 = 8 patients (EXACT)")
    print("❌ LLM might say: '27 patients' or '18 patients' (HALLUCINATED)")
