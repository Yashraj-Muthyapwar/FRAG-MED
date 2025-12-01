#!/usr/bin/env python3
"""
FRAG-MED Federated Orchestrator with DP + Phoenix Observability
Includes comprehensive tracing for debugging.
"""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate

from hospital_rag_dp import HospitalRAGWithDP, DPProtectedResult

logger = logging.getLogger(__name__)


@dataclass
class FederatedDPResult:
    """Final aggregated result with DP guarantees."""
    query: str
    aggregated_response: str
    total_dp_patient_count: int
    hospitals_queried: List[str]
    hospitals_with_results: List[str]
    aggregated_conditions: List[str]
    aggregated_treatments: List[str]
    aggregated_procedures: List[str]
    combined_age_distribution: Dict[str, int]
    hospital_contributions: List[Dict]
    total_latency_seconds: float
    privacy_summary: Dict
    debug_info: Dict  # For Phoenix tracing


# Strict aggregation prompt - forces LLM to use only provided data
AGGREGATION_PROMPT = PromptTemplate(
    """You are a FEDERATED MEDICAL DATA AGGREGATOR. Your ONLY job is to combine the hospital reports below into a unified summary.

CRITICAL RULES:
1. ONLY use information explicitly stated in the HOSPITAL REPORTS below
2. NEVER add general medical knowledge or textbook definitions
3. NEVER explain what a disease is or how it's typically treated
4. If a hospital found 0 patients, exclude it from your summary
5. Report exact numbers and medication names as provided

═══════════════════════════════════════════════════════════════════════════════
HOSPITAL REPORTS (This is your ONLY source of information):
═══════════════════════════════════════════════════════════════════════════════
{hospital_data}
═══════════════════════════════════════════════════════════════════════════════

QUERY: {query}


YOUR TASK:
Write a 3-5 sentence summary that ONLY reports:
1. The total patient count per hospital
2. Which conditions were found 
3. Which treatments were prescribed 
4. Any procedures performed
5. Any patient devices used based on the query

DO NOT:
- Define what any disease is
- Explain how treatments work
- Add symptoms not mentioned in the reports
- Provide medical advice

FEDERATED SUMMARY:"""
)


class FederatedOrchestratorDP:
    """
    Federated orchestrator with DP + Phoenix observability.
    """
    
    def __init__(
        self,
        federated_config,
        dp_epsilon: float = 1.0,
        dp_k_threshold: int = 3,
        enable_phoenix: bool = True,
        verbose: bool = True
    ):
        self.config = federated_config
        self.dp_epsilon = dp_epsilon
        self.dp_k_threshold = dp_k_threshold
        self.verbose = verbose
        self.enable_phoenix = enable_phoenix
        
        self.hospitals: Dict[str, HospitalRAGWithDP] = {}
        
        # Phoenix for orchestrator level
        self.phoenix = None
        if enable_phoenix:
            self._init_phoenix()
        
        # Aggregation LLM
        self.aggregation_llm = Ollama(
            model=federated_config.LLM_MODEL_NAME,
            temperature=0.0,
            request_timeout=300,
        )
        
        self.stats = {
            'total_queries': 0,
            'total_hospitals_queried': 0,
            'total_dp_patients_found': 0,
            'total_latency': 0.0
        }
        
        logger.info(f"🌐 Federated Orchestrator with DP + Phoenix initialized")
    
    def _init_phoenix(self):
        """Initialize Phoenix for orchestrator"""
        try:
            from src.observability import PhoenixObservability
            self.phoenix = PhoenixObservability(
                project_name="frag-med-orchestrator",
                launch_server=False,
                enable_tracing=True
            )
            logger.info("📊 Phoenix connected for orchestrator")
        except Exception as e:
            logger.warning(f"Phoenix init failed: {e}")
            self.phoenix = None
    
    def register_hospital(self, hospital_id: str) -> bool:
        """Register a hospital with DP-enabled RAG"""
        if hospital_id in self.hospitals:
            return True
        
        try:
            hospital_config = self.config.get_hospital_config(hospital_id)
            
            hospital_rag = HospitalRAGWithDP(
                hospital_config=hospital_config,
                embedding_model_path=self.config.EMBEDDING_MODEL_PATH,
                llm_model_name=self.config.LLM_MODEL_NAME,
                similarity_top_k=getattr(self.config, 'SIMILARITY_TOP_K', 5),
                dp_epsilon=self.dp_epsilon,
                dp_k_threshold=self.dp_k_threshold,
                enable_phoenix=self.enable_phoenix,
                verbose=self.verbose
            )
            
            self.hospitals[hospital_id] = hospital_rag
            
            if self.verbose:
                stats = hospital_rag.get_stats()
                print(f"✅ Registered: {hospital_id} ({stats['vector_count']:,} vectors)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register {hospital_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def register_all_hospitals(self) -> int:
        """Register all hospitals"""
        print("\n" + "="*70)
        print("🏥 REGISTERING FEDERATED NETWORK WITH DP + PHOENIX")
        print(f"   ε={self.dp_epsilon}/hospital, k={self.dp_k_threshold}")
        print(f"   Phoenix: {'Enabled' if self.phoenix else 'Disabled'}")
        print("="*70)
        
        success = 0
        for hospital_id in self.config.HOSPITAL_IDS:
            if self.register_hospital(hospital_id):
                success += 1
        
        print(f"\n✅ Registered {success}/{len(self.config.HOSPITAL_IDS)} hospitals")
        print("="*70)
        
        return success
    
    def _query_hospital(self, hospital_id: str, query: str) -> Optional[DPProtectedResult]:
        """Query single hospital"""
        try:
            hospital = self.hospitals[hospital_id]
            return hospital.query_with_dp(query)
        except Exception as e:
            logger.error(f"Query failed at {hospital_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_structured_summary(self, results: List[DPProtectedResult], query: str) -> str:
        """
        Build summary using LLM aggregation of hospital responses.
        The LLM combines the DP-protected responses from each hospital.
        """
        if not results:
            return "No results from any hospital."
        
        # Filter to hospitals with results
        valid_results = [r for r in results if r and r.raw_patient_count > 0]
        
        if not valid_results:
            return "No matching patient records found across the federated hospital network."
        
        # Collect statistics for the prompt
        total_patients = sum(r.dp_patient_count for r in valid_results)
        hospitals_with_results = [r.hospital_id for r in valid_results]
        
        all_conditions = []
        all_treatments = []
        all_procedures = []
        
        # Build hospital data section from LLM responses
        hospital_data_parts = []
        
        for r in valid_results:
            all_conditions.extend(r.dp_conditions)
            all_treatments.extend(r.dp_treatments)
            all_procedures.extend(r.dp_procedures)
            
            # Include each hospital's LLM-generated response
            age_info = ', '.join(f'{k}: {v}' for k, v in r.dp_age_distribution.items()) if r.dp_age_distribution else 'Not available'
            
            hospital_report = f"""
### {r.hospital_id} (DP Patient Count: {r.dp_patient_count})
**Conditions:** {', '.join(r.dp_conditions[:5]) if r.dp_conditions else 'None documented'}
**Treatments:** {', '.join(r.dp_treatments[:5]) if r.dp_treatments else 'None documented'}  
**Procedures:** {', '.join(r.dp_procedures[:3]) if r.dp_procedures else 'None documented'}
**Age Distribution:** {age_info}

**Hospital's Medical Findings:**
{r.dp_response}
"""
            hospital_data_parts.append(hospital_report)
        
        hospital_data = "\n---\n".join(hospital_data_parts)
        
        # Deduplicate lists for the constraint section
        unique_conditions = list(set(all_conditions))[:10]
        unique_treatments = list(set(all_treatments))[:10]
        unique_procedures = list(set(all_procedures))[:5]
        
        # Format the aggregation prompt
        prompt = AGGREGATION_PROMPT.format(
            hospital_data=hospital_data,
            query=query,
            total_patients=total_patients,
            hospitals_with_results=', '.join(hospitals_with_results),
            all_conditions=', '.join(unique_conditions) if unique_conditions else 'None documented',
            all_treatments=', '.join(unique_treatments) if unique_treatments else 'None documented',
            all_procedures=', '.join(unique_procedures) if unique_procedures else 'None documented'
        )
        
        if self.verbose:
            print(f"\n   📝 LLM AGGREGATION DEBUG:")
            print(f"      Hospitals with data: {len(valid_results)}")
            print(f"      Total DP patients: {total_patients}")
            print(f"      Conditions collected: {unique_conditions[:5]}")
            print(f"      Treatments collected: {unique_treatments[:5]}")
            print(f"      Prompt length: {len(prompt)} chars")
        
        try:
            if self.verbose:
                print(f"      ⏳ Calling aggregation LLM...")
            
            start = time.time()
            response = self.aggregation_llm.complete(prompt)
            gen_time = time.time() - start
            
            if self.verbose:
                print(f"      ✅ Aggregation complete in {gen_time:.2f}s")
                print(f"      Response length: {len(response.text)} chars")
            
            # Add privacy footer
            result_text = response.text.strip()
            result_text += f"\n\n[Privacy Protected: ε={self.dp_epsilon}/hospital, k≥{self.dp_k_threshold} | {len(valid_results)} hospitals contributed data]"
            
            return result_text
            
        except Exception as e:
            logger.error(f"LLM aggregation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to structured format
            return self._fallback_summary(valid_results, total_patients, unique_conditions, unique_treatments)
    
    def _fallback_summary(self, results: List[DPProtectedResult], total_patients: int, 
                          conditions: List[str], treatments: List[str]) -> str:
        """Fallback if LLM aggregation fails - returns structured data"""
        parts = [
            f"FEDERATED RESULTS (LLM aggregation failed - showing raw data)",
            f"",
            f"Total Patients: {total_patients} across {len(results)} hospitals",
            f"",
            f"Hospitals: {', '.join(r.hospital_id for r in results)}",
            f"",
            f"Conditions Found: {', '.join(conditions[:7]) if conditions else 'None'}",
            f"",
            f"Treatments Found: {', '.join(treatments[:7]) if treatments else 'None'}",
            f"",
            f"[Privacy: ε={self.dp_epsilon}, Fallback mode]"
        ]
        return "\n".join(parts)
    
    def federated_query_with_dp(
        self,
        query: str,
        hospital_ids: List[str] = None,
        parallel: bool = True,
        max_workers: int = 3
    ) -> FederatedDPResult:
        """
        Execute federated query with DP + comprehensive tracing.
        """
        start_time = time.time()
        
        print("\n" + "="*70)
        print("🔒 FEDERATED QUERY WITH DP + PHOENIX TRACING")
        print("="*70)
        print(f"Query: {query}")
        print(f"Privacy: ε={self.dp_epsilon}/hospital, k={self.dp_k_threshold}")
        
        # Collect debug info
        debug_info = {
            'query': query,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hospital_traces': {}
        }
        
        # Target hospitals
        target_hospitals = hospital_ids or list(self.hospitals.keys())
        target_hospitals = [h for h in target_hospitals if h in self.hospitals]
        
        if not target_hospitals:
            return self._empty_result(query, start_time, debug_info)
        
        print(f"\n📡 Querying {len(target_hospitals)} hospitals...")
        
        # Execute queries (sequential for better debugging)
        results: List[DPProtectedResult] = []
        
        if parallel and len(target_hospitals) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(target_hospitals))) as executor:
                futures = {
                    executor.submit(self._query_hospital, h_id, query): h_id
                    for h_id in target_hospitals
                }
                for future in as_completed(futures):
                    h_id = futures[future]
                    result = future.result()
                    if result:
                        results.append(result)
                        debug_info['hospital_traces'][h_id] = result.retrieval_debug
        else:
            for h_id in target_hospitals:
                result = self._query_hospital(h_id, query)
                if result:
                    results.append(result)
                    debug_info['hospital_traces'][h_id] = result.retrieval_debug
        
        # Aggregate hospital responses using LLM
        print("\n🔄 Aggregating hospital responses with LLM...")
        aggregated = self._build_structured_summary(results, query)
        
        # Collect stats
        total_dp_count = sum(r.dp_patient_count for r in results if r)
        hospitals_with_results = [r.hospital_id for r in results if r and r.dp_patient_count > 0]
        
        all_conditions = set()
        all_treatments = set()
        all_procedures = set()
        combined_age_dist = {}
        
        for result in results:
            if result:
                all_conditions.update(result.dp_conditions)
                all_treatments.update(result.dp_treatments)
                all_procedures.update(result.dp_procedures)
                for age, count in result.dp_age_distribution.items():
                    combined_age_dist[age] = combined_age_dist.get(age, 0) + count
        
        contributions = [
            {
                'hospital_id': r.hospital_id,
                'raw_patient_count': r.raw_patient_count,
                'dp_patient_count': r.dp_patient_count,
                'dp_conditions': r.dp_conditions[:5],
                'dp_treatments': r.dp_treatments[:5],
                'confidence': r.confidence_score,
                'latency': r.latency_seconds
            }
            for r in results if r
        ]
        
        total_latency = time.time() - start_time
        
        # Update stats
        self.stats['total_queries'] += 1
        self.stats['total_hospitals_queried'] += len(target_hospitals)
        self.stats['total_dp_patients_found'] += total_dp_count
        self.stats['total_latency'] += total_latency
        
        privacy_summary = {
            'mechanism': 'Response-Level Differential Privacy',
            'epsilon_per_hospital': self.dp_epsilon,
            'total_epsilon': self.dp_epsilon * len(target_hospitals),
            'k_anonymity_threshold': self.dp_k_threshold,
            'hospitals_queried': len(target_hospitals)
        }
        
        # Print results
        print("\n" + "="*70)
        print("📊 FEDERATED DP QUERY RESULTS")
        print("="*70)
        print(f"Total DP patient count: {total_dp_count}")
        print(f"Hospitals queried: {len(target_hospitals)}")
        print(f"Hospitals with results: {len(hospitals_with_results)}")
        print(f"Total latency: {total_latency:.2f}s")
        
        print(f"\n🔒 Privacy Guarantee:")
        print(f"   Total ε = {privacy_summary['total_epsilon']:.2f}")
        
        print("\n📝 Aggregated Response:")
        print("-"*70)
        print(aggregated)
        print("-"*70)
        
        # Print debug summary
        print("\n📊 DEBUG SUMMARY:")
        for h_id, trace in debug_info['hospital_traces'].items():
            print(f"   {h_id}:")
            print(f"      Retrieved: {trace.get('child_nodes_retrieved', 0)} child nodes")
            print(f"      Loaded: {trace.get('parent_docs_loaded', 0)} parent docs")
            print(f"      Patients: {trace.get('patient_pseudonyms', [])}")
        
        return FederatedDPResult(
            query=query,
            aggregated_response=aggregated,
            total_dp_patient_count=total_dp_count,
            hospitals_queried=target_hospitals,
            hospitals_with_results=hospitals_with_results,
            aggregated_conditions=list(all_conditions)[:15],
            aggregated_treatments=list(all_treatments)[:15],
            aggregated_procedures=list(all_procedures)[:10],
            combined_age_distribution=combined_age_dist,
            hospital_contributions=contributions,
            total_latency_seconds=total_latency,
            privacy_summary=privacy_summary,
            debug_info=debug_info
        )
    
    def _empty_result(self, query: str, start_time: float, debug_info: Dict) -> FederatedDPResult:
        """Return empty result"""
        return FederatedDPResult(
            query=query,
            aggregated_response="No hospitals available.",
            total_dp_patient_count=0,
            hospitals_queried=[],
            hospitals_with_results=[],
            aggregated_conditions=[],
            aggregated_treatments=[],
            aggregated_procedures=[],
            combined_age_distribution={},
            hospital_contributions=[],
            total_latency_seconds=time.time() - start_time,
            privacy_summary={'mechanism': 'N/A'},
            debug_info=debug_info
        )
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            **self.stats,
            'registered_hospitals': len(self.hospitals),
            'dp_epsilon_per_hospital': self.dp_epsilon,
            'dp_k_threshold': self.dp_k_threshold
        }
    
    def get_privacy_report(self) -> str:
        """Generate privacy report"""
        return f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    FRAG-MED PRIVACY COMPLIANCE REPORT                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Privacy Budget (ε) per hospital: {self.dp_epsilon:<10}                          ║
║  K-anonymity threshold: {self.dp_k_threshold:<10}                                     ║
║  Queries processed: {self.stats['total_queries']:<10}                                ║
║  Hospitals in network: {len(self.hospitals):<10}                                ║
║                                                                          ║
║  Privacy Mechanisms:                                                     ║
║  ✓ Laplace noise on patient counts                                       ║
║  ✓ K-anonymity suppression (k≥{self.dp_k_threshold})                                   ║
║  ✓ Age range generalization                                              ║
║  ✓ Patient ID sanitization                                               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    
    def shutdown(self):
        """Shutdown"""
        print("\n🔌 Shutting down...")
        for hospital in self.hospitals.values():
            hospital.shutdown()
        if self.phoenix:
            try:
                self.phoenix.shutdown()
            except:
                pass
        print("✅ Shutdown complete")
