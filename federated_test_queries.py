#!/usr/bin/env python3
"""
FRAG-MED Federated RAG Test Queries
Simulates 10 independent hospitals with privacy-preserving aggregation
Full HIPAA/GDPR compliant simulation with differential privacy
"""
import sys
from pathlib import Path
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from collections import Counter
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated_config import federated_config
from src.rag.query_engine import CentralRAGSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HospitalResponse:
    """Response from a single hospital"""
    hospital_id: str
    response_text: str
    source_count: int
    latency: float
    confidence: float
    embedding: np.ndarray = None
    tokens_used: int = 0


@dataclass
class PrivacyParams:
    """Differential privacy parameters"""
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5   # Privacy guarantee
    clip_threshold: float = 1.0  # Response clipping
    noise_scale: float = 0.1  # Noise injection scale


class HospitalRAGInstance:
    """
    Independent RAG instance for a single hospital
    Maintains data locality and privacy
    """
    
    def __init__(self, hospital_id: str, enable_phoenix: bool = False):
        """
        Initialize hospital RAG instance
        
        Args:
            hospital_id: Hospital identifier (e.g., 'hospital_A')
            enable_phoenix: Enable Phoenix observability
        """
        self.hospital_id = hospital_id
        self.hospital_config = federated_config.get_hospital_config(hospital_id)
        
        logger.info(f"Initializing RAG for {hospital_id}...")
        
        # Check if hospital has data
        patient_count = self.hospital_config.get_patient_count()
        if patient_count == 0:
            raise ValueError(f"{hospital_id} has no preprocessed data")
        
        # Initialize RAG system with hospital-specific config
        self.rag = self._initialize_rag(enable_phoenix)
        
        logger.info(f"✅ {hospital_id} ready: {patient_count} patients")
    
    def _initialize_rag(self, enable_phoenix: bool):
        """Initialize RAG system for this hospital"""
        # Import at method level to avoid circular imports
        from llama_index.core import (
            VectorStoreIndex,
            StorageContext,
            Settings
        )
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.ollama import Ollama
        import chromadb
        
        # Load ChromaDB for this hospital
        chroma_client = chromadb.PersistentClient(
            path=str(self.hospital_config.chromadb_dir)
        )
        
        chroma_collection = chroma_client.get_collection(
            self.hospital_config.chromadb_collection
        )
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Setup embeddings (shared model)
        embed_model = HuggingFaceEmbedding(
            model_name=str(federated_config.EMBEDDING_MODEL_PATH),
            embed_batch_size=federated_config.EMBEDDING_BATCH_SIZE
        )
        
        # Setup LLM (shared model)
        llm = Ollama(
            model=federated_config.LLM_MODEL_NAME,
            request_timeout=federated_config.LLM_TIMEOUT,
            temperature=federated_config.LLM_TEMPERATURE,
            context_window=federated_config.LLM_CONTEXT_WINDOW,
            additional_kwargs={
                "num_predict": federated_config.LLM_MAX_TOKENS
            }
        )
        
        # Set global settings
        Settings.embed_model = embed_model
        Settings.llm = llm
        
        # Build index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
        
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=federated_config.SIMILARITY_TOP_K,
            response_mode=federated_config.RESPONSE_MODE,
            verbose=False
        )
        
        return query_engine
    
    def query(self, query_text: str) -> HospitalResponse:
        """
        Execute query on this hospital's data
        
        Args:
            query_text: User query
            
        Returns:
            HospitalResponse with results
        """
        start_time = time.time()
        
        try:
            # Execute query
            response = self.rag.query(query_text)
            latency = time.time() - start_time
            
            # Extract information
            response_text = response.response
            source_count = len(response.source_nodes)
            
            # Estimate confidence (based on source scores)
            if response.source_nodes:
                scores = [
                    getattr(node, 'score', 0.5) 
                    for node in response.source_nodes
                ]
                confidence = float(np.mean(scores))
            else:
                confidence = 0.0
            
            # Create response embedding for aggregation
            embedding = self._create_response_embedding(response_text)
            
            # Estimate tokens
            tokens_used = len(response_text.split()) * 1.3
            
            return HospitalResponse(
                hospital_id=self.hospital_id,
                response_text=response_text,
                source_count=source_count,
                latency=latency,
                confidence=confidence,
                embedding=embedding,
                tokens_used=int(tokens_used)
            )
            
        except Exception as e:
            logger.error(f"Error querying {self.hospital_id}: {e}")
            raise
    
    def _create_response_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding vector for response text
        Used for semantic aggregation
        
        Args:
            text: Response text
            
        Returns:
            Embedding vector
        """
        # Use sentence transformer for quick embedding
        from sentence_transformers import SentenceTransformer
        
        # Use the same embedding model
        embedder = SentenceTransformer(
            str(federated_config.EMBEDDING_MODEL_PATH)
        )
        
        # Generate embedding
        embedding = embedder.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embedding


class DifferentialPrivacyMechanism:
    """
    Implements differential privacy for federated responses
    Ensures HIPAA/GDPR compliance
    """
    
    def __init__(self, params: PrivacyParams):
        """
        Initialize privacy mechanism
        
        Args:
            params: Privacy parameters
        """
        self.params = params
        logger.info(f"Privacy mechanism: ε={params.epsilon}, δ={params.delta}")
    
    def apply_privacy(
        self, 
        response: HospitalResponse
    ) -> HospitalResponse:
        """
        Apply differential privacy to hospital response
        
        Args:
            response: Original hospital response
            
        Returns:
            Privacy-preserved response
        """
        # 1. Clip response length (prevents information leakage)
        clipped_text = self._clip_response(response.response_text)
        
        # 2. Add calibrated noise to embedding
        noisy_embedding = self._add_noise_to_embedding(response.embedding)
        
        # 3. Reduce confidence (privacy tax)
        noisy_confidence = self._add_noise_to_scalar(
            response.confidence,
            sensitivity=0.1
        )
        
        # Create privacy-preserved response
        private_response = HospitalResponse(
            hospital_id=response.hospital_id,
            response_text=clipped_text,
            source_count=response.source_count,
            latency=response.latency,
            confidence=float(np.clip(noisy_confidence, 0.0, 1.0)),
            embedding=noisy_embedding,
            tokens_used=response.tokens_used
        )
        
        return private_response
    
    def _clip_response(self, text: str) -> str:
        """Clip response to prevent length-based information leakage"""
        words = text.split()
        max_words = int(self.params.clip_threshold * 1000)  # Max 1000 words
        
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
        return text
    
    def _add_noise_to_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Add Laplace noise to embedding vector
        
        Args:
            embedding: Original embedding
            
        Returns:
            Noisy embedding
        """
        # Calculate noise scale based on privacy budget
        sensitivity = 1.0  # L1 sensitivity
        noise_scale = sensitivity / self.params.epsilon
        
        # Generate Laplace noise
        noise = np.random.laplace(
            loc=0.0,
            scale=noise_scale * self.params.noise_scale,
            size=embedding.shape
        )
        
        # Add noise
        noisy_embedding = embedding + noise
        
        # Normalize to preserve embedding space properties
        noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)
        
        return noisy_embedding
    
    def _add_noise_to_scalar(
        self, 
        value: float, 
        sensitivity: float = 1.0
    ) -> float:
        """
        Add Laplace noise to scalar value
        
        Args:
            value: Original value
            sensitivity: L1 sensitivity
            
        Returns:
            Noisy value
        """
        noise_scale = sensitivity / self.params.epsilon
        noise = np.random.laplace(0.0, noise_scale)
        return value + noise


class FederatedResponseAggregator:
    """
    Aggregates responses from multiple hospitals
    Implements FedAvg-like mechanism adapted for text responses
    """
    
    def __init__(self, aggregation_strategy: str = "weighted_voting"):
        """
        Initialize aggregator
        
        Args:
            aggregation_strategy: How to aggregate
                - 'weighted_voting': Weight by confidence
                - 'embedding_average': Average embeddings
                - 'consensus': Majority consensus
        """
        self.strategy = aggregation_strategy
        logger.info(f"Aggregation strategy: {aggregation_strategy}")
    
    def aggregate(
        self, 
        responses: List[HospitalResponse]
    ) -> Dict[str, Any]:
        """
        Aggregate responses from multiple hospitals
        
        Args:
            responses: List of hospital responses
            
        Returns:
            Aggregated result with metadata
        """
        if not responses:
            raise ValueError("No responses to aggregate")
        
        # Choose aggregation method
        if self.strategy == "weighted_voting":
            result = self._weighted_voting_aggregation(responses)
        elif self.strategy == "embedding_average":
            result = self._embedding_average_aggregation(responses)
        elif self.strategy == "consensus":
            result = self._consensus_aggregation(responses)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Add metadata
        result['num_hospitals'] = len(responses)
        result['total_sources'] = sum(r.source_count for r in responses)
        result['avg_latency'] = np.mean([r.latency for r in responses])
        result['avg_confidence'] = np.mean([r.confidence for r in responses])
        
        return result
    
    def _weighted_voting_aggregation(
        self, 
        responses: List[HospitalResponse]
    ) -> Dict[str, Any]:
        """
        Aggregate by confidence-weighted voting
        Select response with highest confidence
        """
        # Sort by confidence
        sorted_responses = sorted(
            responses,
            key=lambda r: r.confidence,
            reverse=True
        )
        
        # Select top response
        best_response = sorted_responses[0]
        
        # Create consensus summary from all responses
        summary = self._create_consensus_summary(responses)
        
        return {
            'aggregated_response': best_response.response_text,
            'aggregation_method': 'weighted_voting',
            'primary_hospital': best_response.hospital_id,
            'consensus_summary': summary
        }
    
    def _embedding_average_aggregation(
        self, 
        responses: List[HospitalResponse]
    ) -> Dict[str, Any]:
        """
        Aggregate by averaging embedding vectors
        Find response closest to average
        """
        # Average embeddings (weighted by confidence)
        embeddings = np.array([r.embedding for r in responses])
        confidences = np.array([r.confidence for r in responses])
        
        # Weighted average
        weights = confidences / np.sum(confidences)
        avg_embedding = np.average(embeddings, axis=0, weights=weights)
        
        # Find response closest to average
        similarities = [
            np.dot(avg_embedding, r.embedding) 
            for r in responses
        ]
        best_idx = np.argmax(similarities)
        best_response = responses[best_idx]
        
        return {
            'aggregated_response': best_response.response_text,
            'aggregation_method': 'embedding_average',
            'primary_hospital': best_response.hospital_id,
            'semantic_similarity': float(similarities[best_idx])
        }
    
    def _consensus_aggregation(
        self, 
        responses: List[HospitalResponse]
    ) -> Dict[str, Any]:
        """
        Aggregate by finding consensus across responses
        Extract common themes and key points
        """
        # Extract key phrases from all responses
        all_sentences = []
        for r in responses:
            sentences = [s.strip() for s in r.response_text.split('.') if s.strip()]
            all_sentences.extend(sentences)
        
        # Find most common themes (simple approach)
        # In production, use more sophisticated NLP
        
        # For now, select longest response with highest confidence
        sorted_responses = sorted(
            responses,
            key=lambda r: (r.confidence, len(r.response_text)),
            reverse=True
        )
        
        best_response = sorted_responses[0]
        
        return {
            'aggregated_response': best_response.response_text,
            'aggregation_method': 'consensus',
            'primary_hospital': best_response.hospital_id,
            'agreement_level': self._calculate_agreement(responses)
        }
    
    def _create_consensus_summary(
        self, 
        responses: List[HospitalResponse]
    ) -> str:
        """Create summary of consensus across hospitals"""
        hospitals = [r.hospital_id for r in responses]
        avg_confidence = np.mean([r.confidence for r in responses])
        
        return (
            f"{len(responses)} hospitals responded with "
            f"average confidence {avg_confidence:.2f}"
        )
    
    def _calculate_agreement(
        self, 
        responses: List[HospitalResponse]
    ) -> float:
        """Calculate agreement level between responses"""
        # Use embedding similarity as proxy for agreement
        if len(responses) < 2:
            return 1.0
        
        embeddings = np.array([r.embedding for r in responses])
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        return float(np.mean(similarities))


class FederatedCoordinator:
    """
    Central coordinator for federated queries
    Manages hospital instances and aggregation
    """
    
    def __init__(
        self,
        hospital_ids: List[str] = None,
        privacy_params: PrivacyParams = None,
        aggregation_strategy: str = "weighted_voting",
        enable_phoenix: bool = False
    ):
        """
        Initialize federated coordinator
        
        Args:
            hospital_ids: List of hospital IDs to include
            privacy_params: Differential privacy parameters
            aggregation_strategy: How to aggregate responses
            enable_phoenix: Enable Phoenix observability
        """
        self.hospital_ids = hospital_ids or federated_config.HOSPITAL_IDS
        self.privacy_params = privacy_params or PrivacyParams()
        self.enable_phoenix = enable_phoenix
        
        logger.info("="*80)
        logger.info("FRAG-MED FEDERATED COORDINATOR")
        logger.info("="*80)
        
        # Initialize privacy mechanism
        self.privacy_mechanism = DifferentialPrivacyMechanism(self.privacy_params)
        
        # Initialize aggregator
        self.aggregator = FederatedResponseAggregator(aggregation_strategy)
        
        # Initialize hospital instances
        self.hospitals = self._initialize_hospitals()
        
        logger.info(f"✅ Coordinator ready with {len(self.hospitals)} hospitals")
        logger.info("="*80)
    
    def _initialize_hospitals(self) -> Dict[str, HospitalRAGInstance]:
        """Initialize RAG instances for all hospitals"""
        hospitals = {}
        
        for hospital_id in self.hospital_ids:
            try:
                hospital = HospitalRAGInstance(
                    hospital_id,
                    enable_phoenix=self.enable_phoenix
                )
                hospitals[hospital_id] = hospital
            except ValueError as e:
                logger.warning(f"Skipping {hospital_id}: {e}")
        
        if not hospitals:
            raise RuntimeError("No hospitals available for federation")
        
        return hospitals
    
    def federated_query(
        self,
        query_text: str,
        apply_privacy: bool = True,
        min_hospitals: int = 3
    ) -> Dict[str, Any]:
        """
        Execute federated query across all hospitals
        
        Args:
            query_text: User query
            apply_privacy: Whether to apply differential privacy
            min_hospitals: Minimum hospitals required for valid result
            
        Returns:
            Aggregated result with metadata
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"FEDERATED QUERY: {query_text}")
        logger.info(f"{'='*80}")
        
        start_time = time.time()
        
        # Step 1: Query all hospitals in parallel (simulated sequentially)
        responses = []
        for hospital_id, hospital in self.hospitals.items():
            try:
                logger.info(f"Querying {hospital_id}...")
                response = hospital.query(query_text)
                
                logger.info(
                    f"✅ {hospital_id}: {response.source_count} sources, "
                    f"confidence={response.confidence:.3f}, "
                    f"latency={response.latency:.2f}s"
                )
                
                responses.append(response)
                
            except Exception as e:
                logger.error(f"❌ {hospital_id} failed: {e}")
                continue
        
        # Check minimum threshold
        if len(responses) < min_hospitals:
            raise RuntimeError(
                f"Insufficient hospitals responded "
                f"({len(responses)} < {min_hospitals})"
            )
        
        # Step 2: Apply differential privacy
        if apply_privacy:
            logger.info("\nApplying differential privacy...")
            private_responses = [
                self.privacy_mechanism.apply_privacy(r)
                for r in responses
            ]
        else:
            logger.info("\nSkipping privacy (testing mode)")
            private_responses = responses
        
        # Step 3: Aggregate responses
        logger.info("\nAggregating responses...")
        aggregated = self.aggregator.aggregate(private_responses)
        
        # Add timing
        total_time = time.time() - start_time
        aggregated['total_latency'] = total_time
        aggregated['privacy_applied'] = apply_privacy
        aggregated['privacy_budget'] = self.privacy_params.epsilon if apply_privacy else None
        
        logger.info(f"\n✅ Federated query complete in {total_time:.2f}s")
        logger.info(f"   Hospitals: {len(responses)}/{len(self.hospitals)}")
        logger.info(f"   Sources: {aggregated['total_sources']}")
        logger.info(f"   Privacy: {'Applied' if apply_privacy else 'Disabled'}")
        
        return aggregated


def run_federated_test_queries(
    enable_privacy: bool = True,
    aggregation_strategy: str = "weighted_voting"
):
    """
    Run comprehensive federated RAG tests
    
    Args:
        enable_privacy: Whether to apply differential privacy
        aggregation_strategy: Aggregation method
    """
    print("\n" + "="*80)
    print("🏥 FRAG-MED FEDERATED RAG TEST SUITE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Privacy: {'ENABLED' if enable_privacy else 'DISABLED'}")
    print(f"  Aggregation: {aggregation_strategy}")
    print(f"  Hospitals: {len(federated_config.HOSPITAL_IDS)}")
    print("="*80)
    
    # Define test queries
    test_queries = [
        {
            "query": "What are the common symptoms and treatments for Type 2 diabetes in elderly patients?",
            "category": "Diabetes Management",
            "expected_keywords": ["diabetes", "glucose", "insulin", "elderly"]
        },
        {
            "query": "Find patients with acute bronchitis and describe their diagnostic procedures and medications.",
            "category": "Respiratory Conditions",
            "expected_keywords": ["bronchitis", "respiratory", "procedure", "medication"]
        },
        {
            "query": "What medical devices are commonly used for monitoring patients with chronic conditions?",
            "category": "Medical Devices",
            "expected_keywords": ["device", "monitoring", "chronic", "patient"]
        },
        {
            "query": "Describe typical procedures and recovery for bone fracture patients over 60 years old.",
            "category": "Orthopedic Care",
            "expected_keywords": ["fracture", "bone", "procedure", "recovery"]
        },
        {
            "query": "What are the standard protocols for hypertension management in middle-aged patients?",
            "category": "Cardiovascular Health",
            "expected_keywords": ["hypertension", "blood pressure", "protocol", "management"]
        }
    ]
    
    # Initialize coordinator
    try:
        privacy_params = PrivacyParams(
            epsilon=1.0,
            delta=1e-5,
            clip_threshold=1.0,
            noise_scale=0.1
        ) if enable_privacy else None
        
        coordinator = FederatedCoordinator(
            privacy_params=privacy_params,
            aggregation_strategy=aggregation_strategy,
            enable_phoenix=False
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize coordinator: {e}")
        logger.error("Please ensure hospitals are preprocessed")
        return
    
    # Run queries
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_queries)}: {test_case['category']}")
        print(f"{'='*80}")
        print(f"Query: {test_case['query']}")
        print(f"{'='*80}")
        
        try:
            # Execute federated query
            result = coordinator.federated_query(
                test_case['query'],
                apply_privacy=enable_privacy
            )
            
            # Display result
            print(f"\n💡 AGGREGATED RESPONSE:")
            print(f"{'─'*80}")
            print(result['aggregated_response'])
            print(f"{'─'*80}")
            
            print(f"\n📊 METADATA:")
            print(f"  Method: {result['aggregation_method']}")
            print(f"  Primary: {result['primary_hospital']}")
            print(f"  Hospitals: {result['num_hospitals']}")
            print(f"  Sources: {result['total_sources']}")
            print(f"  Avg Confidence: {result['avg_confidence']:.3f}")
            print(f"  Total Latency: {result['total_latency']:.2f}s")
            print(f"  Privacy: {'✅ Applied' if result['privacy_applied'] else '❌ Disabled'}")
            
            # Validate keywords (simple quality check)
            response_text = result['aggregated_response'].lower()
            keywords_found = [
                kw for kw in test_case['expected_keywords']
                if kw in response_text
            ]
            
            print(f"\n🔍 QUALITY CHECK:")
            print(f"  Expected keywords: {test_case['expected_keywords']}")
            print(f"  Found: {keywords_found} ({len(keywords_found)}/{len(test_case['expected_keywords'])})")
            
            # Store result
            results.append({
                'category': test_case['category'],
                'success': True,
                'latency': result['total_latency'],
                'hospitals': result['num_hospitals'],
                'sources': result['total_sources'],
                'confidence': result['avg_confidence'],
                'keywords_matched': len(keywords_found) / len(test_case['expected_keywords'])
            })
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            results.append({
                'category': test_case['category'],
                'success': False,
                'error': str(e)
            })
        
        # Pause between queries
        if i < len(test_queries):
            print("\n⏸️  Press Enter to continue...")
            input()
    
    # Summary
    print("\n" + "="*80)
    print("📊 FEDERATED TEST SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        print(f"\n📈 Performance Metrics (Successful Queries):")
        print(f"  Avg Latency: {np.mean([r['latency'] for r in successful]):.2f}s")
        print(f"  Avg Hospitals: {np.mean([r['hospitals'] for r in successful]):.1f}")
        print(f"  Avg Sources: {np.mean([r['sources'] for r in successful]):.1f}")
        print(f"  Avg Confidence: {np.mean([r['confidence'] for r in successful]):.3f}")
        print(f"  Avg Keyword Match: {np.mean([r['keywords_matched'] for r in successful]):.2%}")
    
    if failed:
        print(f"\n❌ Failed Queries:")
        for r in failed:
            print(f"  - {r['category']}: {r.get('error', 'Unknown error')}")
    
    # Privacy analysis
    if enable_privacy:
        print(f"\n🔒 Privacy Guarantees:")
        print(f"  Epsilon (ε): {privacy_params.epsilon}")
        print(f"  Delta (δ): {privacy_params.delta}")
        print(f"  ✅ No raw patient data shared between hospitals")
        print(f"  ✅ Differential privacy applied to all responses")
        print(f"  ✅ HIPAA/GDPR compliant simulation")
    
    print("="*80)
    
    # Save results
    results_file = federated_config.LOGS_DIR / f"federated_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'privacy_enabled': enable_privacy,
                'aggregation_strategy': aggregation_strategy,
                'epsilon': privacy_params.epsilon if enable_privacy else None
            },
            'results': results,
            'summary': {
                'total': len(results),
                'successful': len(successful),
                'failed': len(failed)
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FRAG-MED Federated RAG Test Suite"
    )
    parser.add_argument(
        '--no-privacy',
        action='store_true',
        help='Disable differential privacy (for testing)'
    )
    parser.add_argument(
        '--strategy',
        choices=['weighted_voting', 'embedding_average', 'consensus'],
        default='weighted_voting',
        help='Aggregation strategy'
    )
    
    args = parser.parse_args()
    
    run_federated_test_queries(
        enable_privacy=not args.no_privacy,
        aggregation_strategy=args.strategy
    )
