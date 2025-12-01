#!/usr/bin/env python3
"""
FRAG-MED Hospital RAG with Differential Privacy + Phoenix Observability
FIXED VERSION: Corrected context formatting and token limits

Key fixes:
1. Fixed context_str formatting bug (line 337 original)
2. Added num_predict (output tokens) to Ollama config
3. Increased context window to 8192 (BioMistral supports this)
4. Reduced records per query to prevent overflow
5. Better truncation handling
"""

import logging
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from llama_index.core import VectorStoreIndex, Settings as LlamaSettings, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from dp_mechanisms import ResponseLevelDP, create_dp_protector

logger = logging.getLogger(__name__)


@dataclass
class DPProtectedResult:
    """DP-protected result from a hospital query."""
    hospital_id: str
    dp_patient_count: int
    dp_response: str
    dp_conditions: List[str]
    dp_treatments: List[str]
    dp_procedures: List[str]
    dp_age_distribution: Dict[str, int]
    confidence_score: float
    latency_seconds: float
    privacy_params: Dict
    # Debug info
    raw_patient_count: int  # Before DP
    retrieval_debug: Dict   # For tracing


# =============================================================================
# IMPROVED PROMPT - More concise, forces structured output
# =============================================================================

STRICT_EXTRACTION_PROMPT = PromptTemplate(
    """You are MedExtract-Pro. Extract data ONLY from the patient records below. Never add external knowledge.

PATIENT RECORDS:
{context_str}

QUERY: {query_str}

Extract and format EXACTLY as shown (use "None documented" if not found):

PATIENT COUNT: [number of patients above]
CONDITIONS FOUND:
- [condition 1]
- [condition 2]
TREATMENTS FOUND:
- [medication 1]
- [medication 2]
PROCEDURES FOUND:
- [procedure 1]
DEVICES USED:
-[Device 1]
-[Device 2]
OBSERVATIONS FOUND:
-[Observation 1]
-[Observation 2]

EXTRACTED DATA:"""
)

# Fallback prompt if no records found
NO_RECORDS_RESPONSE = "No matching patient records were found at {hospital_id} for this query."


class HospitalRAGWithDP:
    """
    Hospital RAG with DP protection and Phoenix observability.
    FIXED VERSION with proper context handling.
    """
    
    def __init__(
        self,
        hospital_config,
        embedding_model_path: Path,
        llm_model_name: str = "jsk/bio-mistral",
        similarity_top_k: int = 5,
        dp_epsilon: float = 1.0,
        dp_k_threshold: int = 3,
        enable_phoenix: bool = True,
        verbose: bool = True
    ):
        self.hospital_id = hospital_config.hospital_id
        self.chromadb_dir = hospital_config.chromadb_dir
        self.parent_docs_dir = hospital_config.parent_docs_dir
        self.collection_name = hospital_config.chromadb_collection
        self.embedding_model_path = Path(embedding_model_path)
        self.similarity_top_k = similarity_top_k
        self.verbose = verbose
        self.enable_phoenix = enable_phoenix
        
        # DP setup
        self.dp_protector = create_dp_protector(epsilon=dp_epsilon, k_threshold=dp_k_threshold)
        self.dp_epsilon = dp_epsilon
        self.dp_k_threshold = dp_k_threshold
        
        # Phoenix observability
        self.phoenix = None
        if enable_phoenix:
            self._init_phoenix()
        
        # Initialize components
        self._init_vector_store()
        self._init_models(llm_model_name)
        
        if verbose:
            logger.info(f"✅ HospitalRAGWithDP initialized: {self.hospital_id}")
            logger.info(f"   DP: ε={dp_epsilon}, k={dp_k_threshold}")
            logger.info(f"   Phoenix: {'Enabled' if self.phoenix else 'Disabled'}")
    
    def _init_phoenix(self):
        """Initialize Phoenix observability"""
        try:
            from src.observability import PhoenixObservability
            self.phoenix = PhoenixObservability(
                project_name=f"frag-med-{self.hospital_id}",
                launch_server=False,
                enable_tracing=True
            )
            logger.info(f"📊 Phoenix connected for {self.hospital_id}")
        except Exception as e:
            logger.warning(f"Phoenix init failed: {e}")
            self.phoenix = None
    
    def _init_vector_store(self):
        """Initialize ChromaDB"""
        if not self.chromadb_dir.exists():
            raise FileNotFoundError(f"ChromaDB not found: {self.chromadb_dir}")
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chromadb_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.chroma_collection = self.chroma_client.get_collection(self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        if self.verbose:
            count = self.chroma_collection.count()
            logger.info(f"📊 {self.hospital_id}: {count:,} vectors")
    
    def _init_models(self, llm_model_name: str):
        """Initialize models with FIXED settings"""
        self.embed_model = HuggingFaceEmbedding(
            model_name=str(self.embedding_model_path)
        )
        
        # FIXED: Added num_predict for output tokens, increased context_window
        self.llm = Ollama(
            model=llm_model_name,
            temperature=0.0,
            request_timeout=300,
            context_window=8192,  # FIXED: Increased from 4096
            additional_kwargs={
                "num_predict": 1024,  # FIXED: Ensure enough output tokens
                "num_ctx": 8192,      # FIXED: Match context window
            }
        )
        
        LlamaSettings.embed_model = self.embed_model
        LlamaSettings.llm = self.llm
        
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
    
    def _load_parent_document(self, parent_doc_id: str) -> Optional[str]:
        """Load parent document from disk"""
        for batch_dir in sorted(self.parent_docs_dir.glob("batch_*")):
            file_path = batch_dir / f"{parent_doc_id}.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('content', '')
        return None
    
    def _retrieve_and_trace(self, query: str) -> tuple:
        """Retrieve documents with detailed tracing."""
        debug_info = {
            'query': query,
            'hospital_id': self.hospital_id,
            'similarity_top_k': self.similarity_top_k,
            'child_nodes_retrieved': 0,
            'parent_docs_loaded': 0,
            'retrieval_scores': [],
            'patient_pseudonyms': [],
            'sample_content_preview': None
        }
        
        # Step 1: Retrieve from vector index
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
        )
        
        query_bundle = QueryBundle(query_str=query)
        child_nodes = retriever.retrieve(query_bundle)
        
        debug_info['child_nodes_retrieved'] = len(child_nodes)
        
        if self.verbose:
            print(f"\n   🔍 RETRIEVAL DEBUG [{self.hospital_id}]:")
            print(f"      Query: '{query[:50]}...'")
            print(f"      Child nodes retrieved: {len(child_nodes)}")
        
        # Step 2: Process child nodes
        records = []
        seen_parents = set()
        
        for i, child in enumerate(child_nodes):
            score = child.score
            parent_id = child.node.metadata.get("parent_doc_id")
            patient = child.node.metadata.get("patient_pseudonym", "Unknown")
            
            debug_info['retrieval_scores'].append(score)
            
            if self.verbose:
                print(f"      [{i+1}] Score: {score:.4f} | Parent: {parent_id} | Patient: {patient}")
            
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                
                parent_content = self._load_parent_document(parent_id)
                if parent_content:
                    records.append({
                        'parent_id': parent_id,
                        'content': parent_content,
                        'metadata': child.node.metadata,
                        'score': score
                    })
                    debug_info['patient_pseudonyms'].append(patient)
                    debug_info['parent_docs_loaded'] += 1
                    
                    if debug_info['sample_content_preview'] is None:
                        debug_info['sample_content_preview'] = parent_content[:500]
        
        if self.verbose:
            print(f"      Parent docs loaded: {len(records)}")
            if records:
                print(f"      Patients: {debug_info['patient_pseudonyms']}")
        
        return records, debug_info
    
    def _extract_entities_from_content(self, records: List[Dict]) -> Dict:
        """Extract structured entities from record content"""
        conditions = []
        treatments = []
        procedures = []
        age_dist = {}
        
        for record in records:
            content = record.get('content', '')
            metadata = record.get('metadata', {})
            
            # Age distribution
            age_range = metadata.get('age_range', 'Unknown')
            age_dist[age_range] = age_dist.get(age_range, 0) + 1
            
            # Extract conditions
            if '=== CONDITIONS DOCUMENTED ===' in content:
                section = content.split('=== CONDITIONS DOCUMENTED ===')[1].split('===')[0]
                for line in section.strip().split('\n'):
                    if line.strip().startswith('- '):
                        cond = line.strip()[2:].strip()
                        if cond and cond != 'None':
                            conditions.append(cond)
            
            # Extract medications
            if '=== MEDICATIONS ===' in content:
                section = content.split('=== MEDICATIONS ===')[1].split('===')[0]
                for line in section.strip().split('\n'):
                    if line.strip().startswith('- '):
                        med = line.strip()[2:].strip()
                        if med and med != 'None':
                            treatments.append(med)
            
            # Extract procedures
            if '=== PROCEDURES PERFORMED ===' in content:
                section = content.split('=== PROCEDURES PERFORMED ===')[1].split('===')[0]
                for line in section.strip().split('\n'):
                    if line.strip().startswith('- '):
                        proc = line.strip()[2:].strip()
                        if proc and proc != 'None':
                            procedures.append(proc)
        
        return {
            'conditions': list(set(conditions)),
            'treatments': list(set(treatments)),
            'procedures': list(set(procedures)),
            'age_distribution': age_dist
        }
    
    def _generate_with_trace(self, query: str, records: List[Dict], debug_info: Dict) -> str:
        """Generate response with detailed tracing - FIXED VERSION"""
        
        if not records:
            response = NO_RECORDS_RESPONSE.format(hospital_id=self.hospital_id)
            debug_info['generation_status'] = 'no_records'
            return response
        
        # FIXED: Build context with proper separators
        # Limit to 3 records to prevent context overflow
        max_records = min(3, len(records))
        context_parts = []
        
        for i, record in enumerate(records[:max_records], 1):
            # FIXED: Smarter truncation - keep important sections
            content = self._truncate_record_smartly(record['content'], max_chars=2000)
            context_parts.append(f"[Record {i} - Patient {record['metadata'].get('patient_pseudonym', 'Unknown')}]\n{content}")
        
        # FIXED: Proper separator formatting
        separator = "\n\n" + "=" * 50 + "\n\n"
        context_str = separator.join(context_parts)
        
        debug_info['context_length'] = len(context_str)
        debug_info['num_records_in_context'] = len(context_parts)
        
        if self.verbose:
            print(f"\n   📝 GENERATION DEBUG [{self.hospital_id}]:")
            print(f"      Records in context: {len(context_parts)}")
            print(f"      Context length: {len(context_str)} chars (~{len(context_str)//4} tokens)")
            print(f"      Context preview (first 300 chars):")
            print(f"      {context_str[:300]}...")
        
        # Format prompt
        prompt_text = STRICT_EXTRACTION_PROMPT.format(
            context_str=context_str,
            query_str=query
        )
        
        debug_info['prompt_length'] = len(prompt_text)
        estimated_tokens = len(prompt_text) // 4
        debug_info['estimated_input_tokens'] = estimated_tokens
        
        if self.verbose:
            print(f"      Total prompt length: {len(prompt_text)} chars (~{estimated_tokens} tokens)")
        
        # Check if we're likely to exceed context window
        if estimated_tokens > 6000:
            logger.warning(f"[{self.hospital_id}] Prompt may be too long ({estimated_tokens} tokens)")
        
        try:
            if self.verbose:
                print(f"\n      ⏳ Calling LLM...")
            
            start = time.time()
            response = self.llm.complete(prompt_text)
            gen_time = time.time() - start
            
            debug_info['generation_time'] = gen_time
            debug_info['response_length'] = len(response.text)
            debug_info['generation_status'] = 'success'
            
            if self.verbose:
                print(f"      ✅ Generation complete in {gen_time:.2f}s")
                print(f"      Response length: {len(response.text)} chars")
                print(f"      Response preview:")
                print(f"      {response.text[:500]}...")
            
            # FIXED: Validate response has expected sections
            response_text = response.text
            if not self._validate_response(response_text):
                logger.warning(f"[{self.hospital_id}] Response may be incomplete, using extracted entities")
                # Fall back to extracted entities
                response_text = self._build_fallback_response(records, debug_info)
            
            return response_text
            
        except Exception as e:
            debug_info['generation_status'] = f'error: {str(e)}'
            logger.error(f"LLM generation failed: {e}")
            # Return extracted data as fallback
            return self._build_fallback_response(records, debug_info)
    
    def _truncate_record_smartly(self, content: str, max_chars: int = 2000) -> str:
        """Truncate record while preserving important sections"""
        if len(content) <= max_chars:
            return content
        
        # Try to keep the structure: Demographics, Conditions, Medications, Procedures
        important_sections = [
            '=== PATIENT DEMOGRAPHICS ===',
            '=== CONDITIONS DOCUMENTED ===',
            '=== MEDICATIONS ===',
            '=== PROCEDURES PERFORMED ===',
            '=== CLINICAL SUMMARY ==='
        ]
        
        # Extract important sections
        extracted_parts = []
        remaining_chars = max_chars
        
        for section in important_sections:
            if section in content and remaining_chars > 200:
                start = content.find(section)
                # Find next section or end
                end = len(content)
                for next_section in important_sections:
                    if next_section != section:
                        next_start = content.find(next_section, start + len(section))
                        if next_start > start:
                            end = min(end, next_start)
                
                section_content = content[start:end].strip()
                # Limit each section
                section_max = min(400, remaining_chars)
                if len(section_content) > section_max:
                    section_content = section_content[:section_max] + "..."
                
                extracted_parts.append(section_content)
                remaining_chars -= len(section_content)
        
        if extracted_parts:
            return "\n\n".join(extracted_parts)
        
        # Fallback: simple truncation
        return content[:max_chars] + "\n[...truncated...]"
    
    def _validate_response(self, response: str) -> bool:
        """Check if response has expected sections"""
        required_markers = ['PATIENT COUNT:', 'CONDITIONS', 'TREATMENTS']
        found = sum(1 for marker in required_markers if marker in response.upper())
        return found >= 2  # At least 2 of 3 markers
    
    def _build_fallback_response(self, records: List[Dict], debug_info: Dict) -> str:
        """Build response from extracted entities when LLM fails"""
        entities = self._extract_entities_from_content(records)
        
        conditions = entities['conditions'][:5]
        treatments = entities['treatments'][:5]
        procedures = entities['procedures'][:3]
        
        response = f"""PATIENT COUNT: {len(records)}
CONDITIONS FOUND:
{chr(10).join('- ' + c for c in conditions) if conditions else '- None documented'}
TREATMENTS FOUND:
{chr(10).join('- ' + t for t in treatments) if treatments else '- None documented'}
PROCEDURES FOUND:
{chr(10).join('- ' + p for p in procedures) if procedures else '- None documented'}
KEY CLINICAL NOTES: Extracted from patient records (LLM extraction fallback)."""
        
        debug_info['used_fallback'] = True
        return response
    
    def query_with_dp(self, query: str) -> DPProtectedResult:
        """Execute query with DP protection and full tracing."""
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🏥 [{self.hospital_id}] QUERY WITH DP + TRACING")
            print(f"{'='*60}")
            print(f"   Privacy: ε={self.dp_epsilon}, k={self.dp_k_threshold}")
        
        # Step 1: Retrieve with tracing
        records, debug_info = self._retrieve_and_trace(query)
        raw_patient_count = len(records)
        
        # Step 2: Extract entities directly from content
        entities = self._extract_entities_from_content(records)
        
        if self.verbose:
            print(f"\n   📋 EXTRACTED ENTITIES:")
            print(f"      Conditions: {entities['conditions'][:5]}")
            print(f"      Treatments: {entities['treatments'][:5]}")
            print(f"      Procedures: {entities['procedures'][:3]}")
        
        # Step 3: Generate response with tracing
        raw_response = self._generate_with_trace(query, records, debug_info)
        
        # Step 4: Apply DP protection
        if self.verbose:
            print(f"\n   🔒 APPLYING DIFFERENTIAL PRIVACY:")
        
        dp_result = self.dp_protector.protect_hospital_response(
            hospital_id=self.hospital_id,
            patient_count=raw_patient_count,
            conditions=entities['conditions'],
            treatments=entities['treatments'],
            procedures=entities['procedures'],
            age_distribution=entities['age_distribution'],
            synthesized_response=raw_response
        )
        
        if self.verbose:
            print(f"      Raw count: {raw_patient_count} → DP count: {dp_result['dp_patient_count']}")
        
        # Calculate confidence
        confidence = 0.0
        if records:
            confidence = min(sum(r['score'] for r in records) / len(records), 1.0)
        
        latency = time.time() - start_time
        
        return DPProtectedResult(
            hospital_id=self.hospital_id,
            dp_patient_count=dp_result['dp_patient_count'],
            dp_response=dp_result['dp_response'],
            dp_conditions=dp_result['dp_conditions'],
            dp_treatments=dp_result['dp_treatments'],
            dp_procedures=dp_result['dp_procedures'],
            dp_age_distribution=dp_result['dp_age_distribution'],
            confidence_score=confidence,
            latency_seconds=latency,
            privacy_params=dp_result['privacy_guarantee'],
            raw_patient_count=raw_patient_count,
            retrieval_debug=debug_info
        )
    
    def get_stats(self) -> Dict:
        """Get hospital statistics"""
        return {
            'hospital_id': self.hospital_id,
            'vector_count': self.chroma_collection.count(),
            'collection_name': self.collection_name,
            'dp_epsilon': self.dp_epsilon,
            'dp_k_threshold': self.dp_k_threshold
        }
    
    def shutdown(self):
        """Cleanup"""
        if self.phoenix:
            try:
                self.phoenix.shutdown()
            except:
                pass
        logger.info(f"🔌 {self.hospital_id} shutdown")
