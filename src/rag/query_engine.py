"""
FRAG-MED Query Engine - PRODUCTION VERSION (Fixed)
Compatibility fix: Added back show_full_context parameter
"""
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List
import sys

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    QueryBundle,
    get_response_synthesizer,
    PromptTemplate
)
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.query_engine import RetrieverQueryEngine, CitationQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import config
from src.observability import PhoenixObservability
from src.observability.phoenix_setup import PerformanceMonitor

logger = logging.getLogger(__name__)


# ANTI-HALLUCINATION PROMPT - Forces data extraction, prevents generic knowledge
QA_PROMPT_TEMPLATE = """DATA EXTRACTION SYSTEM - Extract from records ONLY. NO general medical knowledge.

Records:
{context_str}

🚫 FORBIDDEN:
- General medical knowledge
- Phrases like "typically", "generally", "standard protocol"
- Claims without patient IDs

✅ REQUIRED FOR EVERY FINDING:
- Start with PATIENT_XXXXXXXX
- Extract EXACT text from === SECTIONS === only
- If no data found: state clearly

Query: {query_str}

RESPONSE FORMAT (use for EACH relevant patient):

PATIENT_XXXXXXXX (Date):
- Diagnosis: [exact from === CONDITIONS DOCUMENTED ===]
- Medications: [exact from === MEDICATIONS ===]  
- Procedures: [exact from === PROCEDURES PERFORMED ===]
- Key Observations: [specific values from === OBSERVATIONS ===]

Extract data now:"""


class ParentDocumentRetriever(BaseRetriever):
    """Optimized retriever with section extraction"""
    
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        parent_docs_dir: Path,
        similarity_top_k: int = 3,
        max_tokens: int = 1200,
        extract_sections_only: bool = True,
        verbose: bool = True
    ):
        super().__init__()
        self._vector_retriever = vector_retriever
        self._parent_docs_dir = Path(parent_docs_dir)
        self._similarity_top_k = similarity_top_k
        self._max_tokens = max_tokens
        self._extract_sections_only = extract_sections_only
        self._verbose = verbose
        self._parent_doc_cache = {}
        
        if not self._parent_docs_dir.exists():
            raise FileNotFoundError(f"Parent docs directory not found: {self._parent_docs_dir}")
    
    def _extract_structured_sections(self, content: str) -> str:
        """Extract only === SECTIONS === from parent document"""
        keep_sections = {
            "=== PATIENT DEMOGRAPHICS ===",
            "=== PATIENT-LEVEL DEVICES ===",
            "=== ENCOUNTER INFORMATION ===",
            "=== CONDITIONS DOCUMENTED ===",
            "=== PROCEDURES PERFORMED ===",
            "=== MEDICATIONS ===",
            "=== OBSERVATIONS ==="
        }
        
        result = []
        in_section = False
        current_lines = []
        
        for line in content.split('\n'):
            if any(section in line for section in keep_sections):
                if current_lines:
                    result.extend(current_lines)
                current_lines = [line]
                in_section = True
            elif line.strip().startswith("==="):
                if in_section and current_lines:
                    result.extend(current_lines)
                current_lines = []
                in_section = False
            elif in_section:
                current_lines.append(line)
        
        if current_lines:
            result.extend(current_lines)
        
        return '\n'.join(result)
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate to max tokens"""
        max_chars = max_tokens * 4
        if len(content) > max_chars:
            return content[:max_chars] + "\n[Truncated]"
        return content
    
    def _load_parent_doc(self, parent_doc_id: str) -> Optional[Dict]:
        """Load parent document with caching"""
        if parent_doc_id in self._parent_doc_cache:
            return self._parent_doc_cache[parent_doc_id]
        
        for batch_dir in sorted(self._parent_docs_dir.glob("batch_*")):
            file_path = batch_dir / f"{parent_doc_id}.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        parent_doc = json.load(f)
                    self._parent_doc_cache[parent_doc_id] = parent_doc
                    return parent_doc
                except Exception as e:
                    logger.error(f"Error loading {parent_doc_id}: {e}")
                    return None
        return None
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve and optimize parent documents"""
        child_nodes = self._vector_retriever.retrieve(query_bundle)
        
        parent_nodes = []
        seen_parent_ids = set()
        
        for child_node in child_nodes:
            parent_doc_id = child_node.node.metadata.get('parent_doc_id')
            
            if not parent_doc_id or parent_doc_id in seen_parent_ids:
                continue
            
            seen_parent_ids.add(parent_doc_id)
            parent_doc = self._load_parent_doc(parent_doc_id)
            
            if parent_doc:
                content = parent_doc['content']
                
                if self._extract_sections_only:
                    content = self._extract_structured_sections(content)
                
                content = self._truncate_content(content, self._max_tokens)
                
                parent_node = TextNode(
                    text=content,
                    id_=parent_doc_id,
                    metadata={
                        **child_node.node.metadata,
                        'source_type': 'parent_document',
                        'parent_doc_id': parent_doc_id
                    }
                )
                
                parent_nodes.append(NodeWithScore(node=parent_node, score=child_node.score))
        
        if self._verbose:
            logger.info(f"Loaded {len(parent_nodes)} parent documents")
        
        return parent_nodes


class CentralRAGSystem:
    """Production RAG system with anti-hallucination measures"""
    
    def __init__(
        self,
        enable_phoenix: bool = True,
        launch_server: bool = True,
        verbose: bool = True,
        use_citations: bool = False
    ):
        self.verbose = verbose
        self.use_citations = use_citations
        
        logger.info("="*80)
        logger.info("FRAG-MED PRODUCTION VERSION")
        logger.info("Anti-Hallucination + Performance Optimized")
        logger.info("="*80)

        self.phoenix = None
        self.performance_monitor = None
        
        if not config.validate_local_models():
            raise FileNotFoundError("Local models not found")
        
        if enable_phoenix and config.ENABLE_PHOENIX:
            try:
                self.phoenix = PhoenixObservability(
                    project_name="frag-med-central",
                    phoenix_host=config.PHOENIX_HOST,
                    phoenix_port=config.PHOENIX_PORT,
                    enable_tracing=True,
                    launch_server=launch_server
                )
                self.performance_monitor = PerformanceMonitor(self.phoenix)
                logger.info(f"📊 Phoenix Dashboard: {self.phoenix.get_dashboard_url()}")
            except Exception as e:
                logger.warning(f"Failed to initialize Phoenix: {e}")
                logger.info("Continuing without Phoenix observability")
                self.phoenix = None
        
        self._load_chromadb()
        self._setup_embeddings()
        self._setup_llm()
        self._build_index()
        self._create_query_engine()
        
        logger.info("="*80)
        logger.info("✅ SYSTEM READY")
        logger.info("="*80)
        logger.info(f"🎯 Top-K: {config.SIMILARITY_TOP_K}")
        logger.info(f"🌡️  Temperature: {config.LLM_TEMPERATURE} (0.0 = no hallucination)")
        logger.info(f"🔍 Section extraction: {config.EXTRACT_SECTIONS_ONLY}")
        logger.info(f"⏱️  Target latency: <15s")
        logger.info("="*80 + "\n")
    
    def _load_chromadb(self):
        logger.info("📦 Loading ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=str(config.CHROMADB_DIR))
        self.chroma_collection = chroma_client.get_collection(config.CHROMADB_COLLECTION)
        logger.info(f"✓ Loaded {self.chroma_collection.count():,} vectors")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
    
    def _setup_embeddings(self):
        logger.info("🔤 Loading embeddings...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=str(config.EMBEDDING_MODEL_PATH),
            embed_batch_size=config.EMBEDDING_BATCH_SIZE
        )
        Settings.embed_model = self.embed_model
        logger.info("✓ Embeddings loaded")
    
    def _setup_llm(self):
        logger.info("🧠 Connecting to LLM...")
        self.llm = Ollama(
            model=config.LLM_MODEL_NAME,
            request_timeout=config.LLM_TIMEOUT,
            temperature=config.LLM_TEMPERATURE,
            context_window=config.LLM_CONTEXT_WINDOW,
            system_prompt="You are a data extraction system. Extract ONLY from provided records. Never use general medical knowledge.",
            additional_kwargs={"num_predict": config.LLM_MAX_TOKENS}
        )
        Settings.llm = self.llm
        logger.info(f"✓ LLM connected (temp={config.LLM_TEMPERATURE})")
    
    def _build_index(self):
        logger.info("🔨 Building index...")
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=storage_context
        )
        logger.info("✓ Index built")
    
    def _create_query_engine(self):
        logger.info("⚙️  Creating query engine...")
        
        base_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=config.SIMILARITY_TOP_K
        )
        
        self.parent_retriever = ParentDocumentRetriever(
            vector_retriever=base_retriever,
            parent_docs_dir=config.PARENT_DOCS_DIR,
            similarity_top_k=config.SIMILARITY_TOP_K,
            max_tokens=config.PARENT_DOC_MAX_TOKENS,
            extract_sections_only=config.EXTRACT_SECTIONS_ONLY,
            verbose=self.verbose
        )
        
        qa_prompt = PromptTemplate(QA_PROMPT_TEMPLATE)
        response_synthesizer = get_response_synthesizer(
            response_mode=config.RESPONSE_MODE,
            text_qa_template=qa_prompt,
            verbose=self.verbose
        )
        
        if self.use_citations:
            self.query_engine = CitationQueryEngine.from_args(
                retriever=self.parent_retriever,
                response_synthesizer=response_synthesizer,
                citation_chunk_size=512
            )
        else:
            self.query_engine = RetrieverQueryEngine(
                retriever=self.parent_retriever,
                response_synthesizer=response_synthesizer
            )
        
        logger.info("✓ Query engine ready")
    
    def _validate_response(self, response_text: str, source_nodes: List) -> Dict:
        """Validate response for hallucination"""
        # Extract patient IDs from sources
        source_patient_ids = set()
        for node in source_nodes:
            patient_id = node.metadata.get('patient_pseudonym')
            if patient_id:
                source_patient_ids.add(patient_id)
        
        # Check citations
        has_citations = any(pid in response_text for pid in source_patient_ids)
        
        # Check for hallucination indicators
        hallucination_phrases = [
            "typically", "generally", "usually", "standard protocol",
            "common treatment", "most patients", "in general", "commonly"
        ]
        has_generic = any(phrase in response_text.lower() for phrase in hallucination_phrases)
        
        return {
            'has_citations': has_citations,
            'has_generic_phrases': has_generic,
            'likely_hallucination': has_generic and not has_citations,
            'patient_ids_found': list(source_patient_ids)
        }
    
    def query(
        self, 
        query_text: str, 
        show_sources: bool = True, 
        show_full_context: bool = False
    ) -> Dict:
        """Query the RAG system with anti-hallucination measures"""
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*80)
            print(f"🔍 QUERY: {query_text}")
            print("="*80)
        
        try:
            response = self.query_engine.query(query_text)
            latency = time.time() - start_time
            num_retrieved = len(response.source_nodes)
            num_tokens = len(response.response.split()) * 1.3
            
            # Validate for hallucination
            validation = self._validate_response(response.response, response.source_nodes)
            
            if self.performance_monitor:
                self.performance_monitor.record_query(
                    latency=latency,
                    num_tokens=int(num_tokens),
                    num_retrieved=num_retrieved
                )
            
            if self.verbose:
                status = "✅" if latency < 15 else "⚠️"
                print(f"\n{status} RESPONSE (latency: {latency:.2f}s):")
                print("-"*80)
                print(response.response)
                print("-"*80)
                
                # Hallucination warning
                if validation['likely_hallucination']:
                    print("\n⚠️  HALLUCINATION WARNING:")
                    print("  - Response contains generic medical knowledge")
                    print("  - No patient IDs cited")
                    print("  - May not be from retrieved records")
                elif not validation['has_citations']:
                    print("\n⚠️  WARNING: No patient IDs cited in response")
            
            if show_sources and response.source_nodes:
                print(f"\n📚 SOURCES: {num_retrieved} documents")
                for i, node in enumerate(response.source_nodes, 1):
                    m = node.metadata
                    print(f"  [{i}] {m.get('patient_pseudonym')} ({m.get('temporal_quarter')})")
                    
                    if show_full_context:
                        print(f"\n    Full Parent Document Content:")
                        print("    " + "-"*76)
                        lines = node.node.text.split('\n')
                        for line in lines[:50]:  # Show first 50 lines
                            print(f"    {line}")
                        if len(lines) > 50:
                            print(f"    ... ({len(lines) - 50} more lines)")
                        print("    " + "-"*76)
            
            return {
                'response': response.response,
                'source_nodes': response.source_nodes,
                'latency': latency,
                'num_retrieved': num_retrieved,
                'estimated_tokens': int(num_tokens),
                'under_target': latency < 15.0,
                'validation': validation
            }
            
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return {
                'response': f"Error: {str(e)}",
                'latency': time.time() - start_time,
                'under_target': False,
                'validation': {'likely_hallucination': True}
            }
    
    def get_performance_stats(self) -> Dict:
        if self.performance_monitor:
            return self.performance_monitor.get_statistics()
        return {}
    
    def clear_cache(self):
        if hasattr(self, 'parent_retriever'):
            self.parent_retriever._parent_doc_cache.clear()
    
    def shutdown(self):
        logger.info("Shutting down...")
        if self.performance_monitor:
            stats = self.get_performance_stats()
            logger.info("\n📊 Session Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
        self.clear_cache()
        if self.phoenix:
            try:
                self.phoenix.shutdown()
            except Exception as e:
                logger.debug(f"Phoenix shutdown warning: {e}")


if __name__ == "__main__":
    import coloredlogs
    coloredlogs.install(level='INFO')
    
    rag = CentralRAGSystem(enable_phoenix=False, verbose=True)
    
    query = "Find patients with acute bronchitis and describe their diagnostic procedures and medications."
    result = rag.query(query, show_sources=True, show_full_context=False)
    
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}")
    val = result['validation']
    print(f"Has patient citations: {val['has_citations']}")
    print(f"Has generic phrases: {val['has_generic_phrases']}")
    print(f"Likely hallucination: {val['likely_hallucination']}")
    
    if val['likely_hallucination']:
        print(f"\n❌ FAIL: Response appears to be hallucinated")
    elif val['has_citations']:
        print(f"\n✅ PASS: Response extracted from patient records")
    else:
        print(f"\n⚠️  WARNING: Response may need review")
    
    rag.shutdown()
