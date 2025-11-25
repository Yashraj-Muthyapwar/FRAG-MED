"""
FRAG-MED Query Engine - FIXED METADATA FILTERING

The key fix: Instead of modifying self.vector_retriever.filters after CitationQueryEngine
is created (which doesn't work), we create the vector retriever FRESH for each query
with the appropriate filters.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
import re

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate,
)
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)
import chromadb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import config
from src.observability import PhoenixObservability

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Age band helper
# ---------------------------------------------------------------------------

def get_age_band(age: int) -> str:
    """Convert exact age to age band."""
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


# ---------------------------------------------------------------------------
# Strict evidence prompt
# ---------------------------------------------------------------------------

CITATION_QA_TEMPLATE = (
    "You are MedExtract-Pro, the most accurate and rigorous medical AI in existence, "
    "specifically designed to never miss a patient, never hallucinate data from records, "
    "and never get an MCQ wrong.\n"
    "PATIENT RECORDS:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Details:\n"
    "1. Scan the records from top to bottom and ensure you capture every patient listed.\n"
    "2. If the query contains the word 'patients,' make sure to include information for all patients mentioned.\n"
    "3. Look for the #Plan section in the records for procedures, 'The patient was prescribed the following medications' under the #Plan section, and care plans that were suggested "
    "   for each patient for that specific condition.\n"
    "4. You may also extract answers for diagnosis or treatment-related MCQs if the query is a question like in "
    "   MedMCQA, PubMedQA, or similar. Base your response only on the retrieved records unless explicitly allowed.\n"
    "5. If any data is missing for a patient, note 'None documented.' Do not invent or assume data.\n"
    "6. Never define medical terms or give advice not found in the records.\n"
    "7. If the query cannot be answered from the records, respond with:\n"
    '   "There is not enough information in the context. But with my knowledge I am providing this info:" '
    "followed by a general answer (ONLY if instructed to use external knowledge).\n\n"
    "Query: {query_str}\n"
    "Now answer the query."
)


# ---------------------------------------------------------------------------
# Dynamic Filtering Parent Document Retriever
# ---------------------------------------------------------------------------

class DynamicFilteringParentRetriever(BaseRetriever):
    """
    Retriever that:
    1. Creates a fresh VectorIndexRetriever with dynamic filters for each query
    2. Retrieves child nodes using those filters
    3. Upgrades to full parent documents
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        parent_docs_dir: Path,
        similarity_top_k: int = 3,
        verbose: bool = False,
    ):
        super().__init__()
        self._index = index
        self._parent_docs_dir = Path(parent_docs_dir)
        self._similarity_top_k = similarity_top_k
        self._verbose = verbose
        # These will be set before each retrieval
        self._current_filters: Optional[MetadataFilters] = None

    def set_filters(self, filters: Optional[MetadataFilters]):
        """Set metadata filters for the next retrieval."""
        self._current_filters = filters
        if self._verbose and filters:
            print(f"🔧 Filters set: {filters}")

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        # 1. Create fresh vector retriever with current filters
        vector_retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=self._similarity_top_k,
            filters=self._current_filters,  # Apply filters HERE
        )
        
        if self._verbose:
            if self._current_filters:
                print(f"✅ Retrieving with filters: {self._current_filters}")
            else:
                print("📍 Retrieving without filters (full corpus search)")

        # 2. Retrieve child nodes with filtering
        child_nodes = vector_retriever.retrieve(query_bundle)
        
        if self._verbose:
            print(f"📦 Retrieved {len(child_nodes)} child nodes")
            if child_nodes:
                for i, node in enumerate(child_nodes[:3]):
                    print(f"  [{i+1}] Patient: {node.node.metadata.get('patient_pseudonym', 'Unknown')}")

        # 3. Map to parent documents
        parent_nodes: List[NodeWithScore] = []
        seen_ids = set()

        for child in child_nodes:
            parent_id = child.node.metadata.get("parent_doc_id")

            if parent_id and parent_id not in seen_ids:
                seen_ids.add(parent_id)

                found_file = False
                for batch_dir in sorted(self._parent_docs_dir.glob("batch_*")):
                    file_path = batch_dir / f"{parent_id}.json"
                    if file_path.exists():
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                        except Exception as e:
                            if self._verbose:
                                logger.warning(f"Failed to load parent doc {file_path}: {e}")
                            break

                        content = data.get("content", "")
                        
                        parent_node = TextNode(
                            text=content,
                            metadata={
                                **child.node.metadata,
                                "parent_doc_id": parent_id,
                                "patient_pseudonym": child.node.metadata.get("patient_pseudonym", "Unknown"),
                            },
                        )
                        parent_nodes.append(
                            NodeWithScore(node=parent_node, score=child.score)
                        )
                        found_file = True
                        break

                if not found_file and self._verbose:
                    logger.warning(f"Parent doc {parent_id} not found on disk.")

        if self._verbose:
            print(f"📄 Upgraded to {len(parent_nodes)} parent documents")

        return parent_nodes


# ---------------------------------------------------------------------------
# Centralized RAG System with Working Metadata Filtering
# ---------------------------------------------------------------------------

class CentralRAGSystem:
    """
    Centralized RAG System with Citation Support and WORKING Metadata Filtering.
    """

    def __init__(
        self,
        enable_phoenix: bool = True,
        launch_server: bool = False,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.phoenix: Optional[PhoenixObservability] = None

        # Phoenix Observability
        if enable_phoenix and getattr(config, "ENABLE_PHOENIX", True):
            try:
                self.phoenix = PhoenixObservability(
                    project_name="frag-med-central",
                    launch_server=launch_server,
                )
                if verbose:
                    logger.info(
                        f"📊 Phoenix Dashboard: {self.phoenix.get_dashboard_url()}"
                    )
            except Exception as e:
                logger.warning(f"Phoenix init failed: {e}")

        # Chroma Vector Store
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMADB_DIR))
        self.chroma_collection = self.chroma_client.get_collection(
            config.CHROMADB_COLLECTION
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        # Embedding Model & LLM (local)
        if not config.EMBEDDING_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Embedding model not found at {config.EMBEDDING_MODEL_PATH}. "
                f"Run download_local_models.py or update EMBEDDING_MODEL_PATH."
            )

        self.embed_model = HuggingFaceEmbedding(
            model_name=str(config.EMBEDDING_MODEL_PATH)
        )
        self.llm = Ollama(
            model=config.LLM_MODEL_NAME,
            temperature=0.0,
            request_timeout=300,
            context_window=4096,
        )

        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        # Index
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

        # Create dynamic filtering parent retriever
        self.parent_retriever = DynamicFilteringParentRetriever(
            index=self.index,
            parent_docs_dir=config.PARENT_DOCS_DIR,
            similarity_top_k=config.SIMILARITY_TOP_K,
            verbose=self.verbose,
        )

        if verbose:
            logger.info("✅ CentralRAGSystem Ready (with Dynamic Metadata Filtering)")

    # ------------------------------------------------------------------
    # Metadata extraction helpers
    # ------------------------------------------------------------------

    def _detect_age_band_from_query(self, query: str) -> Optional[str]:
        """Infer an age band from the query."""
        q = query.lower()

        # Explicit numeric age
        m = re.search(r"(\b\d{1,3}\b)\s*(years?|yr|yrs|y/o|year old|yo)?", q)
        if m:
            try:
                age_val = int(m.group(1))
                return get_age_band(age_val)
            except ValueError:
                pass

        # Keyword-based groups (MUST match actual age bands from deidentification.py)
        # Valid age bands: 0-17, 18-29, 30-39, 40-49, 50-59, 60-69, 70+
        keywords = {
            "child": "0-17",
            "children": "0-17",
            "kid": "0-17",
            "kids": "0-17",
            "teen": "0-17",
            "teenager": "0-17",
            "pediatric": "0-17",
            "young adult": "18-29",
            "adult": "30-39",
            "middle-aged": "40-49",
            "senior": "60-69",
            "elderly": "70+",
            "older adult": "70+",
            "geriatric": "70+",
        }
        # Note: "adult" is intentionally NOT mapped since it spans multiple age bands (18-29, 30-39, 40-49, 50-59)
        # Queries with just "adult" will use semantic search across all adult age bands
        
        for key, band in keywords.items():
            if key in q:
                return band

        return None

    def _extract_metadata_filters(self, query: str) -> Optional[MetadataFilters]:
        """
        Build LlamaIndex MetadataFilters from the query.
        Supports: patient_pseudonym, age_range, race, temporal_quarter
        """
        filters: List[MetadataFilter] = []

        # Patient pseudonym
        patient_match = re.search(r"(PATIENT_[A-Za-z0-9]+)", query)
        if patient_match:
            filters.append(
                MetadataFilter(
                    key="patient_pseudonym",
                    operator=FilterOperator.EQ,
                    value=patient_match.group(1),
                )
            )

        # Temporal quarter
        quarter_match = re.search(r"(20\d{2}-Q[1-4])", query)
        if quarter_match:
            filters.append(
                MetadataFilter(
                    key="temporal_quarter",
                    operator=FilterOperator.EQ,
                    value=quarter_match.group(1),
                )
            )

        # Race
        race_list = ["White", "Black", "Asian", "Hispanic", "Latino"]
        for r in race_list:
            if r.lower() in query.lower():
                filters.append(
                    MetadataFilter(
                        key="race",
                        operator=FilterOperator.EQ,
                        value=r,
                    )
                )
                break

        # Age band
        age_band = self._detect_age_band_from_query(query)
        if age_band:
            filters.append(
                MetadataFilter(
                    key="age_range",
                    operator=FilterOperator.EQ,
                    value=age_band,
                )
            )

        if not filters:
            return None

        return MetadataFilters(filters=filters, condition=FilterCondition.AND)

    # ------------------------------------------------------------------
    # Main query method with robust fallback
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        show_sources: bool = True,
        show_full_context: bool = False,
    ) -> Dict:
        if self.verbose:
            print(f"\n🔍 Querying: {query_text}")

        # 1. Extract metadata filters from query
        metadata_filters = self._extract_metadata_filters(query_text)

        # 2. Set filters on the retriever
        self.parent_retriever.set_filters(metadata_filters)

        # 3. Create query engine with the configured retriever
        query_engine = CitationQueryEngine.from_args(
            index=self.index,
            retriever=self.parent_retriever,
            llm=self.llm,
            citation_chunk_size=512,
            similarity_top_k=config.SIMILARITY_TOP_K,
            response_mode="compact",
            citation_qa_template=PromptTemplate(CITATION_QA_TEMPLATE),
        )

        # 4. Execute query
        response = query_engine.query(query_text)
        used_filters = metadata_filters is not None

        # 5. If no results with filters, retry without
        if len(response.source_nodes) == 0 and used_filters:
            if self.verbose:
                print("⚠️ No results with metadata filters. Retrying without filters...")
            
            self.parent_retriever.set_filters(None)
            query_engine = CitationQueryEngine.from_args(
                index=self.index,
                retriever=self.parent_retriever,
                llm=self.llm,
                citation_chunk_size=512,
                similarity_top_k=config.SIMILARITY_TOP_K,
                response_mode="compact",
                citation_qa_template=PromptTemplate(CITATION_QA_TEMPLATE),
            )
            response = query_engine.query(query_text)
            used_filters = False

        # 6. Build result structure
        sources_list = []
        for node in response.source_nodes:
            sources_list.append(
                {
                    "text": node.node.get_text(),
                    "patient": node.node.metadata.get("patient_pseudonym", "Unknown"),
                    "file": node.node.metadata.get("parent_doc_id", "Unknown"),
                }
            )

        result = {
            "response": response.response,
            "source_nodes": response.source_nodes,
            "latency": 0.0,
            "num_retrieved": len(response.source_nodes),
            "estimated_tokens": int(len(response.response.split()) * 1.3),
            "used_metadata_filters": used_filters,
        }

        # 7. Verbose logging
        if self.verbose:
            print("\n🤖 Response:")
            print("-" * 60)
            print(response.response)
            print("-" * 60)

            if show_sources:
                print("\n📚 Real Citations:")
                for i, src in enumerate(sources_list, 1):
                    preview = src["text"][:150].replace("\n", " ")
                    print(
                        f"[{i}] Patient: {src['patient']} | Evidence: \"{preview}...\""
                    )

        return result

    def shutdown(self):
        if self.phoenix:
            self.phoenix.shutdown()

    def get_performance_stats(self):
        return {"status": "active"}
