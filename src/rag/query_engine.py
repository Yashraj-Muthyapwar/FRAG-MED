"""
FRAG-MED Query Engine - FINAL FIX
Fixes:
1. Corrects argument name to 'citation_qa_template' to apply custom prompt.
2. Maintains single-pass compact mode.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
import chromadb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import config
from src.observability import PhoenixObservability

logger = logging.getLogger(__name__)

# --- STRICT EVIDENCE PROMPT ---
CITATION_QA_TEMPLATE = (
    "You are MedExtract-Pro, the most accurate and rigorous medical AI in existence, specifically designed to never miss a patient, never hallucinate data from records, and never get an MCQ wrong.\n"
    "PATIENT RECORDS:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Details:\n"
    "1. Scan the records from top to bottom and ensure you capture every patient listed.\n"
    "2. If the query contains the word 'patients,' make sure to include information for all patients mentioned.\n"
    "3. Look for the #Plan section in the records for procedures, medications, and care plans that were suggested for each patient.\n"
    "4. You may also extract answers for diagnosis or treatment-related MCQs if the query is a question like in MedMCQA, PubMedQA, or similar. Base your response only on the retrieved records unless explicitly allowed.\n"
    "5. If any data is missing for a patient, note 'None documented.' Do not invent or assume data.\n"
    "6. Never define medical terms or give advice not found in the records.\n"
    """7. If the query cannot be answered from the records, respond with:
   "There is not enough information in the context. But with my knowledge I am providing this info:" followed by a general answer (ONLY if instructed to use external knowledge)."""
    "Query: {query_str}\n"
    "Now answer the query."
)


class ParentDocumentRetriever(BaseRetriever):
    """Retriever that fetches full parent documents based on vector search"""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        parent_docs_dir: Path,
        verbose: bool = False,
    ):
        super().__init__()
        self._vector_retriever = vector_retriever
        self._parent_docs_dir = Path(parent_docs_dir)
        self._verbose = verbose

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        # 1. Retrieve small child chunks
        child_nodes = self._vector_retriever.retrieve(query_bundle)

        # 2. Map to parent documents
        parent_nodes: List[NodeWithScore] = []
        seen_ids = set()

        for child in child_nodes:
            parent_id = child.node.metadata.get("parent_doc_id")

            # Avoid duplicates
            if parent_id and parent_id not in seen_ids:
                seen_ids.add(parent_id)

                # Load full parent content
                found_file = False
                for batch_dir in sorted(self._parent_docs_dir.glob("batch_*")):
                    file_path = batch_dir / f"{parent_id}.json"
                    if file_path.exists():
                        import json

                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                            # Create a fresh node with the full content
                            parent_node = TextNode(
                                text=data["content"],
                                metadata={
                                    **child.node.metadata,
                                    "parent_doc_id": parent_id,
                                    "patient_pseudonym": data.get(
                                        "metadata", {}
                                    ).get("patient_id", "Unknown"),
                                },
                            )
                            parent_nodes.append(
                                NodeWithScore(node=parent_node, score=child.score)
                            )
                            found_file = True
                            break

                if not found_file and self._verbose:
                    logger.warning(f"Parent doc {parent_id} not found on disk.")

        return parent_nodes


class CentralRAGSystem:
    """
    Centralized RAG System with Citation Support
    """

    def __init__(
        self,
        enable_phoenix: bool = True,
        launch_server: bool = False,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.phoenix: Optional[PhoenixObservability] = None

        # Initialize Phoenix
        if enable_phoenix and config.ENABLE_PHOENIX:
            try:
                self.phoenix = PhoenixObservability(
                    project_name="frag-med-central", launch_server=launch_server
                )
                if verbose:
                    logger.info(
                        f"📊 Phoenix Dashboard: {self.phoenix.get_dashboard_url()}"
                    )
            except Exception as e:
                logger.warning(f"Phoenix init failed: {e}")

        # 1. Setup ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMADB_DIR))
        self.chroma_collection = self.chroma_client.get_collection(
            config.CHROMADB_COLLECTION
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        # 2. Setup Models (Local)
        if not config.EMBEDDING_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Embedding model not found. Run download_local_models.py"
            )

        self.embed_model = HuggingFaceEmbedding(
            model_name=str(config.EMBEDDING_MODEL_PATH)
        )
        self.llm = Ollama(
            model=config.LLM_MODEL_NAME,
            temperature=0.0,  # Strict factualness
            request_timeout=300,
            context_window=4096,
        )

        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        # 3. Build Index & Retriever
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=config.SIMILARITY_TOP_K,
        )

        self.parent_retriever = ParentDocumentRetriever(
            vector_retriever=self.vector_retriever,
            parent_docs_dir=config.PARENT_DOCS_DIR,
            verbose=self.verbose,
        )

        # 4. Setup Citation Engine
        # FIX: Use correct argument name 'citation_qa_template'
        self.query_engine = CitationQueryEngine.from_args(
            index=self.index,
            retriever=self.parent_retriever,
            llm=self.llm,
            citation_chunk_size=512,
            similarity_top_k=config.SIMILARITY_TOP_K,
            response_mode="compact",
            
            # CORRECTED LINE BELOW:
            citation_qa_template=PromptTemplate(CITATION_QA_TEMPLATE),
        )

        if verbose:
            logger.info("✅ CentralRAGSystem Ready (Strict Evidence Mode)")

    def query(
        self,
        query_text: str,
        show_sources: bool = True,
        show_full_context: bool = False,
    ) -> Dict:
        if self.verbose:
            print(f"\n🔍 Querying: {query_text}")

        # Execute Query
        response = self.query_engine.query(query_text)

        # Extract citations
        sources_list = []
        for node in response.source_nodes:
            sources_list.append(
                {
                    "text": node.node.get_text(),
                    "patient": node.node.metadata.get(
                        "patient_pseudonym", "Unknown"
                    ),
                    "file": node.node.metadata.get("parent_doc_id", "Unknown"),
                }
            )

        result = {
            "response": response.response,
            "source_nodes": response.source_nodes,
            "latency": 0.0,
            "num_retrieved": len(response.source_nodes),
            "estimated_tokens": len(response.response.split()) * 1.3,
        }

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
