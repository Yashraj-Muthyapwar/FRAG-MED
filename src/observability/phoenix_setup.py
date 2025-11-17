"""
Arize Phoenix Observability Integration
Updated for Phoenix 4.x with OpenTelemetry instrumentation
"""
import logging
from typing import Optional
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

logger = logging.getLogger(__name__)


class PhoenixObservability:
    """
    Manages Arize Phoenix observability for FRAG-MED
    
    Features:
    - LLM call tracing
    - Embedding generation tracking
    - Retrieval quality metrics
    - Latency monitoring
    - Token usage tracking
    """
    
    def __init__(
        self,
        project_name: str = "frag-med-central",
        phoenix_host: str = "127.0.0.1",
        phoenix_port: int = 6006,
        enable_tracing: bool = True,
        launch_server: bool = True
    ):
        """
        Initialize Phoenix observability
        
        Args:
            project_name: Name for the Phoenix project
            phoenix_host: Phoenix server host
            phoenix_port: Phoenix server port
            enable_tracing: Whether to enable tracing
        """
        self.project_name = project_name
        self.phoenix_host = phoenix_host
        self.phoenix_port = phoenix_port
        self.enable_tracing = enable_tracing
        self.launch_server = launch_server
        
        self.session = None
        self.tracer_provider = None
        self.instrumentor = None
        
        if self.enable_tracing:
            self._initialize_phoenix()

    def _initialize_phoenix(self):
        """Initialize Phoenix server and tracing with OpenTelemetry"""
        try:
            logger.info("Initializing Arize Phoenix observability...")
            
            if self.launch_server: # <--- ADD THIS CONDITIONAL CHECK
                # Launch Phoenix server
                logger.info(f"Starting Phoenix server on {self.phoenix_host}:{self.phoenix_port}...")
                self.session = px.launch_app(
                    host=self.phoenix_host,
                    port=self.phoenix_port
                )
                
                logger.info(
                    f"🔍 Phoenix UI available at: "
                    f"http://{self.phoenix_host}:{self.phoenix_port}"
                )
            else: # <--- ADD THIS BLOCK
                # Connect to existing Phoenix server
                logger.info("📡 Connecting to externally running Phoenix server...")
                server_url = f"http://{self.phoenix_host}:{self.phoenix_port}"
                self.session = px.Client(endpoint=server_url)
            
            # Register OpenTelemetry tracer provider - This is needed whether launching or connecting
            logger.info("Registering OpenTelemetry tracer provider...")
            self.tracer_provider = register(
                project_name=self.project_name,
                endpoint=f"http://{self.phoenix_host}:{self.phoenix_port}/v1/traces"
            )
            
            # Instrument LlamaIndex with OpenInference
            logger.info("Instrumenting LlamaIndex with OpenInference...")
            self.instrumentor = LlamaIndexInstrumentor()
            self.instrumentor.instrument(tracer_provider=self.tracer_provider)
            
            logger.info("✅ Phoenix observability initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Phoenix: {e}")
            logger.warning("Continuing without Phoenix observability")
            self.enable_tracing = False
            import traceback
            traceback.print_exc()
    
    def log_custom_metric(self, metric_name: str, value: float, metadata: dict = None):
        """
        Log custom metrics to Phoenix
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metadata: Additional metadata
        """
        if not self.enable_tracing:
            return
        
        try:
            # Custom metrics logging
            logger.info(f"Custom Metric - {metric_name}: {value}")
            if metadata:
                logger.info(f"Metadata: {metadata}")
        except Exception as e:
            logger.warning(f"Failed to log custom metric: {e}")
    
    def shutdown(self):
        """Shutdown Phoenix server and uninstrument"""
        logger.info("Shutting down Phoenix observability...")
        
        # Uninstrument LlamaIndex
        if self.instrumentor:
            try:
                self.instrumentor.uninstrument()
                logger.info("✓ LlamaIndex uninstrumented")
            except Exception as e:
                logger.warning(f"Error uninstrumenting LlamaIndex: {e}")
        
        # Close Phoenix session
        if self.session:
            try:
                px.close_app()
                logger.info("✓ Phoenix server shut down")
            except Exception as e:
                logger.warning(f"Error closing Phoenix: {e}")
    
    def get_dashboard_url(self) -> str:
        """Get the Phoenix dashboard URL"""
        return f"http://{self.phoenix_host}:{self.phoenix_port}"
    
    def export_traces(self, output_path: str):
        """
        Export traces to file for analysis
        
        Args:
            output_path: Path to save traces
        """
        if not self.enable_tracing:
            logger.warning("Tracing not enabled, cannot export traces")
            return
        
        try:
            logger.info(f"Exporting traces to {output_path}")
            # Phoenix export logic would go here
            logger.info("Note: Trace export requires additional implementation")
        except Exception as e:
            logger.error(f"Failed to export traces: {e}")
    
    @staticmethod
    def create_trace_metadata(
        patient_pseudonym: str = None,
        encounter_type: str = None,
        age_range: str = None,
        num_retrieved: int = None
    ) -> dict:
        """
        Create metadata for trace annotations
        
        Args:
            patient_pseudonym: Patient identifier
            encounter_type: Type of encounter
            age_range: Patient age range
            num_retrieved: Number of documents retrieved
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        if patient_pseudonym:
            metadata['patient_pseudonym'] = patient_pseudonym
        if encounter_type:
            metadata['encounter_type'] = encounter_type
        if age_range:
            metadata['age_range'] = age_range
        if num_retrieved:
            metadata['num_retrieved_documents'] = num_retrieved
        
        return metadata


class PerformanceMonitor:
    """Monitors system performance metrics"""
    
    def __init__(self, phoenix: PhoenixObservability):
        self.phoenix = phoenix
        self.query_count = 0
        self.total_latency = 0.0
        self.total_tokens = 0
    
    def record_query(
        self,
        latency: float,
        num_tokens: int,
        num_retrieved: int,
        query_type: str = "diagnostic"
    ):
        """
        Record query performance metrics
        
        Args:
            latency: Query latency in seconds
            num_tokens: Number of tokens used
            num_retrieved: Number of documents retrieved
            query_type: Type of query
        """
        self.query_count += 1
        self.total_latency += latency
        self.total_tokens += num_tokens
        
        # Log to Phoenix
        self.phoenix.log_custom_metric(
            "query_latency",
            latency,
            metadata={
                "query_type": query_type,
                "num_retrieved": num_retrieved,
                "num_tokens": num_tokens
            }
        )
    
    def get_statistics(self) -> dict:
        """Get performance statistics"""
        avg_latency = (
            self.total_latency / self.query_count 
            if self.query_count > 0 
            else 0.0
        )
        
        return {
            "total_queries": self.query_count,
            "average_latency": avg_latency,
            "total_tokens_used": self.total_tokens,
            "average_tokens_per_query": (
                self.total_tokens / self.query_count 
                if self.query_count > 0 
                else 0
            )
        }
