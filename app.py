#!/usr/bin/env python3

import streamlit as st
import sys
from pathlib import Path
import time
from datetime import datetime
import logging
import subprocess
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.query_engine import CentralRAGSystem

# Page configuration
st.set_page_config(
    page_title="FRAG-MED: Federated Medical Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Mode badges */
    .mode-badge-centralized {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .mode-badge-federated {
        display: inline-block;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    
    /* Status card in sidebar */
    .status-card-mini {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-online {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .status-offline {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .status-federated {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* Stats in sidebar */
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #e9ecef;
    }
    .stat-row:last-child {
        border-bottom: none;
    }
    .stat-label {
        color: #6c757d;
        font-size: 0.85rem;
    }
    .stat-value {
        font-weight: bold;
        color: #1f77b4;
    }
    
    /* Info boxes */
    .warning-box {
        background-color: #fff3cd;
        padding: 0.75rem 1rem;
        border-left: 4px solid #ffc107;
        margin: 0.75rem 0;
        border-radius: 0.25rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 0.75rem 1rem;
        border-left: 4px solid #28a745;
        margin: 0.75rem 0;
        border-radius: 0.25rem;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 0.75rem 1rem;
        border-left: 4px solid #17a2b8;
        margin: 0.75rem 0;
        border-radius: 0.25rem;
    }
    .privacy-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.75rem 0;
        font-size: 0.9rem;
    }
    
    .phoenix-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .phoenix-box a {
        color: white !important;
        text-decoration: underline;
    }
    .phoenix-active {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    /* Button styling */
    .stButton > button {
        transition: all 0.3s ease;
        border-radius: 8px;
        height: auto !important;
        white-space: normal !important;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Feature list in sidebar */
    .feature-list {
        font-size: 0.85rem;
        line-height: 1.6;
    }
    
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        # System state
        'rag_system': None,
        'federated_system': None,
        'system_mode': 'centralized',  # 'centralized' or 'federated'
        'system_initialized': False,
        'initialization_error': None,
        
        # Query state
        'query_history': [],
        'current_query': "",
        'run_query_flag': False,
        'last_result': None,
        
        # Phoenix state
        'phoenix_enabled': False,
        'phoenix_launched': False,
        'phoenix_server_running': False,
        
        # Federated-specific state
        'federated_epsilon': 1.0,
        'federated_k_threshold': 3,
        'selected_hospitals': [],
        'federated_hospitals_registered': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


def check_phoenix_server():
    """Check if Phoenix server is running on port 6006"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(('127.0.0.1', 6006))
        return result == 0
    except:
        return False
    finally:
        sock.close()


def launch_phoenix_server():
    """Launch Phoenix server if not already running"""
    if check_phoenix_server():
        logger.info("Phoenix server already running on port 6006")
        return True
    
    try:
        import phoenix as px
        logger.info("Launching Phoenix server...")
        px.launch_app(host="127.0.0.1", port=6006)
        time.sleep(2)
        
        if check_phoenix_server():
            logger.info("Phoenix server started successfully")
            return True
        else:
            logger.warning("Phoenix server may not have started properly")
            return False
            
    except Exception as e:
        logger.error(f"Failed to launch Phoenix: {e}")
        return False


def initialize_rag_system(
    mode: str,
    enable_phoenix: bool,
    launch_phoenix: bool = False,
    dp_epsilon: float = 1.0,
    dp_k_threshold: int = 3,
    hospital_ids: list = None
):
    """Initialize RAG system (centralized or federated)"""
    try:
        # Launch Phoenix if requested
        if enable_phoenix and launch_phoenix:
            phoenix_running = launch_phoenix_server()
            if phoenix_running:
                st.session_state.phoenix_server_running = True
            else:
                st.warning("⚠️ Could not start Phoenix server. Continuing without it.")
                st.session_state.phoenix_server_running = False
        
        if mode.lower() == "centralized":
            return CentralRAGSystem(
                enable_phoenix=enable_phoenix,
                launch_server=launch_phoenix,
                verbose=True
            )
        
        elif mode.lower() == "federated":
            # Try to import federated components
            try:
                from federated_orchestrator_dp import FederatedOrchestratorDP
                from federated_config import federated_config
                
                logger.info(f"Initializing federated with ε={dp_epsilon}, k={dp_k_threshold}")
                
                # Initialize orchestrator with DP parameters
                orchestrator = FederatedOrchestratorDP(
                    federated_config=federated_config,
                    dp_epsilon=float(dp_epsilon),  # Ensure float
                    dp_k_threshold=int(dp_k_threshold),  # Ensure int
                    enable_phoenix=enable_phoenix,
                    verbose=True
                )
                
                # Store DP parameters in orchestrator for later retrieval
                orchestrator._dp_epsilon_used = float(dp_epsilon)
                orchestrator._dp_k_threshold_used = int(dp_k_threshold)
                
                # Register hospitals
                if hospital_ids:
                    for h_id in hospital_ids:
                        orchestrator.register_hospital(h_id)
                else:
                    orchestrator.register_all_hospitals()
                
                if not orchestrator.hospitals:
                    raise RuntimeError("No hospitals registered. Check federated_config.")
                
                st.session_state.federated_hospitals_registered = True
                logger.info(f"✅ Federated system initialized with {len(orchestrator.hospitals)} hospitals")
                return orchestrator
                
            except ImportError as e:
                raise RuntimeError(
                    f"Federated components not available: {e}\n"
                    f"Ensure federated_orchestrator_dp.py, hospital_rag_dp.py, "
                    f"and federated_config.py are in the project root."
                )
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def execute_query(rag_system, query_text: str, show_sources: bool, show_full_context: bool, mode: str = "centralized"):
    """Execute query with appropriate system"""
    start_time = time.time()
    
    if mode.lower() == "centralized":
        result = rag_system.query(
            query_text,
            show_sources=show_sources,
            show_full_context=show_full_context
        )
    
    elif mode.lower() == "federated":
        result = rag_system.federated_query_with_dp(
            query=query_text,
            parallel=False,
            max_workers=1
        )
        
        # Get epsilon from system config
        epsilon = getattr(rag_system, 'dp_epsilon', 
                         getattr(rag_system, '_dp_epsilon_used', None))
        k_threshold = getattr(rag_system, 'dp_k_threshold',
                            getattr(rag_system, '_dp_k_threshold_used', None))
        
        # Convert federated result to standard format
        result = {
            'response': result.aggregated_response,
            'latency': result.total_latency_seconds,
            'num_retrieved': len(result.hospitals_with_results),
            'estimated_tokens': int(len(result.aggregated_response.split()) * 1.3),
            'source_nodes': [],  # Federated doesn't return source nodes same way
            'privacy_summary': {
                'epsilon': epsilon,
                'k_threshold': k_threshold,
                'mechanism': 'Response-level DP + Secure Aggregation',
            },
            'hospital_contributions': result.hospital_contributions,
        }
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return result


def set_sample_query(query_text):
    """Set a sample query"""
    st.session_state.current_query = query_text
    st.session_state.run_query_flag = True


def display_centralized_results(result, show_sources, show_full_context):
    """Display centralized RAG results"""
    st.divider()
    
    # Success message with metrics
    st.markdown(f"""
        <div class="success-box">
            ✅ <strong>Query completed </strong>
            &nbsp;|&nbsp; 📄 {result['num_retrieved']} documents
            &nbsp;|&nbsp; 🔤 ~{result['estimated_tokens']} tokens
            &nbsp;|&nbsp; 🔍 Filters: {"Yes" if result.get('used_metadata_filters') else "No"}
        </div>
    """, unsafe_allow_html=True)
    
    # Clinical findings
    st.markdown("### 📋 Clinical Findings")
    
    st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Medical Disclaimer:</strong> For research and educational purposes only.
        </div>
    """, unsafe_allow_html=True)
    
    response_text = result['response']
    if response_text:
        st.markdown(response_text)
    else:
        st.warning("No response generated. Please try a different query.")
    
    # Query details
    with st.expander("🔍 Query Processing Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Metadata Filters:** {'✅ Applied' if result.get('used_metadata_filters') else '❌ Not applied'}")
        with col2:
            st.write(f"**Estimated Tokens:** ~{result['estimated_tokens']}")
    
    # Source citations
    if show_sources and result.get('source_nodes'):
        st.divider()
        st.markdown(f"### 📚 Source Citations ({len(result['source_nodes'])} records)")
        
        for i, node in enumerate(result['source_nodes'], 1):
            meta = node.node.metadata
            patient = meta.get('patient_pseudonym', 'Unknown')
            enc_type = meta.get('encounter_type', 'Unknown')
            quarter = meta.get('temporal_quarter', 'Unknown')
            
            with st.expander(f"📄 {patient} | {enc_type} | {quarter}", expanded=(i == 1)):
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown(f"""
                        **Demographics**  
                        👤 {patient}  
                        🎂 {meta.get('age_range', 'N/A')}  
                        ⚧ {meta.get('gender', 'N/A')}
                    """)
                
                with c2:
                    st.markdown(f"""
                        **Background**  
                        🌍 {meta.get('race', 'N/A')}  
                        🏥 {meta.get('ethnicity', 'N/A')}  
                        📋 {enc_type}
                    """)
                
                with c3:
                    st.markdown(f"""
                        **Clinical**  
                        📅 {quarter}  
                        💊 {meta.get('num_conditions', 0)} conditions  
                        🔬 {meta.get('num_procedures', 0)} procedures
                    """)
                
                st.markdown("---")
                
                node_text = node.node.get_text()
                if show_full_context:
                    st.text(node_text)
                else:
                    preview = node_text[:800]
                    st.text(preview + "..." if len(node_text) > 800 else preview)
                    
                    if len(node_text) > 800:
                        if st.checkbox(f"Show full record", key=f"full_{i}"):
                            st.text(node_text)


def display_federated_results(result, query_text, system=None):
    """Display federated RAG with DP results"""
    st.divider()
    
    # Extract privacy info from multiple sources
    privacy_info = result.get('privacy_summary', {})
    epsilon = privacy_info.get('epsilon', 'N/A')
    
    # If epsilon still N/A, try to get from system
    if epsilon == 'N/A' and system is not None:
        try:
            epsilon = getattr(system, 'dp_epsilon', getattr(system, '_dp_epsilon_used', 'N/A'))
        except:
            epsilon = 'N/A'
    
    # Get hospital count
    hospital_count = len(result.get('hospital_contributions', []))
    if hospital_count == 0:
        hospital_count = result.get('num_retrieved', 0)
    
    # Success message with federated metrics
    st.markdown(f"""
        <div class="success-box">
            ✅ <strong>Federated query completed </strong>
            &nbsp;|&nbsp; 🏥 {hospital_count} hospital(s)
            &nbsp;|&nbsp; 🔒 DP-Protected
        </div>
    """, unsafe_allow_html=True)
    
    # Privacy guarantee with actual epsilon value
    if privacy_info or epsilon != 'N/A':
        epsilon_display = f"ε={epsilon}" if epsilon != 'N/A' else "ε=N/A"
        k_display = f"k={privacy_info.get('k_threshold', 'N/A')}" if privacy_info.get('k_threshold') else "k=N/A"
        
        st.markdown(f"""
            <div class="privacy-box">
                🔒 <strong>Privacy Guarantee:</strong> {epsilon_display}, {k_display}-anonymity
            </div>
        """, unsafe_allow_html=True)
    
    # Federated findings
    st.markdown("### 📋 Federated Medical Findings")
    
    st.markdown("""
        <div class="info-box">
            ℹ️ <strong>Note:</strong> Results are aggregated with differential privacy protection. 
            No individual patient data is exposed.
        </div>
    """, unsafe_allow_html=True)
    
    response_text = result['response']
    if response_text:
        st.markdown(response_text)
    else:
        st.warning("No results found across the federated network.")
    
    # Hospital contributions
    if result.get('hospital_contributions'):
        st.markdown("### 🏥 Hospital Contributions")
        
        # Calculate totals
        hospital_count_actual = len(result['hospital_contributions'])
        total_patients = sum(
            h.get('dp_patient_count', 0) 
            for h in result['hospital_contributions']
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Hospitals",
                hospital_count_actual,
                delta=None
            )
        
        with col2:
            st.metric(
                "Total Patients (DP-Protected)",
                total_patients,
                delta=None
            )
        
        with col3:
            epsilon_metric = epsilon if epsilon != 'N/A' else 'N/A'
            st.metric(
                "Privacy Budget (ε)",
                epsilon_metric,
                delta=None
            )
        
        # Detailed breakdown
        with st.expander("📊 Detailed Hospital Breakdown"):
            for hosp in result['hospital_contributions']:
                st.write(
                    f"**{hosp.get('hospital_id', 'Unknown')}**: "
                    f"{hosp.get('dp_patient_count', 0)} patients (DP-noised)"
                )
    
    # Query details
    with st.expander("🔍 Aggregation Details", expanded=False):
        st.write(f"**Query:** {query_text}")
        st.write(f"**Aggregation Method:** {privacy_info.get('mechanism', 'Secure aggregation')}")
        st.write(f"**Total Latency:** {result['latency']:.2f}s")
        st.write(f"**Estimated Tokens:** ~{result['estimated_tokens']}")
        
        # Privacy details
        if epsilon != 'N/A':
            st.write(f"**Privacy Budget (ε):** {epsilon}")
        if privacy_info.get('k_threshold'):
            st.write(f"**K-Anonymity Threshold:** {privacy_info.get('k_threshold')}")


def main():
    """Main Streamlit application"""
    
    # Title and header
    st.markdown('<div class="main-header"><span style="-webkit-text-fill-color: initial; text-shadow: none;">🏥</span> FRAG-MED</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Federated Retrieval-Augmented Generation for Medical Diagnosis</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.divider()
        
        # System mode selection
        st.markdown("### 🔄 System Mode")
        
        mode = st.radio(
            "Select mode:",
            options=["Centralized RAG", "Federated RAG (DP)"],
            key="system_mode_radio"
        )
        
        mode_key = "centralized" if "Centralized" in mode else "federated"
        
        if st.session_state.system_mode != mode_key:
            # Clean up old system
            if st.session_state.rag_system and hasattr(st.session_state.rag_system, 'shutdown'):
                try:
                    st.session_state.rag_system.shutdown()
                except:
                    pass
            if st.session_state.federated_system and hasattr(st.session_state.federated_system, 'shutdown'):
                try:
                    st.session_state.federated_system.shutdown()
                except:
                    pass
            
            st.session_state.system_mode = mode_key
            st.session_state.system_initialized = False
            st.rerun()
        
        st.divider()
        
        # Phoenix configuration
        st.markdown("### 📊 Phoenix Observability")
        st.session_state.phoenix_enabled = st.checkbox(
            "Enable Phoenix tracing",
            value=st.session_state.phoenix_enabled,
            help="Monitor queries in real-time at http://127.0.0.1:6006"
        )
        
        if st.session_state.phoenix_enabled:
            st.session_state.phoenix_launched = st.checkbox(
                "Launch Phoenix server",
                value=st.session_state.phoenix_launched,
                help="Auto-start Phoenix if not running"
            )
        else:
            st.session_state.phoenix_launched = False
        
        st.divider()
        
        # Mode-specific configuration
        if mode_key == "centralized":
            st.markdown("### 🎯 Centralized RAG Settings")
            
            show_sources = st.checkbox(
                "Show source citations",
                value=True,
                help="Display referenced patient records"
            )
            
            show_full_context = st.checkbox(
                "Show full context",
                value=False,
                help="Display complete encounter documents"
            )
        
        else:  # federated
            st.markdown("### 🔐 Federated DP Settings")
            
            # Store previous values to detect changes
            prev_epsilon = st.session_state.federated_epsilon
            prev_k = st.session_state.federated_k_threshold
            
            st.session_state.federated_epsilon = st.slider(
                "Privacy Budget (ε)",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.federated_epsilon,
                step=0.1,
                help="Lower ε = more private, more noise. Range 0.1-2.0 typical."
            )
            
            st.session_state.federated_k_threshold = st.slider(
                "K-Anonymity Threshold",
                min_value=1,
                max_value=10,
                value=st.session_state.federated_k_threshold,
                step=1,
                help="Suppress counts below this threshold"
            )
            
            # If DP parameters changed, reset system to reinitialize with new values
            if (prev_epsilon != st.session_state.federated_epsilon or 
                prev_k != st.session_state.federated_k_threshold):
                logger.info(f"DP parameters changed: ε {prev_epsilon}→{st.session_state.federated_epsilon}, k {prev_k}→{st.session_state.federated_k_threshold}")
                st.session_state.system_initialized = False
                st.info("ℹ️ Privacy settings updated. System will re-initialize with new values.")
                st.rerun()
            
            show_sources = False  # Federated doesn't support source citations
            show_full_context = False
        
        st.divider()
        
        # System status
        st.markdown("### 📊 System Status")
        
        if st.session_state.system_initialized:
            status_text = "✅ Active"
            status_class = "status-online"
            if mode_key == "federated":
                status_class = "status-federated"
        else:
            status_text = "⏳ Not initialized"
            status_class = "status-offline"
        
        st.markdown(
            f'<div class="status-card-mini {status_class}">{status_text}</div>',
            unsafe_allow_html=True
        )
        
        if st.session_state.phoenix_enabled:
            if check_phoenix_server():
                st.markdown(
                    '<div class="phoenix-box phoenix-active">🟢 Phoenix Running</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="phoenix-box" style="margin-top: 0.3rem">'
                    f'📊 <a href="http://127.0.0.1:6006" target="_blank">Dashboard</a>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="phoenix-box">🔴 Phoenix Not Running</div>',
                    unsafe_allow_html=True
                )
        
        # Query history
        st.divider()
        st.markdown("### 📜 Query History")
        if st.session_state.query_history:
            st.markdown(f"**{len(st.session_state.query_history)} queries executed**")
            with st.expander("View history", expanded=False):
                for i, q in enumerate(st.session_state.query_history[-5:], 1):
                    st.caption(f"{i}. {q['query'][:60]}...")
        else:
            st.caption("No queries yet")
    
    # Main content area
    st.markdown("### 🔍 Medical Query Interface")
    
    # Initialize system if needed
    if not st.session_state.system_initialized:
        init_message = "🔄 Initializing system..."
        if st.session_state.phoenix_enabled:
            init_message = "🔄 Initializing with Phoenix..."
        
        with st.spinner(init_message):
            try:
                if mode_key == "centralized":
                    st.session_state.rag_system = initialize_rag_system(
                        mode="centralized",
                        enable_phoenix=st.session_state.phoenix_enabled,
                        launch_phoenix=st.session_state.phoenix_launched
                    )
                else:
                    st.session_state.federated_system = initialize_rag_system(
                        mode="federated",
                        enable_phoenix=st.session_state.phoenix_enabled,
                        launch_phoenix=st.session_state.phoenix_launched,
                        dp_epsilon=st.session_state.federated_epsilon,
                        dp_k_threshold=st.session_state.federated_k_threshold
                    )
                
                st.session_state.system_initialized = True
                st.session_state.initialization_error = None
                
                # Success message
                if st.session_state.phoenix_enabled and check_phoenix_server():
                    st.success(f"✅ System initialized! Phoenix: http://127.0.0.1:6006")
                else:
                    st.success(f"✅ System initialized!")
                
            except Exception as e:
                st.session_state.initialization_error = str(e)
                st.error(f"❌ Failed to initialize: {e}")
                import traceback
                with st.expander("🐛 Error details"):
                    st.code(traceback.format_exc())
                st.stop()
    
    # Sample queries
    st.markdown("**💡 Quick Start** - Click to run:")
    
    if mode_key == "centralized":
        sample_queries = [
            ("🔬", "What conditions are documented for PATIENT_51344eea?"),
            ("💊", "What procedures and medications are documented for PATIENT_560864a2?"),
            ("🫁", "Find patients with acute bronchitis and describe their treatments."),
            ("🩺", "What are common symptoms and treatments for Type 2 diabetes in elderly?"),
            ("📟", "What medical devices are commonly used for chronic condition monitoring?"),
            ("🦴", "Describe typical procedures for bone fracture patients over 60."),
        ]
    else:
        sample_queries = [
            ("🫁", "Find patients with acute bronchitis and describe treatments"),
            ("🩺", "What conditions are common in diabetic patients?"),
            ("💊", "What medications are most frequently prescribed?"),
            ("📟", "What medical devices are used for condition monitoring?"),
            ("🏥", "Find patients with hypertension and respiratory conditions"),
        ]
    
    cols = st.columns(3)
    for idx, (icon, sq) in enumerate(sample_queries):
        with cols[idx % 3]:
            if st.button(f"{icon} {sq}", key=f"sample_{idx}", use_container_width=True):
                set_sample_query(sq)
                st.rerun()
    
    st.divider()
    
    # Query input
    st.markdown("**💬 Custom Query**")
    query_text = st.text_area(
        "Enter your medical query:",
        value=st.session_state.current_query,
        placeholder="Example: What conditions are documented for PATIENT_51344eea?",
        height=80,
        key="query_input",
        label_visibility="collapsed"
    )
    
    if query_text != st.session_state.current_query:
        st.session_state.current_query = query_text
    
    # Query buttons
    col_btn1, col_btn2, col_btn3 = st.columns([3, 2, 7])
    
    with col_btn1:
        run_query_clicked = st.button(
            "🚀 Run Query",
            type="primary",
            disabled=not query_text.strip(),
            use_container_width=True
        )
    
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.current_query = ""
            st.session_state.run_query_flag = False
            st.session_state.last_result = None
            st.rerun()
    
    # Execute query
    should_run_query = run_query_clicked or st.session_state.run_query_flag
    if st.session_state.run_query_flag:
        st.session_state.run_query_flag = False
    
    if should_run_query and query_text.strip():
        import io
        import contextlib
        
        stdout_buffer = io.StringIO()
        
        processing_msg = (
            "🔬 Processing query across hospitals..." 
            if mode_key == "federated" 
            else "🔬 Processing query..."
        )
        
        with st.spinner(f"{processing_msg} (10-30 seconds)"):
            try:
                with contextlib.redirect_stdout(stdout_buffer):
                    system = (
                        st.session_state.rag_system 
                        if mode_key == "centralized" 
                        else st.session_state.federated_system
                    )
                    
                    result = execute_query(
                        system,
                        query_text,
                        show_sources=show_sources if mode_key == "centralized" else False,
                        show_full_context=show_full_context if mode_key == "centralized" else False,
                        mode=mode_key
                    )
                
                console_output = stdout_buffer.getvalue()
                
                st.session_state.last_result = {
                    'result': result,
                    'console_output': console_output,
                    'query': query_text,
                    'mode': mode_key
                }
                
                st.session_state.query_history.append({
                    'query': query_text,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'latency': result['latency'],
                    'mode': mode_key
                })
                
            except Exception as e:
                st.error(f"❌ Query failed: {e}")
                import traceback
                with st.expander("🐛 Error details"):
                    st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.last_result:
        result = st.session_state.last_result['result']
        last_mode = st.session_state.last_result.get('mode', 'centralized')
        
        if last_mode == "centralized":
            display_centralized_results(result, show_sources, show_full_context)
        else:
            # Pass system object for epsilon extraction
            display_federated_results(
                result, 
                query_text,
                system=st.session_state.federated_system
            )


if __name__ == "__main__":
    main()
