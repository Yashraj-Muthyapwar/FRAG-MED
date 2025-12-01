#!/usr/bin/env python3
"""
FRAG-MED: Run Federated Queries with DP + Phoenix Observability

Usage:
    # Start Phoenix first (in another terminal):
    python monitor_system.py
    
    # Then run federated queries:
    python run_federated_dp.py
    
    # With custom parameters:
    python run_federated_dp.py --epsilon 0.5 --k-threshold 5
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

sys.path.insert(0, str(Path(__file__).parent))

from federated_config import federated_config
from federated_orchestrator_dp import FederatedOrchestratorDP


def show_comparison():
    """Show centralized vs federated comparison"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║               FRAG-MED: CENTRALIZED vs FEDERATED with DP                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PROBLEM WITH CENTRALIZED (Your previous output):                            ║
║  ────────────────────────────────────────────────                            ║
║  "PATIENT_560864a2, PATIENT_6aff4a21, PATIENT_e9e8a9eb have bronchitis..."  ║
║                                                                              ║
║  ❌ Patient IDs exposed                                                       ║
║  ❌ Individual records revealed                                               ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FEDERATED with DP (This system):                                            ║
║  ─────────────────────────────────                                           ║
║  "TOTAL PATIENTS: 5 (across 3 hospitals)                                     ║
║   CONDITIONS FOUND: Acute bronchitis (3 records), Viral sinusitis (2)...    ║
║   TREATMENTS: Acetaminophen (5 times), Respiratory therapy (3)..."          ║
║                                                                              ║
║  ✅ No patient IDs                                                            ║
║  ✅ Counts have Laplace noise                                                 ║
║  ✅ Data extracted directly from records (no LLM hallucination)              ║
║  ✅ DP-guaranteed privacy                                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def check_phoenix_running():
    """Check if Phoenix server is running"""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 6006))
        sock.close()
        return result == 0
    except:
        return False


def run_federated_dp(
    epsilon: float = 1.0,
    k_threshold: int = 3,
    hospital_ids: list = None,
    custom_query: str = None,
    enable_phoenix: bool = True
):
    """Run federated queries with DP"""
    
    print("\n" + "="*80)
    print("🔒 FRAG-MED: FEDERATED RAG WITH DP + PHOENIX TRACING")
    print("="*80)
    
    # Check Phoenix
    if enable_phoenix:
        if check_phoenix_running():
            print("✅ Phoenix server detected at http://127.0.0.1:6006")
        else:
            print("⚠️  Phoenix not running. Start it with: python monitor_system.py")
            print("   Continuing without Phoenix tracing...")
            enable_phoenix = False
    
    show_comparison()
    
    # Initialize orchestrator
    print(f"\n📡 Initializing Federated Orchestrator with DP...")
    print(f"   Privacy budget (ε): {epsilon}")
    print(f"   K-anonymity threshold: {k_threshold}")
    print(f"   Phoenix tracing: {'Enabled' if enable_phoenix else 'Disabled'}")
    
    orchestrator = FederatedOrchestratorDP(
        federated_config=federated_config,
        dp_epsilon=epsilon,
        dp_k_threshold=k_threshold,
        enable_phoenix=enable_phoenix,
        verbose=True
    )
    
    # Register hospitals
    if hospital_ids:
        for h_id in hospital_ids:
            orchestrator.register_hospital(h_id)
    else:
        orchestrator.register_all_hospitals()
    
    if not orchestrator.hospitals:
        print("\n❌ No hospitals registered!")
        print("   Run hospital_preprocessing.py for each hospital first.")
        return
    
    # Test queries
    if custom_query:
        test_queries = [{"query": custom_query, "desc": "Custom Query"}]
    else:
        test_queries = [
            {
                "query": "Find patients with acute bronchitis and describe their diagnostic procedures and medications.",
                "desc": "Respiratory Condition"
            },
            {
                "query": "What are common devices used by Type 2 diabetes patients?",
                "desc": "Diabetes Management"
            }
        ]
    
    # Run queries
    results = []
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_queries)}: {test['desc']}")
        print(f"{'='*80}")
        
        try:
            result = orchestrator.federated_query_with_dp(
                query=test['query'],
                parallel=False,  # Sequential for better debugging
                max_workers=1
            )
            results.append(result)
            
            # Verify no patient IDs
            if 'PATIENT_' in result.aggregated_response:
                print("\n⚠️ WARNING: Patient IDs detected in response!")
            else:
                print("\n✅ Privacy verified: No patient IDs in response")
            
        except Exception as e:
            print(f"\n❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(test_queries) and not custom_query:
            input("\n⏸️ Press Enter for next query...")
    
    # Print reports
    print(orchestrator.get_privacy_report())
    
    if enable_phoenix:
        print("\n📊 View detailed traces in Phoenix:")
        print("   http://127.0.0.1:6006")
    
    # Cleanup
    orchestrator.shutdown()
    
    print("\n✅ Federated DP test complete!")


def test_single_hospital(hospital_id: str, query: str, epsilon: float = 1.0):
    """Test single hospital for debugging"""
    print(f"\n🏥 Testing single hospital: {hospital_id}")
    
    from hospital_rag_dp import HospitalRAGWithDP
    
    hospital_config = federated_config.get_hospital_config(hospital_id)
    
    try:
        hospital = HospitalRAGWithDP(
            hospital_config=hospital_config,
            embedding_model_path=federated_config.EMBEDDING_MODEL_PATH,
            llm_model_name=federated_config.LLM_MODEL_NAME,
            dp_epsilon=epsilon,
            dp_k_threshold=3,
            enable_phoenix=check_phoenix_running(),
            verbose=True
        )
        
        result = hospital.query_with_dp(query)
        
        print(f"\n" + "="*60)
        print(f"📊 FINAL RESULT")
        print("="*60)
        print(f"Raw patient count: {result.raw_patient_count}")
        print(f"DP patient count: {result.dp_patient_count} (noised)")
        print(f"DP Conditions: {result.dp_conditions}")
        print(f"DP Treatments: {result.dp_treatments}")
        print(f"DP Age distribution: {result.dp_age_distribution}")
        
        print(f"\n📝 DP-Sanitized Response:")
        print("-"*60)
        print(result.dp_response)
        print("-"*60)
        
        print(f"\n🔍 Debug Info:")
        print(f"   Child nodes retrieved: {result.retrieval_debug.get('child_nodes_retrieved', 0)}")
        print(f"   Parent docs loaded: {result.retrieval_debug.get('parent_docs_loaded', 0)}")
        print(f"   Patients found: {result.retrieval_debug.get('patient_pseudonyms', [])}")
        
        if 'patient_' in result.dp_response.lower():
            print("\n⚠️ Patient IDs detected - check DP sanitization")
        else:
            print("\n✅ Privacy check passed")
        
        hospital.shutdown()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FRAG-MED Federated Queries with DP + Phoenix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start Phoenix first (in another terminal):
    python monitor_system.py
    
    # Run with defaults:
    python run_federated_dp.py
    
    # Stronger privacy:
    python run_federated_dp.py --epsilon 0.5 --k-threshold 5
    
    # Test single hospital:
    python run_federated_dp.py --single hospital_A
    
    # Custom query:
    python run_federated_dp.py --query "Find diabetic patients"
        """
    )
    
    parser.add_argument("--epsilon", "-e", type=float, default=1.0,
                        help="Privacy budget ε (default: 1.0)")
    parser.add_argument("--k-threshold", "-k", type=int, default=3,
                        help="K-anonymity threshold (default: 3)")
    parser.add_argument("--hospitals", nargs="+",
                        help="Specific hospital IDs")
    parser.add_argument("--query", type=str,
                        help="Custom query")
    parser.add_argument("--single", type=str,
                        help="Test single hospital")
    parser.add_argument("--no-phoenix", action="store_true",
                        help="Disable Phoenix tracing")
    
    args = parser.parse_args()
    
    if args.single:
        query = args.query or "Find patients with bronchitis"
        test_single_hospital(args.single, query, args.epsilon)
    else:
        run_federated_dp(
            epsilon=args.epsilon,
            k_threshold=args.k_threshold,
            hospital_ids=args.hospitals,
            custom_query=args.query,
            enable_phoenix=not args.no_phoenix
        )
