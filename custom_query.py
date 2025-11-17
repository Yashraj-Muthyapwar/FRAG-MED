#!/usr/bin/env python3
"""
Run custom queries on FRAG-MED system
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag.query_engine import CentralRAGSystem

def main():
    # Initialize RAG system
    rag = CentralRAGSystem(enable_phoenix=True, launch_server=False, verbose=True)
    
    print("\n" + "="*80)
    print("🏥 FRAG-MED CUSTOM QUERY INTERFACE")
    print("="*80)
    print("\nType 'exit' to quit")
    print("Type 'stats' to see performance statistics")
    print("="*80 + "\n")
    
    try:
        while True:
            # Get user input
            query = input("\n💬 Enter your medical query: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'exit':
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'stats':
                stats = rag.get_performance_stats()
                print("\n📊 Performance Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            # Run query
            result = rag.query(
                query,
                show_sources=True,
                show_full_context=False
            )
            
    finally:
        rag.shutdown()


if __name__ == "__main__":
    main()
