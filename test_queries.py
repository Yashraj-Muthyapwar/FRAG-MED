#!/usr/bin/env python3
"""
Sample queries to test FRAG-MED system with Phoenix observability
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag.query_engine import CentralRAGSystem
import time

def run_sample_queries():
    """Run sample medical queries and display results"""
    
    print("\n" + "="*80)
    print("🏥 FRAG-MED SAMPLE QUERIES")
    print("="*80)
    print("\nInitializing RAG system...")
    
    # Initialize system with Phoenix enabled
    rag = CentralRAGSystem(enable_phoenix=True,launch_server=False, verbose=True)
    
    # Define sample queries
    sample_queries = [
        {
            "query": "What are the common symptoms and treatments for Type 2 diabetes in elderly patients?",
            "category": "Diabetes Management"
        },
        {
            "query": "Find patients with acute bronchitis and describe their diagnostic procedures and medications.",
            "category": "Respiratory Conditions"
        },
        {
            "query": "What medical devices are commonly used for monitoring patients with chronic conditions?",
            "category": "Medical Devices"
        },
        {
            "query": "Describe typical procedures and recovery for bone fracture patients over 60 years old.",
            "category": "Orthopedic Care"
        },
        {
            "query": "What are the standard protocols for hypertension management in middle-aged patients?",
            "category": "Cardiovascular Health"
        }
    ]
    
    results = []
    
    try:
        for i, item in enumerate(sample_queries, 1):
            print(f"\n{'='*80}")
            print(f"QUERY {i}/{len(sample_queries)}: {item['category']}")
            print(f"{'='*80}")
            
            # Run query
            result = rag.query(
                item["query"],
                show_sources=True,
                show_full_context=False
            )
            
            results.append({
                'query': item['query'],
                'category': item['category'],
                'latency': result['latency'],
                'num_retrieved': result['num_retrieved'],
                'tokens': result['estimated_tokens']
            })
            
            # Pause between queries
            if i < len(sample_queries):
                print("\n⏸️  Press Enter to continue to next query...")
                input()
        
        # Display summary
        print("\n" + "="*80)
        print("📊 QUERY SUMMARY")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['category']}")
            print(f"   Latency: {result['latency']:.2f}s")
            print(f"   Retrieved: {result['num_retrieved']} encounters")
            print(f"   Tokens: ~{result['tokens']}")
        
        # Performance stats
        print("\n" + "="*80)
        print("📈 OVERALL PERFORMANCE")
        print("="*80)
        stats = rag.get_performance_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\n" + "="*80)
        print("✅ ALL QUERIES COMPLETE!")
        print("="*80)
        print("\n🔍 Check Phoenix Dashboard for detailed traces:")
        print(f"   {rag.phoenix.get_dashboard_url() if rag.phoenix else 'http://127.0.0.1:6006'}")
        print("\n💡 You can now run custom queries in the RAG system!")
        
    finally:
        rag.shutdown()


if __name__ == "__main__":
    run_sample_queries()
