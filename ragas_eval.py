#!/usr/bin/env python3
"""
FRAG-MED RAGAS Evaluation - Integrated with Centralized Architecture
Uses Llama-Index for RAG generation and RAGAS for evaluation.

Key Features:
- Uses centralized RAG from query_engine.py for answer generation
- Local embeddings and LLM from config.py for retrieval/generation
- OpenAI GPT-4o-mini exclusively for RAGAS evaluation
- Processes 100 FRAG-MED questions from frag_med_100_qa.json
- CHECKPOINTING: Saves generated answers before evaluation to prevent data loss

Requirements:
- pip install ragas --upgrade
- pip install openai
- pip install langchain-openai
- Set OPENAI_API_KEY environment variable
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized RAG system
from src.rag.query_engine import CentralRAGSystem

# RAGAS imports
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)

# OpenAI Integration for Ragas (Requires langchain-openai)
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    logger.error("❌ Missing required package: langchain-openai")
    logger.error("Run: pip install langchain-openai")
    sys.exit(1)


class FragMedRAGASEvaluator:
    """
    Evaluator for FRAG-MED using centralized RAG and RAGAS.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            openai_api_key: OpenAI key for evaluation (required)
        """
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY must be set in environment or passed to init")
        
        self.rag_system = None
        self.questions = []
        
        logger.info("="*80)
        logger.info("🏥 FRAG-MED RAGAS EVALUATION - CENTRALIZED MODE")
        logger.info("="*80)
        
        # Initialize RAG system using config.py models
        self._init_rag_system()
        
        # Initialize RAGAS with GPT-4o-mini for evaluation
        self._init_ragas()
    
    def _init_rag_system(self):
        """Initialize centralized RAG system from query_engine.py"""
        try:
            logger.info("\n📡 Initializing CentralRAGSystem...")
            
            self.rag_system = CentralRAGSystem(
                enable_phoenix=False,  # Disable for evaluation
                verbose=False
            )
            logger.info("✅ CentralRAGSystem initialized with local models from config.py")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def _init_ragas(self):
        """Initialize RAGAS with GPT-4o-mini for evaluation"""
        try:
            logger.info("\n📊 Initializing RAGAS with GPT-4o-mini...")
            
            # Evaluation LLM: GPT-4o-mini (Via LangChain Wrapper)
            self.evaluation_llm = ChatOpenAI(model="gpt-4o-mini")
            
            # Evaluation embeddings: text-embedding-3-small (Via LangChain Wrapper)
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # Metrics to evaluate
            self.metrics = [
                faithfulness,          # How faithful is answer to context
                answer_relevancy,      # How relevant is answer to question
                context_precision,     # Precision of retrieved contexts
                context_recall,        # Recall of retrieved contexts
                answer_correctness,    # Correctness vs ground truth
                answer_similarity      # Semantic similarity to ground truth
            ]
            
            logger.info("✅ RAGAS initialized with GPT-4o-mini for evaluation")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAGAS: {e}")
            raise
    
    def load_questions(self, questions_file: str):
        """Load questions and ground truths from JSON"""
        file_path = Path(questions_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.questions = json.load(f)
        
        logger.info(f"✅ Loaded {len(self.questions)} questions from {questions_file}")
    
    def generate_responses(self, sample_size: Optional[int] = None) -> List[Dict]:
        """
        Generate responses using centralized RAG system.
        """
        if not self.questions:
            raise ValueError("No questions loaded. Call load_questions first.")
        
        questions_to_process = self.questions[:sample_size] if sample_size else self.questions
        
        results = []
        for i, item in enumerate(questions_to_process, 1):
            question = item['question']
            ground_truth = item['ground_truth']
            
            logger.info(f"\n📝 Processing question {i}/{len(questions_to_process)}")
            logger.info(f"Question: {question}")
            
            try:
                # Query centralized RAG
                result = self.rag_system.query(
                    question,
                    show_sources=False,
                    show_full_context=False
                )
                
                # Extract contexts
                contexts = [
                    node.node.get_text() 
                    for node in result.get('source_nodes', [])
                ]
                
                # Extract answer
                answer = result.get('response', "No response generated")
                
                results.append({
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth
                })
                
                logger.info(f"✅ Generated answer (length: {len(answer)} chars)")
                logger.info(f"   Retrieved {len(contexts)} contexts")
                
            except Exception as e:
                logger.error(f"Failed to process question {i}: {e}")
                continue
        
        return results
    
    def run_evaluation(self, sample_size: Optional[int] = None, cache_file: str = "intermediate_rag_results.json") -> Dict:
        """
        Run full evaluation pipeline with Checkpointing.
        """
        logger.info("\n🚀 Starting evaluation pipeline...")
        
        raw_results = []
        cache_path = Path(cache_file)

        # 1. CHECK FOR CACHE
        if cache_path.exists():
            logger.info(f"\n📂 Found cached results at {cache_file}!")
            logger.info("   Skipping generation and loading from file...")
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    raw_results = json.load(f)
                
                # Apply sample size if requested, even on cached data
                if sample_size and len(raw_results) > sample_size:
                    logger.info(f"   Slicing cached results to {sample_size} samples")
                    raw_results = raw_results[:sample_size]
                    
                logger.info(f"✅ Successfully loaded {len(raw_results)} samples from cache")
            except Exception as e:
                logger.warning(f"❌ Failed to load cache: {e}. Will generate fresh.")
        
        # 2. GENERATE IF NO CACHE
        if not raw_results:
            logger.info("\n⚡ No cache found. Generating responses...")
            raw_results = self.generate_responses(sample_size)
            
            if not raw_results:
                raise ValueError("No valid responses generated")
            
            # Save Checkpoint Immediately
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(raw_results, f, indent=2, ensure_ascii=False)
            logger.info(f"\n💾 Checkpoint saved: Generated results saved to {cache_file}")

        # 3. CONVERT TO RAGAS DATASET
        logger.info(f"\n🔄 Converting {len(raw_results)} results to RAGAS Dataset...")
        ragas_samples = []
        for item in raw_results:
            ragas_samples.append(
                SingleTurnSample(
                    user_input=item['question'],
                    response=item['answer'],
                    retrieved_contexts=item['contexts'],
                    reference=item['ground_truth']
                )
            )
        
        evaluation_dataset = EvaluationDataset(samples=ragas_samples)
        
        # 4. RUN EVALUATION
        logger.info(f"📊 Evaluating samples with RAGAS...")
        try:
            eval_result = evaluate(
                dataset=evaluation_dataset,
                metrics=self.metrics,
                llm=self.evaluation_llm,
                embeddings=self.embeddings,
                raise_exceptions=False
            )
            
            # Process results
            scores = {metric.name: [] for metric in self.metrics}
            invalid_counts = Counter()
            
            for row in eval_result.scores:
                for metric_name, value in row.items():
                    if np.isnan(value):
                        invalid_counts[metric_name] += 1
                    else:
                        scores[metric_name].append(value)
            
            summary = {
                'dataset_size': len(raw_results),
                'metrics': {}
            }
            
            logger.info("\n📈 EVALUATION RESULTS")
            for metric, valid_scores in scores.items():
                if valid_scores:
                    mean = np.mean(valid_scores)
                    std = np.std(valid_scores)
                    summary['metrics'][metric] = {
                        'mean': float(mean),
                        'std': float(std),
                        'min': float(np.min(valid_scores)),
                        'max': float(np.max(valid_scores)),
                        'samples': len(valid_scores)
                    }
                    
                    logger.info(f"\n✅ {metric.upper()}")
                    logger.info(f"   Mean: {mean:.4f} ± {std:.4f}")
                    logger.info(f"   Range: [{np.min(valid_scores):.4f}, {np.max(valid_scores):.4f}]")
                    logger.info(f"   Valid samples: {len(valid_scores)}/{len(raw_results)}")
                    if invalid_counts[metric]:
                        logger.warning(f"   Invalid (NaN): {invalid_counts[metric]}")
                else:
                    logger.warning(f"❌ No valid scores for {metric}")
            
            # Overall score
            all_means = [m['mean'] for m in summary['metrics'].values() if 'mean' in m]
            if all_means:
                overall = np.mean(all_means)
                summary['overall_score'] = float(overall)
                logger.info(f"\n🎯 OVERALL SCORE: {overall:.4f}")
            
            summary['results'] = raw_results
            
            logger.info("\n" + "="*80)
            logger.info("✅ Evaluation complete!")
            logger.info("="*80)
            
            return summary
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_results(self, results: Dict, output_dir: str = "ragas_results"):
        """Save final evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_file = output_path / f"frag_med_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Final Results saved to {results_file}")
        return str(results_file)


def main():
    """Main evaluation workflow"""
    
    # Configuration
    QUESTIONS_FILE = "frag_med_100_qa.json"
    SAMPLE_SIZE = None  # Set to integer for subset, None for all
    
    # Checkpoint File
    # If this file exists, generation is skipped!
    CACHE_FILE = "intermediate_rag_results.json"
    
    try:
        # Initialize evaluator
        evaluator = FragMedRAGASEvaluator()
        
        # Load questions
        evaluator.load_questions(QUESTIONS_FILE)
        
        # Run evaluation (Checks for cache automatically)
        results = evaluator.run_evaluation(
            sample_size=SAMPLE_SIZE,
            cache_file=CACHE_FILE
        )
        
        # Save results
        evaluator.save_results(results)
        
        logger.info("\n✅ ALL DONE!")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
