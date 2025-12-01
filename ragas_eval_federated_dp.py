#!/usr/bin/env python3

import json
import logging
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import Counter
import csv
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Import federated orchestrator
from federated_config import federated_config
from federated_orchestrator_dp import FederatedOrchestratorDP

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

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    logger.error("âŒ Missing required package: langchain-openai")
    logger.error("Run: pip install langchain-openai")
    sys.exit(1)


class FederatedRAGASEvaluator:
    """
    Evaluator for FRAG-MED Federated System with DP Epsilon Testing.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, enable_phoenix: bool = False):
        """
        Initialize federated evaluator.
        
        Args:
            openai_api_key: OpenAI key for evaluation
            enable_phoenix: Enable Phoenix tracing
        """
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY must be set in environment or passed to init")
        
        self.orchestrators = {}  # epsilon -> orchestrator
        self.questions = []
        self.results_by_epsilon = {}  # epsilon -> evaluation results
        
        logger.info("="*80)
        logger.info("ðŸ FRAG-MED FEDERATED RAGAS EVALUATION WITH DP")
        logger.info("="*80)
        
        # Initialize RAGAS with GPT-4o-mini for evaluation
        self._init_ragas()
    
    def _init_ragas(self):
        """Initialize RAGAS with GPT-4o-mini for evaluation"""
        try:
            logger.info("\nðŸ Š Initializing RAGAS with GPT-4o-mini...")
            
            # Evaluation LLM: GPT-4o-mini
            self.evaluation_llm = ChatOpenAI(model="gpt-4o-mini")
            
            # Evaluation embeddings: text-embedding-3-small
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # Metrics to evaluate
            self.metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
                answer_similarity
            ]
            
            logger.info("âœ… RAGAS initialized with GPT-4o-mini")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAGAS: {e}")
            raise
    
    def init_federated_system(self, epsilon: float, k_threshold: int = 3) -> FederatedOrchestratorDP:
        """
        Initialize federated orchestrator for given epsilon.
        
        Args:
            epsilon: DP privacy budget
            k_threshold: K-anonymity threshold
            
        Returns:
            Initialized FederatedOrchestratorDP
        """
        if epsilon in self.orchestrators:
            return self.orchestrators[epsilon]
        
        logger.info(f"\nðŸ ¡ Initializing Federated Orchestrator with Îµ={epsilon}...")
        
        orchestrator = FederatedOrchestratorDP(
            federated_config=federated_config,
            dp_epsilon=epsilon,
            dp_k_threshold=k_threshold,
            enable_phoenix=False,  # Disable for evaluation
            verbose=False  # Less verbose for evaluation runs
        )
        
        # Register hospitals
        success = orchestrator.register_all_hospitals()
        
        if success < len(federated_config.HOSPITAL_IDS):
            logger.warning(f"Only {success}/{len(federated_config.HOSPITAL_IDS)} hospitals registered")
        
        self.orchestrators[epsilon] = orchestrator
        logger.info(f"âœ… Federated system ready with Îµ={epsilon}")
        
        return orchestrator
    
    def load_questions(self, questions_file: str):
        """Load questions and ground truths"""
        file_path = Path(questions_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.questions = json.load(f)
        
        logger.info(f"âœ… Loaded {len(self.questions)} questions")
    
    def generate_responses(
        self,
        orchestrator: FederatedOrchestratorDP,
        epsilon: float,
        sample_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate responses using federated RAG with DP.
        
        Args:
            orchestrator: FederatedOrchestratorDP instance
            epsilon: Privacy budget value
            sample_size: Number of questions to process
            
        Returns:
            List of results with questions, answers, contexts
        """
        if not self.questions:
            raise ValueError("No questions loaded. Call load_questions first.")
        
        questions_to_process = self.questions[:sample_size] if sample_size else self.questions
        
        results = []
        for i, item in enumerate(questions_to_process, 1):
            question = item['question']
            ground_truth = item['ground_truth']
            
            if i % 10 == 0 or i == 1:
                logger.info(f"\n   Processing question {i}/{len(questions_to_process)} (Îµ={epsilon})")
            
            try:
                # Query federated system with DP
                fed_result = orchestrator.federated_query_with_dp(
                    query=question,
                    parallel=False,
                    max_workers=1
                )
                
                # Extract contexts from hospital contributions
                # Since federated doesn't return source nodes like centralized,
                # we create synthetic contexts from the response
                contexts = self._extract_contexts_from_federated_result(fed_result)
                
                # Extract answer
                answer = fed_result.aggregated_response
                
                results.append({
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                    "epsilon": epsilon,
                    "hospitals_queried": len(fed_result.hospitals_queried),
                    "hospitals_with_results": len(fed_result.hospitals_with_results),
                    "total_dp_patients": fed_result.total_dp_patient_count,
                    "latency": fed_result.total_latency_seconds
                })
                
            except Exception as e:
                logger.error(f"Failed at question {i}: {e}")
                continue
        
        return results
    
    def _extract_contexts_from_federated_result(self, fed_result) -> List[str]:
        """
        Extract contexts from federated result.
        Since federated returns aggregated data, we construct context from response.
        """
        contexts = []
        
        # Add aggregated conditions
        if fed_result.aggregated_conditions:
            conditions_text = "CONDITIONS: " + ", ".join(fed_result.aggregated_conditions[:5])
            contexts.append(conditions_text)
        
        # Add aggregated treatments
        if fed_result.aggregated_treatments:
            treatments_text = "TREATMENTS: " + ", ".join(fed_result.aggregated_treatments[:5])
            contexts.append(treatments_text)
        
        # Add aggregated procedures
        if fed_result.aggregated_procedures:
            procedures_text = "PROCEDURES: " + ", ".join(fed_result.aggregated_procedures[:3])
            contexts.append(procedures_text)
        
        # Add age distribution
        if fed_result.combined_age_distribution:
            age_text = "AGE DISTRIBUTION: " + ", ".join(
                f"{k}: {v}" for k, v in fed_result.combined_age_distribution.items()
            )
            contexts.append(age_text)
        
        # Add hospital contributions
        if fed_result.hospital_contributions:
            hosp_text = "HOSPITALS: " + ", ".join(
                f"{h['hospital_id']} ({h['dp_patient_count']} patients)"
                for h in fed_result.hospital_contributions[:3]
            )
            contexts.append(hosp_text)
        
        return contexts if contexts else [fed_result.aggregated_response[:500]]
    
    def run_evaluation(
        self,
        epsilon_values: List[float],
        sample_size: Optional[int] = None,
        k_threshold: int = 3
    ) -> Dict:
        """
        Run full evaluation across multiple epsilon values.
        
        Args:
            epsilon_values: List of epsilon values to test
            sample_size: Number of questions per epsilon
            k_threshold: K-anonymity threshold
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("\n" + "="*80)
        logger.info(f"Starting federated evaluation with {len(epsilon_values)} epsilon values")
        logger.info("="*80)
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'epsilon_values': epsilon_values,
            'sample_size': sample_size or len(self.questions),
            'metrics_evaluated': [m.name for m in self.metrics],
            'results_by_epsilon': {},
            'comparison_summary': {}
        }
        
        epsilon_summaries = []
        
        for eps_idx, epsilon in enumerate(epsilon_values, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"EVALUATION {eps_idx}/{len(epsilon_values)}: Îµ={epsilon}")
            logger.info(f"{'='*80}")
            
            try:
                # Initialize federated system for this epsilon
                orchestrator = self.init_federated_system(epsilon, k_threshold)
                
                # Generate responses
                logger.info(f"\n Generating responses with µ={epsilon}...")
                cache_file = f"federated_results_eps{epsilon:.1f}.json"
                
                if Path(cache_file).exists():
                    logger.info(f"   ðŸ‚ Loading from cache: {cache_file}")
                    with open(cache_file, 'r') as f:
                        raw_results = json.load(f)
                    if sample_size and len(raw_results) > sample_size:
                        raw_results = raw_results[:sample_size]
                    logger.info(f"Loaded {len(raw_results)} cached results")
                else:
                    logger.info(f"   Generating fresh results...")
                    raw_results = self.generate_responses(orchestrator, epsilon, sample_size)
                    
                    # Save checkpoint
                    with open(cache_file, 'w') as f:
                        json.dump(raw_results, f, indent=2)
                    logger.info(f"Checkpoint saved: {cache_file}")
                
                if not raw_results:
                    logger.error(f"No results for Îµ={epsilon}")
                    continue
                
                # Convert to RAGAS dataset
                logger.info(f"\nConverting {len(raw_results)} results to RAGAS format...")
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
                
                # Run evaluation
                logger.info(f"Running RAGAS evaluation with GPT-4o-mini...")
                eval_result = evaluate(
                    dataset=evaluation_dataset,
                    metrics=self.metrics,
                    llm=self.evaluation_llm,
                    embeddings=self.embeddings,
                    raise_exceptions=False
                )
                
                # Process scores
                scores = {metric.name: [] for metric in self.metrics}
                invalid_counts = Counter()
                
                for row in eval_result.scores:
                    for metric_name, value in row.items():
                        if np.isnan(value):
                            invalid_counts[metric_name] += 1
                        else:
                            scores[metric_name].append(value)
                
                # Build summary
                eps_summary = {
                    'epsilon': epsilon,
                    'samples': len(raw_results),
                    'metrics': {}
                }
                
                logger.info(f"\nðŸ RESULTS FOR Îµ={epsilon}:")
                
                for metric, valid_scores in scores.items():
                    if valid_scores:
                        mean = np.mean(valid_scores)
                        std = np.std(valid_scores)
                        eps_summary['metrics'][metric] = {
                            'mean': float(mean),
                            'std': float(std),
                            'min': float(np.min(valid_scores)),
                            'max': float(np.max(valid_scores)),
                            'samples': len(valid_scores)
                        }
                        logger.info(f"   {metric}: {mean:.4f} Â± {std:.4f}")
                    else:
                        logger.warning(f"   {metric}: No valid scores")
                
                # Overall score
                all_means = [m['mean'] for m in eps_summary['metrics'].values() if 'mean' in m]
                if all_means:
                    overall = np.mean(all_means)
                    eps_summary['overall_score'] = float(overall)
                    logger.info(f"\n   OVERALL: {overall:.4f}")
                
                # Add metadata
                eps_summary['avg_latency'] = np.mean([r['latency'] for r in raw_results])
                eps_summary['avg_hospitals'] = np.mean([r['hospitals_with_results'] for r in raw_results])
                eps_summary['avg_dp_patients'] = np.mean([r['total_dp_patients'] for r in raw_results])
                
                all_results['results_by_epsilon'][epsilon] = eps_summary
                epsilon_summaries.append(eps_summary)
                
                # Cleanup
                orchestrator.shutdown()
                
            except Exception as e:
                logger.error(f"Failed for Îµ={epsilon}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compare epsilon values
        if epsilon_summaries:
            logger.info(f"\n{'='*80}")
            logger.info("PRIVACY-UTILITY TRADEOFF COMPARISON")
            logger.info(f"{'='*80}\n")
            
            all_results['comparison_summary'] = self._compare_epsilon_values(epsilon_summaries)
        
        return all_results
    
    def _compare_epsilon_values(self, summaries: List[Dict]) -> Dict:
        """Compare evaluation results across epsilon values"""
        comparison = {
            'by_metric': {},
            'privacy_utility_tradeoff': []
        }
        
        # Organize by metric
        for summary in summaries:
            eps = summary['epsilon']
            for metric, scores in summary['metrics'].items():
                if metric not in comparison['by_metric']:
                    comparison['by_metric'][metric] = []
                
                comparison['by_metric'][metric].append({
                    'epsilon': eps,
                    'score': scores['mean'],
                    'std': scores['std']
                })
        
        # Print comparison table
        print("\n METRIC COMPARISON TABLE:")
        print("-" * 80)
        print(f"{'Metric':<30} {'Îµ=0.1':<15} {'Îµ=0.5':<15} {'Îµ=1.0':<15} {'Îµ=2.0':<15}")
        print("-" * 80)
        
        for metric, data in comparison['by_metric'].items():
            row = f"{metric:<30}"
            for item in sorted(data, key=lambda x: x['epsilon']):
                row += f"{item['score']:.4f}Â±{item['std']:.3f}  ".ljust(15)
            print(row)
        
        # Privacy-utility analysis
        print("\n\n PRIVACY-UTILITY ANALYSIS:")
        print("-" * 80)
        
        for summary in sorted(summaries, key=lambda x: x['epsilon']):
            eps = summary['epsilon']
            overall = summary.get('overall_score', 0)
            avg_latency = summary.get('avg_latency', 0)
            
            # Privacy level assessment
            if eps < 0.5:
                privacy_level = "STRONG"
            elif eps < 1.0:
                privacy_level = "BALANCED"
            else:
                privacy_level = "RELAXED"
            
            print(f"\nµ = {eps:.1f} ({privacy_level} privacy)")
            print(f"   Overall Score: {overall:.4f}")
            print(f"   Avg Latency: {avg_latency:.2f}s")
            print(f"   Avg Hospitals: {summary.get('avg_hospitals', 0):.1f}")
            print(f"   Avg DP Patients Found: {summary.get('avg_dp_patients', 0):.1f}")
        
        return comparison
    
    def save_results(self, results: Dict, output_dir: str = "federated_ragas_results"):
        """Save comprehensive evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Main results JSON
        results_file = output_path / f"federated_eval_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nðŸ Full results saved: {results_file}")
        
        # CSV comparison
        csv_file = output_path / f"epsilon_comparison_{timestamp}.csv"
        if results['results_by_epsilon']:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                eps_values = sorted(results['results_by_epsilon'].keys())
                metrics = list(self.metrics)
                header = ['Metric'] + [f"Îµ={e}" for e in eps_values]
                writer.writerow(header)
                
                # Rows
                for metric in metrics:
                    row = [metric.name]
                    for eps in eps_values:
                        if eps in results['results_by_epsilon']:
                            score = results['results_by_epsilon'][eps]['metrics'].get(metric.name, {}).get('mean', 'N/A')
                            row.append(f"{score:.4f}" if isinstance(score, float) else score)
                        else:
                            row.append('N/A')
                    writer.writerow(row)
            
            logger.info(f"ðŸ CSV comparison saved: {csv_file}")
        
        return str(results_file)


def main():
    """Main evaluation workflow"""
    
    parser = argparse.ArgumentParser(
        description="FRAG-MED Federated RAGAS Evaluation with DP Epsilon Testing"
    )
    parser.add_argument("--questions-file", default="frag_med_100_qa.json",
                        help="Path to questions file")
    parser.add_argument("--epsilons", nargs="+", type=float, 
                        default=[0.1, 0.5, 1.0, 2.0],
                        help="Epsilon values to test")
    parser.add_argument("--sample-size", type=int,
                        help="Number of questions to evaluate")
    parser.add_argument("--k-threshold", type=int, default=3,
                        help="K-anonymity threshold")
    parser.add_argument("--output-dir", default="federated_ragas_results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = FederatedRAGASEvaluator()
        
        # Load questions
        evaluator.load_questions(args.questions_file)
        
        # Run evaluation across epsilon values
        results = evaluator.run_evaluation(
            epsilon_values=args.epsilons,
            sample_size=args.sample_size,
            k_threshold=args.k_threshold
        )
        
        # Save results
        evaluator.save_results(results, args.output_dir)
        
        logger.info("\nâœ… ALL EVALUATIONS COMPLETE!")
        
        return 0
        
    except Exception as e:
        logger.error(f"\nâŒ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
