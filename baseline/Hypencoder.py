import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import logging
import os
import random
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sklearn.model_selection import train_test_split

# Install hypencoder if not available
try:
    from hypencoder_cb.modeling.hypencoder import Hypencoder, HypencoderDualEncoder, TextEncoder
except ImportError:
    print("Installing hypencoder...")
    os.system("pip install git+https://github.com/jfkback/hypencoder-paper.git")
    from hypencoder_cb.modeling.hypencoder import Hypencoder, HypencoderDualEncoder, TextEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader:
    """Load and split TREC-COVID dataset"""
    
    def __init__(self, dataset_name: str = 'fiqa'):
        self.dataset_name = dataset_name

    def load_dataset(self, max_queries: int = 400000):
        logger.info(f"Loading {self.dataset_name} dataset")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        data_path = util.download_and_unzip(url, 'datasets')
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split='test')
        
        # Create train/val/test splits
        qids = list(queries.keys())[:max_queries]
        train_ids, temp_ids = train_test_split(qids, test_size=0.3, random_state=42)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
        
        splits = {}
        for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
            sq = {i: queries[i] for i in ids if i in qrels}
            sr = {i: qrels[i] for i in ids if i in qrels}
            splits[name] = (corpus, sq, sr)
            logger.info(f"{name} split: {len(sq)} queries, {len(corpus)} docs")
            
        return splits


class HypencoderIR(nn.Module):
    """Hypencoder-based Information Retrieval model"""
    
    def __init__(self, model_name: str = "jfkback/hypencoder.6_layer"):
        super().__init__()
        
        # Load pre-trained Hypencoder model
        logger.info(f"Loading Hypencoder model: {model_name}")
        self.dual_encoder = HypencoderDualEncoder.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Extract components
        self.query_encoder: Hypencoder = self.dual_encoder.query_encoder
        self.passage_encoder: TextEncoder = self.dual_encoder.passage_encoder
        
        # Set to evaluation mode
        self.eval()
        
    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Encode queries into q-nets (neural networks)"""
        query_inputs = self.tokenizer(
            queries, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=64
        ).to(device)
        
        with torch.no_grad():
            q_nets = self.query_encoder(
                input_ids=query_inputs["input_ids"],
                attention_mask=query_inputs["attention_mask"]
            ).representation
            
        return q_nets
    
    def encode_passages(self, passages: List[str]) -> torch.Tensor:
        """Encode passages into embeddings"""
        passage_inputs = self.tokenizer(
            passages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            passage_embeddings = self.passage_encoder(
                input_ids=passage_inputs["input_ids"],
                attention_mask=passage_inputs["attention_mask"]
            ).representation
            
        return passage_embeddings
    
    def compute_scores(self, q_nets, passage_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute relevance scores using q-nets and passage embeddings"""
        # q_nets is a callable neural network or list of networks from Hypencoder
        # passage_embeddings shape: (num_passages, embedding_dim)
        
        num_passages = passage_embeddings.shape[0]
        
        # Check if q_nets is a single network or a batch
        if callable(q_nets):
            # Single query case - q_nets is a single callable network
            # Add batch dimension: (1, num_passages, embedding_dim)
            passages_expanded = passage_embeddings.unsqueeze(0)
            
            # Compute scores using the q-net
            scores = q_nets(passages_expanded)
            
            # Handle different output shapes
            if isinstance(scores, torch.Tensor):
                if scores.dim() > 2:
                    scores = scores.squeeze(-1)  # Remove last dimension if it exists
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)  # Add batch dimension back
            else:
                # Convert to tensor if needed
                scores = torch.tensor(scores, device=passage_embeddings.device)
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)
                    
            return scores
            
        else:
            # Multiple queries case - q_nets might be a list or batch
            try:
                # Try to iterate over q_nets
                all_scores = []
                
                for query_net in q_nets:
                    # Add batch dimension: (1, num_passages, embedding_dim)
                    passages_expanded = passage_embeddings.unsqueeze(0)
                    
                    # Compute scores for this query
                    scores = query_net(passages_expanded)
                    
                    # Handle different output shapes
                    if isinstance(scores, torch.Tensor):
                        if scores.dim() > 2:
                            scores = scores.squeeze(-1)
                        if scores.dim() == 1:
                            scores = scores.unsqueeze(0)
                    else:
                        scores = torch.tensor(scores, device=passage_embeddings.device)
                        if scores.dim() == 1:
                            scores = scores.unsqueeze(0)
                            
                    all_scores.append(scores)
                
                # Concatenate all query scores
                final_scores = torch.cat(all_scores, dim=0)
                return final_scores
                
            except (TypeError, AttributeError):
                # If q_nets is not iterable, treat as single network
                passages_expanded = passage_embeddings.unsqueeze(0)
                scores = q_nets(passages_expanded)
                
                if isinstance(scores, torch.Tensor):
                    if scores.dim() > 2:
                        scores = scores.squeeze(-1)
                    if scores.dim() == 1:
                        scores = scores.unsqueeze(0)
                else:
                    scores = torch.tensor(scores, device=passage_embeddings.device)
                    if scores.dim() == 1:
                        scores = scores.unsqueeze(0)
                        
                return scores


class HypencoderEvaluator:
    """Evaluator for Hypencoder IR model"""
    
    def __init__(self, model: HypencoderIR, corpus: Dict, queries: Dict, qrels: Dict):
        self.model = model.to(device)
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        
        # Pre-encode all documents
        logger.info("Pre-encoding all documents...")
        self._precompute_document_embeddings()
        
    def _precompute_document_embeddings(self):
        """Pre-encode all documents in the corpus"""
        doc_ids = list(self.corpus.keys())
        doc_texts = [f"passage: {self.corpus[doc_id]['text']}" for doc_id in doc_ids]
        
        batch_size = 32  # Reduced batch size to avoid memory issues
        all_embeddings = []
        
        for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding documents"):
            batch_texts = doc_texts[i:i+batch_size]
            embeddings = self.model.encode_passages(batch_texts)
            all_embeddings.append(embeddings.cpu())
            
            # Clear GPU cache periodically
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
        
        self.doc_embeddings = torch.cat(all_embeddings, dim=0).to(device)
        self.doc_ids = doc_ids
        logger.info(f"Encoded {len(doc_ids)} documents")
        logger.info(f"Document embeddings shape: {self.doc_embeddings.shape}")
    
    def evaluate_metrics(self, k_values: List[int] = [5, 10, 20, 50, 100], max_queries: int = None) -> Dict:
        """Evaluate MRR, Recall, and NDCG metrics"""
        self.model.eval()
        
        metrics = {f'MRR@{k}': [] for k in k_values}
        metrics.update({f'Recall@{k}': [] for k in k_values})
        metrics.update({f'NDCG@{k}': [] for k in k_values})
        
        query_times = []
        evaluated_queries = 0
        
        with torch.no_grad():
            for qid, query_text in tqdm(self.queries.items(), desc="Evaluating"):
                if qid not in self.qrels:
                    continue
                if max_queries and evaluated_queries >= max_queries:
                    break
                
                start_time = time.time()
                
                # Get relevant documents
                relevant_docs = {doc_id for doc_id, rel in self.qrels[qid].items() if rel > 0}
                if not relevant_docs:
                    continue
                
                try:
                    # Encode query and compute scores
                    q_net = self.model.encode_queries([f"query: {query_text}"])
                    
                    # Log debug info about q_net type and structure
                    logger.debug(f"Query net type: {type(q_net)}")
                    if hasattr(q_net, 'shape'):
                        logger.debug(f"Query net shape: {q_net.shape}")
                    logger.debug(f"Document embeddings shape: {self.doc_embeddings.shape}")
                    
                    scores = self.model.compute_scores(q_net, self.doc_embeddings)
                    logger.debug(f"Scores shape: {scores.shape}")
                    
                    # Get top-k documents - handle single query case
                    if scores.dim() == 2:
                        # Multiple queries (shouldn't happen in this loop but handle it)
                        scores_flat = scores[0]  # Take first query
                    else:
                        # Single query
                        scores_flat = scores
                    
                    _, top_indices = torch.topk(scores_flat, min(max(k_values), len(self.doc_ids)))
                    rankings = [self.doc_ids[idx] for idx in top_indices.cpu().numpy()]
                    
                    query_time = time.time() - start_time
                    query_times.append(query_time)
                    
                    # Calculate metrics for each k
                    for k in k_values:
                        top_k_docs = rankings[:k]
                        
                        # MRR@k
                        mrr = 0.0
                        for rank, doc_id in enumerate(top_k_docs, 1):
                            if doc_id in relevant_docs:
                                mrr = 1.0 / rank
                                break
                        metrics[f'MRR@{k}'].append(mrr)
                        
                        # Recall@k
                        retrieved_relevant = len(set(top_k_docs) & relevant_docs)
                        total_relevant = len(relevant_docs)
                        recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0
                        metrics[f'Recall@{k}'].append(recall)
                        
                        # NDCG@k
                        ndcg = self._calculate_ndcg(top_k_docs, self.qrels[qid])
                        metrics[f'NDCG@{k}'].append(ndcg)
                    
                    evaluated_queries += 1
                    
                    # Log progress every 50 queries
                    if evaluated_queries % 50 == 0:
                        logger.info(f"Evaluated {evaluated_queries} queries...")
                        
                except Exception as e:
                    logger.error(f"Error processing query {qid}: {str(e)}")
                    continue
        
        # Calculate averages
        avg_metrics = {}
        for metric_name, values in metrics.items():
            avg_metrics[metric_name] = sum(values) / len(values) if values else 0.0
        
        # Add timing information
        avg_metrics['avg_query_time_ms'] = np.mean(query_times) * 1000 if query_times else 0.0
        avg_metrics['total_queries_evaluated'] = evaluated_queries
        
        return avg_metrics
    
    def _calculate_ndcg(self, rankings: List[str], qrel: Dict[str, int]) -> float:
        """Calculate NDCG for a single query"""
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG
        for i, doc_id in enumerate(rankings):
            if doc_id in qrel and qrel[doc_id] > 0:
                relevance = qrel[doc_id]
                dcg += (2 ** relevance - 1) / np.log2(i + 2)
        
        # Calculate IDCG
        ideal_relevances = sorted([rel for rel in qrel.values() if rel > 0], reverse=True)
        for i, relevance in enumerate(ideal_relevances[:len(rankings)]):
            idcg += (2 ** relevance - 1) / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0


def main():
    """Main execution function - evaluates all 3 Hypencoder models"""
    
    # Define all 3 Hypencoder models
    model_names = [
        'jfkback/hypencoder.2_layer',
        'jfkback/hypencoder.4_layer', 
        'jfkback/hypencoder.6_layer',
        'jfkback/hypencoder.8_layer'
    ]
    
    base_config = {
        'dataset_name': 'fiqa',
        'max_queries': 50000,
        'eval_k_values': [5, 10, 20, 50, 100],
        # 'max_eval_queries': 500,  # Limit for faster evaluation
        'results_dir': 'results'
    }
    
    # Create results directory
    os.makedirs(base_config['results_dir'], exist_ok=True)
    
    # Load dataset once (shared across all models)
    logger.info("Loading dataset...")
    data_loader = DataLoader(base_config['dataset_name'])
    splits = data_loader.load_dataset(base_config['max_queries'])
    
    # Use only test split as requested
    test_corpus, test_queries, test_qrels = splits['test']
    logger.info(f"Test split: {len(test_queries)} queries, {len(test_corpus)} documents")
    
    # Store all results
    all_results = {}
    
    # Evaluate each model
    for i, model_name in enumerate(model_names, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATING MODEL {i}/{len(model_names)}: {model_name}")
        logger.info(f"{'='*80}")
        
        # Create config for this model
        config = base_config.copy()
        config['model_name'] = model_name
        
        try:
            # Clear GPU cache before loading new model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Initialize Hypencoder model
            logger.info(f"Initializing {model_name}...")
            model = HypencoderIR(model_name)
            
            # Initialize evaluator
            logger.info("Initializing evaluator...")
            evaluator = HypencoderEvaluator(model, test_corpus, test_queries, test_qrels)
            
            # Evaluate model
            logger.info("Starting evaluation...")
            start_time = time.time()
            
            results = evaluator.evaluate_metrics(
                k_values=config['eval_k_values']
            )
            
            total_time = time.time() - start_time
            results['total_evaluation_time_seconds'] = total_time
            
            # Store results
            all_results[model_name] = results
            
            # Log results for this model
            logger.info("=" * 60)
            logger.info(f"RESULTS FOR {model_name}")
            logger.info("=" * 60)
            
            for metric, score in results.items():
                if isinstance(score, float):
                    logger.info(f"{metric}: {score:.4f}")
                else:
                    logger.info(f"{metric}: {score}")
            
            # Save individual model results to JSON
            model_safe_name = model_name.replace('/', '_').replace('.', '_')
            results_file = os.path.join(
                config['results_dir'], 
                f'hypencoder_evaluation_results_{model_safe_name}.json'
            )
            
            # Prepare results for JSON serialization
            json_results = {
                'model_name': model_name,
                'dataset': config['dataset_name'],
                'config': config,
                'results': results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Individual results saved to: {results_file}")
            
            # Print summary for this model
            print(f"\n{'='*60}")
            print(f"SUMMARY - {model_name}")
            print(f"{'='*60}")
            print(f"Dataset: {config['dataset_name']}")
            print(f"Queries evaluated: {results['total_queries_evaluated']}")
            print(f"Average query time: {results['avg_query_time_ms']:.2f} ms")
            print(f"Total evaluation time: {total_time:.2f} seconds")
            print("\nKey Metrics:")
            for k in [5, 10, 20]:
                print(f"  MRR@{k}: {results[f'MRR@{k}']:.4f}")
                print(f"  Recall@{k}: {results[f'Recall@{k}']:.4f}")
                print(f"  NDCG@{k}: {results[f'NDCG@{k}']:.4f}")
            
            # Clean up model to free memory
            del model
            del evaluator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            all_results[model_name] = {'error': str(e)}
            continue
    
    # Save combined results
    combined_results_file = os.path.join(base_config['results_dir'], 'hypencoder_fiqa_all_models_comparison.json')
    combined_json_results = {
        'dataset': base_config['dataset_name'],
        'base_config': base_config,
        'all_model_results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(combined_results_file, 'w') as f:
        json.dump(combined_json_results, f, indent=2)
    
    logger.info(f"Combined results saved to: {combined_results_file}")
    
    # Print final comparison summary
    print(f"\n{'='*80}")
    print("FINAL COMPARISON - ALL MODELS")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"{'Model':<35} {'MRR@10':<10} {'Recall@10':<12} {'NDCG@10':<10} {'Avg Time (ms)':<15}")
    print("-" * 80)
    
    for model_name in model_names:
        if model_name in all_results and 'error' not in all_results[model_name]:
            results = all_results[model_name]
            mrr10 = results.get('MRR@10', 0)
            recall10 = results.get('Recall@10', 0)
            ndcg10 = results.get('NDCG@10', 0)
            avg_time = results.get('avg_query_time_ms', 0)
            
            print(f"{model_name:<35} {mrr10:<10.4f} {recall10:<12.4f} {ndcg10:<10.4f} {avg_time:<15.2f}")
        else:
            print(f"{model_name:<35} {'ERROR':<10} {'ERROR':<12} {'ERROR':<10} {'ERROR':<15}")
    
    return all_results


if __name__ == '__main__':
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set debug logging for first few queries
    if logger.level <= logging.INFO:
        logger.setLevel(logging.DEBUG)
        # Reset to INFO after a few queries to avoid spam
        
    # Run evaluation for all models
    all_results = main()