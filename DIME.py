import os
import json
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import pickle
import math
from statistics import mean
import argparse
import logging
from typing import Dict, Tuple
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dime_evaluation.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_embed_queries(model_name: str, queries: Dict[str, str], device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    Load the embedding model and embed queries.
    
    Args:
        model_name: Name/path of the sentence transformer model
        queries: Dictionary of query_id -> query_text
        device: Device to run the model on ('cuda' or 'cpu')
        
    Returns:
        Dictionary mapping query_id to query embedding tensor
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load the model
    model = SentenceTransformer(model_name)
    model.to(device)
    model.eval()
    
    logger.info(f"Embedding {len(queries)} queries...")
    
    # Prepare queries for embedding
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    # Embed queries in batches
    query_embeddings = {}
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[i:i+batch_size]
            batch_ids = query_ids[i:i+batch_size]
            
            # Get embeddings
            batch_embeddings = model.encode(
                batch_texts, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                device=device
            )
            
            # Store embeddings
            for qid, embedding in zip(batch_ids, batch_embeddings):
                query_embeddings[qid] = embedding.cpu()
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Embedded {min(i + batch_size, len(query_texts))}/{len(query_texts)} queries")
    
    logger.info(f"Completed embedding {len(query_embeddings)} queries")
    return query_embeddings


def load_dataset(dataset_name: str, max_queries: int = 400000):
        logger.info(f"Loading {dataset_name} dataset with max {max_queries} queries")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = util.download_and_unzip(url, 'datasets')
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split='test')
        
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

def load_dataset_and_embeddings(cache_dir: str, dataset_name: str = None, split: str = "test", 
                              model_name: str = None, device: str = 'cuda') -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load dataset, cached document embeddings, and embed queries using the model.
    
    Args:
        cache_dir: Directory containing cached document embeddings
        dataset_name: Name of the BEIR dataset. If None, extracted from cache_dir
        split: Dataset split to load
        model_name: Model name for embedding queries. If None, extracted from cache_dir
        device: Device to run the model on
        
    Returns:
        Tuple of (embeddings_cache, corpus, queries, qrels)
        Note: embeddings_cache will contain both document and query embeddings
    """
    logger.info(f"Loading dataset and embeddings from {cache_dir} for split: {split}")
    
    # Extract dataset name from cache_dir if not provided
    if dataset_name is None:
        # Extract from path like 'models/cache_e5_scifact' -> 'scifact'
        cache_dirname = os.path.basename(cache_dir)
        if 'scifact' in cache_dirname.lower():
            dataset_name = 'scifact'
        elif 'hotpot' in cache_dirname.lower():
            dataset_name = 'hotpotqa'
        elif 'msmarco' in cache_dirname.lower():
            dataset_name = 'msmarco'
        elif 'nfcorpus' in cache_dirname.lower():
            dataset_name = 'nfcorpus'
        elif 'trec' in cache_dirname.lower():
            dataset_name = 'trec-covid'
        elif 'fiqa' in cache_dirname.lower():
            dataset_name = 'fiqa'
        elif 'arguana' in cache_dirname.lower():
            dataset_name = 'arguana'
        elif 'webis' in cache_dirname.lower():
            dataset_name = 'webis-touche2020'
        elif 'quora' in cache_dirname.lower():
            dataset_name = 'quora'
        elif 'dbpedia' in cache_dirname.lower():
            dataset_name = 'dbpedia-entity'
        elif 'scidocs' in cache_dirname.lower():
            dataset_name = 'scidocs'
        elif 'fever' in cache_dirname.lower():
            dataset_name = 'fever'
        elif 'climate' in cache_dirname.lower():
            dataset_name = 'climate-fever'
        elif 'nq' in cache_dirname.lower():
            dataset_name = 'nq'
        else:
            raise ValueError(f"Cannot determine dataset name from cache_dir: {cache_dir}. Please provide dataset_name explicitly.")
    
    # Extract model name if not provided
    if model_name is None:
        model_name = extract_model_name_from_cache_dir(cache_dir)
    
    logger.info(f"Using dataset: {dataset_name}")
    logger.info(f"Using model: {model_name}")
    
    # Load document embeddings
    split_cache_dir = os.path.join(cache_dir, split)
    embeddings_path = os.path.join(split_cache_dir, "embeddings.pt")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    logger.info(f"Loading document embeddings from: {embeddings_path}")
    doc_embeddings_cache = torch.load(embeddings_path, map_location='cpu')
    logger.info(f"Loaded {len(doc_embeddings_cache)} document embeddings")
    
    # Load dataset using BEIR
    splits = load_dataset(dataset_name)
    corpus, queries, qrels = splits['test']
    
    # Embed queries using the model
    logger.info("Embedding queries using the model...")
    query_embeddings = load_model_and_embed_queries(model_name, queries, device)
    
    # Combine document and query embeddings into single cache
    # Prefix query embeddings with 'query_' to avoid conflicts
    embeddings_cache = doc_embeddings_cache.copy()
    for qid, embedding in query_embeddings.items():
        embeddings_cache[f"query_{qid}"] = embedding
    
    logger.info(f"Successfully loaded:")
    logger.info(f"  - {len(doc_embeddings_cache)} document embeddings")
    logger.info(f"  - {len(query_embeddings)} query embeddings")
    logger.info(f"  - {len(embeddings_cache)} total embeddings in cache")
    logger.info(f"  - {len(corpus)} documents") 
    logger.info(f"  - {len(queries)} queries")
    logger.info(f"  - {len(qrels)} qrel entries")
    
    return embeddings_cache, corpus, queries, qrels

class DIMEOriginalMagnitudeAnalyzer:
    """
    Original DIME Magnitude technique implementation based on the SIGIR 2024 paper:
    "Dimension Importance Estimation for Dense Information Retrieval"
    """
    
    def __init__(self, embeddings_cache: Dict[str, torch.Tensor]):
        """
        Initialize DIME analyzer with pre-computed embeddings.
        
        Args:
            embeddings_cache: Dictionary mapping document IDs to embeddings
        """
        self.embeddings_cache = embeddings_cache
        self.embedding_dim = None
        self.dimension_scores = None
        
        if embeddings_cache:
            # Get embedding dimension from first embedding
            first_embedding = next(iter(embeddings_cache.values()))
            self.embedding_dim = first_embedding.shape[0]
            logger.info(f"Initialized DIME with {len(embeddings_cache)} embeddings, "
                       f"dimension: {self.embedding_dim}")
    
    def compute_magnitude_importance(self, relevant_doc_ids: Set[str]) -> torch.Tensor:
        """
        Compute dimension importance using the original DIME magnitude approach.
        
        The original technique computes the magnitude (L2 norm) of each dimension
        across all relevant documents to determine dimension importance.
        
        Args:
            relevant_doc_ids: Set of relevant document IDs for importance estimation
            
        Returns:
            torch.Tensor: Importance scores for each dimension
        """
        logger.info(f"Computing DIME magnitude importance for {len(relevant_doc_ids)} relevant documents")
        
        # Filter embeddings for relevant documents only
        relevant_embeddings = []
        found_docs = 0
        
        for doc_id in relevant_doc_ids:
            if doc_id in self.embeddings_cache:
                relevant_embeddings.append(self.embeddings_cache[doc_id])
                found_docs += 1
        
        if not relevant_embeddings:
            raise ValueError("No relevant documents found in embeddings cache")
        
        logger.info(f"Found {found_docs}/{len(relevant_doc_ids)} relevant documents in cache")
        
        # Stack embeddings: [num_relevant_docs, embedding_dim]
        embeddings_matrix = torch.stack(relevant_embeddings, dim=0)
        
        # Compute magnitude-based importance (L2 norm across documents for each dimension)
        # This is the core DIME magnitude technique
        magnitude_scores = torch.norm(embeddings_matrix, dim=0, p=2)  # [embedding_dim]
        
        # Normalize scores to sum to 1 (probability distribution)
        magnitude_scores = magnitude_scores / magnitude_scores.sum()
        
        self.dimension_scores = magnitude_scores
        
        logger.info(f"Computed magnitude importance scores - "
                   f"min: {magnitude_scores.min():.6f}, "
                   f"max: {magnitude_scores.max():.6f}, "
                   f"mean: {magnitude_scores.mean():.6f}")
        
        return magnitude_scores
    
    def get_zero_out_dimensions(self, zero_out_ratio: float) -> torch.Tensor:
        """
        Get dimensions to zero out based on importance scores.
        
        Args:
            zero_out_ratio: Fraction of dimensions to zero out (0.2, 0.4, 0.6, 0.8)
            
        Returns:
            torch.Tensor: Boolean mask indicating dimensions to zero out
        """
        if self.dimension_scores is None:
            raise ValueError("Must compute dimension importance scores first")
        
        num_dims_to_zero = int(self.embedding_dim * zero_out_ratio)
        
        # Get indices of least important dimensions
        _, bottom_indices = torch.topk(self.dimension_scores, num_dims_to_zero, largest=False)
        
        # Create boolean mask
        zero_mask = torch.zeros(self.embedding_dim, dtype=torch.bool)
        zero_mask[bottom_indices] = True
        
        logger.info(f"Zeroing out {num_dims_to_zero} dimensions ({zero_out_ratio*100:.1f}%)")
        
        return zero_mask
    
    def apply_zero_out(self, embeddings: torch.Tensor, zero_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply dimension zeroing to embeddings.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            zero_mask: Boolean mask indicating dimensions to zero out
            
        Returns:
            torch.Tensor: Embeddings with specified dimensions zeroed out
        """
        modified_embeddings = embeddings.clone()
        modified_embeddings[:, zero_mask] = 0.0
        return modified_embeddings

class DIMEEvaluator:
    """
    Evaluator for DIME technique using standard IR metrics.
    """
    
    def __init__(self, embeddings_cache: Dict[str, torch.Tensor], 
                 corpus: Dict[str, Dict], queries: Dict[str, str], qrels: Dict[str, Dict]):
        """
        Initialize evaluator.
        
        Args:
            embeddings_cache: Pre-computed embeddings
            corpus: Document corpus
            queries: Query collection  
            qrels: Relevance judgments
        """
        self.embeddings_cache = embeddings_cache
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.dime_analyzer = None
        
        logger.info(f"Initialized evaluator with {len(corpus)} docs, "
                   f"{len(queries)} queries, {len(qrels)} qrels")
    
    def _compute_metrics(self, results: List[Tuple[str, float]], 
                        qrel: Dict[str, int], k_values: List[int]) -> Dict[str, float]:
        """
        Compute IR metrics for a single query.
        
        Args:
            results: List of (doc_id, score) tuples
            qrel: Relevance judgments for the query
            k_values: Values of k for evaluation
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Extract ranked document IDs
        ranked_docs = [doc_id for doc_id, _ in results]
        
        # Compute MRR
        mrr_score = 0.0
        for rank, doc_id in enumerate(ranked_docs, 1):
            if doc_id in qrel and qrel[doc_id] > 0:
                mrr_score = 1.0 / rank
                break
        metrics['MRR'] = mrr_score

            
        
        # Compute Recall@k and NDCG@k for each k
        for k in k_values:
            mrr_k_score = 0.0
            for rank, doc_id in enumerate(ranked_docs[:k], 1):  # Only consider top k documents
                if doc_id in qrel and qrel[doc_id] > 0:
                    mrr_k_score = 1.0 / rank
                    break
            metrics[f'MRR@{k}'] = mrr_k_score
            # Recall@k
            relevant_retrieved = 0
            total_relevant = sum(1 for rel in qrel.values() if rel > 0)
            
            for doc_id in ranked_docs[:k]:
                if doc_id in qrel and qrel[doc_id] > 0:
                    relevant_retrieved += 1
            
            recall_k = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
            metrics[f'Recall@{k}'] = recall_k
            
            # NDCG@k
            dcg = 0.0
            idcg = 0.0
            
            # DCG computation
            for rank, doc_id in enumerate(ranked_docs[:k], 1):
                relevance = qrel.get(doc_id, 0)
                if relevance > 0:
                    dcg += (2**relevance - 1) / math.log2(rank + 1)
            
            # IDCG computation
            ideal_relevances = sorted(qrel.values(), reverse=True)[:k]
            for rank, rel in enumerate(ideal_relevances, 1):
                if rel > 0:
                    idcg += (2**rel - 1) / math.log2(rank + 1)
            
            ndcg_k = dcg / idcg if idcg > 0 else 0.0
            metrics[f'NDCG@{k}'] = ndcg_k
        
        return metrics
    
    def _build_faiss_index(self, doc_embeddings: torch.Tensor) -> faiss.Index:
        """
        Build FAISS index for efficient similarity search.
        
        Args:
            doc_embeddings: Document embeddings matrix
            
        Returns:
            FAISS index
        """
        embedding_dim = doc_embeddings.shape[1]
        
        # Use inner product for cosine similarity (embeddings should be normalized)
        index = faiss.IndexFlatIP(embedding_dim)
        
        # Normalize embeddings for cosine similarity
        doc_embeddings_np = F.normalize(doc_embeddings, p=2, dim=1).numpy()
        index.add(doc_embeddings_np.astype('float32'))
        
        return index

    def _prepare_document_data(self) -> Tuple[List[str], torch.Tensor]:
        """
        Prepare consistent document IDs and embeddings for evaluation.
        This ensures the same document ordering is used in both baseline and DIME evaluation.
        
        Returns:
            Tuple of (doc_ids, doc_embeddings_tensor)
        """
        doc_ids = []
        doc_embeddings = []
        
        # Use sorted document IDs to ensure consistent ordering
        for doc_id in sorted(self.corpus.keys()):
            if doc_id in self.embeddings_cache:
                doc_ids.append(doc_id)
                doc_embeddings.append(self.embeddings_cache[doc_id])
        
        if not doc_embeddings:
            raise ValueError("No document embeddings found in cache")
        
        doc_embeddings_tensor = torch.stack(doc_embeddings)
        logger.info(f"Using {len(doc_embeddings)} documents for evaluation")
        
        return doc_ids, doc_embeddings_tensor
    
    def evaluate_baseline(self, k_values: List[int] = [5, 10, 20, 50, 100]) -> Dict[str, float]:
        """
        Evaluate baseline performance (no dimension modification).
        
        Args:
            k_values: Values of k for evaluation
            
        Returns:
            Dictionary of averaged metric scores
        """
        logger.info("Evaluating baseline performance...")
        
        # Prepare document embeddings and IDs
        doc_ids, doc_embeddings_tensor = self._prepare_document_data()
        
        # Build FAISS index
        index = self._build_faiss_index(doc_embeddings_tensor)
        
        # Initialize metric collectors
        all_metrics = defaultdict(list)
        query_times = []
        processed_queries = 0
        
        # Process each query
        for qid, query_text in tqdm(self.queries.items(), desc="Evaluating queries"):
            if qid not in self.qrels:
                logger.warning(f"No qrels found for query {qid}")
                continue
            
            # Get query embedding - try both formats
            query_embedding = None
            for query_key in [f"query_{qid}", qid]:
                if query_key in self.embeddings_cache:
                    query_embedding = self.embeddings_cache[query_key]
                    break
            
            if query_embedding is None:
                logger.warning(f"Query embedding not found for {qid}")
                continue
            
            # Measure query time
            start_time = time.time()
            results = self._search_query(query_embedding, index, doc_ids, k=max(k_values)*2)
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # Compute metrics
            metrics = self._compute_metrics(results, self.qrels[qid], k_values)
            
            # Collect metrics
            for metric_name, score in metrics.items():
                all_metrics[metric_name].append(score)
            
            processed_queries += 1
        
        if processed_queries == 0:
            raise ValueError("No queries could be processed")
        
        # Average metrics
        averaged_metrics = {}
        for metric_name, scores in all_metrics.items():
            averaged_metrics[metric_name] = mean(scores) if scores else 0.0
        
        # Add timing information
        averaged_metrics['avg_query_time'] = mean(query_times) if query_times else 0.0
        averaged_metrics['num_queries'] = processed_queries
        
        logger.info(f"Baseline evaluation completed. Processed {processed_queries} queries.")
        logger.info(f"Average query time: {averaged_metrics['avg_query_time']:.4f} seconds")
        # logger.info(f"Sample metrics - MRR: {averaged_metrics.get('MRR', 0):.4f}, "
        #             f"NDCG@10: {averaged_metrics.get('NDCG@10', 0):.4f}")
        
        return averaged_metrics

    def _search_query(self, query_embedding: torch.Tensor, index: faiss.Index, 
                    doc_ids: List[str], k: int = 1000) -> List[Tuple[str, float]]:
        """
        Search for a single query.
        
        Args:
            query_embedding: Query embedding
            index: FAISS index
            doc_ids: List of document IDs corresponding to index
            k: Number of results to retrieve
            
        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        # Normalize query embedding for cosine similarity
        query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        
        # Ensure k doesn't exceed index size
        actual_k = min(k, index.ntotal)
        
        # Search
        scores, indices = index.search(query_norm.numpy().astype('float32'), actual_k)
        
        # Format results and sort by score (descending)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(doc_ids):
                results.append((doc_ids[idx], float(score)))
        
        # Sort by score descending (higher similarity first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def calculate_performance_delta(self, baseline_metrics: Dict[str, float], 
                                  dime_results: Dict[float, Dict[str, float]]) -> Dict[float, Dict[str, float]]:
        """
        Calculate percentage change in performance compared to baseline.
        
        Args:
            baseline_metrics: Baseline performance metrics
            dime_results: DIME results for different zero-out ratios
            
        Returns:
            Dictionary mapping zero_out_ratio to percentage changes for each metric
        """
        logger.info("Calculating performance deltas compared to baseline...")
        
        delta_results = {}
        
        for zero_ratio, dime_metrics in dime_results.items():
            delta_metrics = {}
            
            for metric_name, dime_value in dime_metrics.items():
                # Skip non-performance metrics
                if metric_name in ['avg_query_time', 'num_queries']:
                    delta_metrics[metric_name] = dime_value
                    continue
                
                if metric_name in baseline_metrics:
                    baseline_value = baseline_metrics[metric_name]
                    
                    # Calculate percentage change
                    if baseline_value == 0:
                        # Handle division by zero - if baseline is 0, set change to 0 or inf appropriately
                        if dime_value == 0:
                            percent_change = 0.0
                        else:
                            percent_change = float('inf') if dime_value > 0 else float('-inf')
                    else:
                        percent_change = ((dime_value - baseline_value) / baseline_value) * 100
                    
                    delta_metrics[f'{metric_name}_delta_%'] = percent_change
                    delta_metrics[f'{metric_name}_baseline'] = baseline_value
                    delta_metrics[f'{metric_name}_dime'] = dime_value
                else:
                    logger.warning(f"Metric {metric_name} not found in baseline results")
            
            delta_results[zero_ratio] = delta_metrics
            
            # Log key metrics for this ratio
            key_metrics = ['MRR@10_delta_%']
            log_parts = []
            for key_metric in key_metrics:
                if key_metric in delta_metrics:
                    log_parts.append(f"{key_metric.replace('_delta_%', '')}: {delta_metrics[key_metric]:+.2f}%")
            
            if log_parts:
                logger.info(f"Zero ratio {zero_ratio:.1f} - Performance changes: {', '.join(log_parts)}")
        
        return delta_results
    
    # def print_performance_summary(self, baseline_metrics: Dict[str, float],
    #                             delta_results: Dict[float, Dict[str, float]],
    #                             key_metrics: List[str] = ['MRR', 'NDCG@10', 'NDCG@20', 'Recall@10', 'Recall@20']):
    #     """
    #     Print a formatted summary of performance changes.
        
    #     Args:
    #         baseline_metrics: Baseline performance metrics
    #         delta_results: Performance delta results from calculate_performance_delta
    #         key_metrics: List of key metrics to include in summary
    #     """
    #     print("\n" + "="*80)
    #     print("DIME PERFORMANCE SUMMARY")
    #     print("="*80)
        
    #     # Print baseline performance
    #     print(f"\nBASELINE PERFORMANCE:")
    #     print("-" * 40)
    #     for metric in key_metrics:
    #         if metric in baseline_metrics:
    #             print(f"{metric:>15}: {baseline_metrics[metric]:.4f}")
        
    #     # Print performance changes for each zero ratio
    #     print(f"\nPERFORMANCE CHANGES (% vs Baseline):")
    #     print("-" * 40)
        
    #     # Header
    #     header_parts = ["Zero Ratio"]
    #     for metric in key_metrics:
    #         header_parts.append(f"{metric}")
    #     print(f"{'':>12} | {' | '.join(f'{m:>10}' for m in key_metrics)}")
    #     print("-" * (12 + len(key_metrics) * 13))
        
    #     # Data rows
    #     for zero_ratio in sorted(delta_results.keys()):
    #         delta_metrics = delta_results[zero_ratio]
    #         row_parts = [f"{zero_ratio:>10.1f}"]
            
    #         for metric in key_metrics:
    #             delta_key = f"{metric}_delta_%"
    #             if delta_key in delta_metrics:
    #                 delta_value = delta_metrics[delta_key]
    #                 if abs(delta_value) == float('inf'):
    #                     row_parts.append(f"{'∞':>10}")
    #                 else:
    #                     row_parts.append(f"{delta_value:>+9.2f}%")
    #             else:
    #                 row_parts.append(f"{'N/A':>10}")
            
    #         print(" | ".join(row_parts))
        
    #     # Print interpretation
    #     print("\nINTERPRETation:")
    #     print("-" * 40)
    #     print("• Positive values indicate improvement over baseline")
    #     print("• Negative values indicate degradation from baseline") 
    #     print("• Zero ratio 0.0 should be ≈0% (identical to baseline)")
    #     print("• Higher zero ratios typically show degradation")
    #     print("="*80)
    
    def evaluate_with_dime(self, zero_out_ratios: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8],
                          k_values: List[int] = [5, 10, 20, 50, 100]) -> Dict[float, Dict[str, float]]:
        """
        Evaluate performance with DIME dimension zeroing.
        
        Args:
            zero_out_ratios: Ratios of dimensions to zero out
            k_values: Values of k for evaluation
            
        Returns:
            Dictionary mapping zero_out_ratio to metric scores
        """
        logger.info("Evaluating with DIME dimension zeroing...")
        
        # Initialize DIME analyzer
        self.dime_analyzer = DIMEOriginalMagnitudeAnalyzer(self.embeddings_cache)
        
        # Get all relevant document IDs for importance estimation
        relevant_doc_ids = set()
        for qrel in self.qrels.values():
            for doc_id, relevance in qrel.items():
                if relevance > 0:
                    relevant_doc_ids.add(doc_id)
        
        # Compute dimension importance
        self.dime_analyzer.compute_magnitude_importance(relevant_doc_ids)
        
        # Prepare consistent document data (same as baseline)
        doc_ids, original_doc_embeddings = self._prepare_document_data()
        
        results = {}
        
        for zero_ratio in zero_out_ratios:
            logger.info(f"Evaluating with {zero_ratio*100:.1f}% dimensions zeroed out")
            
            # Get zero-out mask
            if zero_ratio == 0.0:
                # No dimensions to zero out
                zero_mask = torch.zeros(self.dime_analyzer.embedding_dim, dtype=torch.bool)
            else:
                zero_mask = self.dime_analyzer.get_zero_out_dimensions(zero_ratio)
            
            # Apply zero-out to document embeddings
            modified_doc_embeddings = self.dime_analyzer.apply_zero_out(
                original_doc_embeddings, zero_mask
            )
            
            # Build FAISS index with modified embeddings
            index = self._build_faiss_index(modified_doc_embeddings)
            
            # Initialize metric collectors
            all_metrics = defaultdict(list)
            query_times = []
            processed_queries = 0
            
            # Process each query
            for qid, query_text in tqdm(self.queries.items(), 
                                      desc=f"Zero ratio {zero_ratio:.1f}"):
                if qid not in self.qrels:
                    continue
                
                # Get query embedding - try both formats
                original_query_emb = None
                for query_key in [f"query_{qid}", qid]:
                    if query_key in self.embeddings_cache:
                        original_query_emb = self.embeddings_cache[query_key]
                        break
                
                if original_query_emb is None:
                    continue
                
                # Apply zero-out to query embedding
                modified_query_emb = self.dime_analyzer.apply_zero_out(
                    original_query_emb.unsqueeze(0), zero_mask
                ).squeeze(0)
                
                # Measure query time
                start_time = time.time()
                results_query = self._search_query(modified_query_emb, index, doc_ids)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # Compute metrics
                metrics = self._compute_metrics(results_query, self.qrels[qid], k_values)
                
                # Collect metrics
                for metric_name, score in metrics.items():
                    all_metrics[metric_name].append(score)
                
                processed_queries += 1
            
            # Average metrics for this zero ratio
            averaged_metrics = {}
            for metric_name, scores in all_metrics.items():
                averaged_metrics[metric_name] = mean(scores) if scores else 0.0
            
            # Add timing information
            averaged_metrics['avg_query_time'] = mean(query_times) if query_times else 0.0
            averaged_metrics['num_queries'] = processed_queries
            
            results[zero_ratio] = averaged_metrics
            
            # logger.info(f"Zero ratio {zero_ratio:.1f} completed. "
            #            f"NDCG@10: {averaged_metrics.get('NDCG@10', 0):.4f}, "
            #            f"MRR: {averaged_metrics.get('MRR', 0):.4f}")
        
        return results

def load_cached_data_like_code2(cache_dir: str, split: str = "test") -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load cached embeddings and dataset files following code 2's pattern.
    
    Args:
        cache_dir: Directory containing cached files (e.g., 'models/cache_e5_hotpotqa')
        split: Dataset split to load ('test', 'train', 'val')
        
    Returns:
        Tuple of (embeddings_cache, corpus, queries, qrels)
    """
    logger.info(f"Loading cached data from {cache_dir} for split: {split}")
    
    # Construct the embeddings path like code 2 does
    # Code 2 uses: os.path.join(cfg['cache_dir'], 'test')
    split_cache_dir = os.path.join(cache_dir, split)
    
    # Load embeddings - following code 2's pattern
    # Code 2 loads with: torch.load(test_embeddings_path, map_location='cpu')
    embeddings_path = os.path.join(split_cache_dir, "embeddings.pt")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    logger.info(f"Loading embeddings from: {embeddings_path}")
    embeddings_cache = torch.load(embeddings_path, map_location='cpu')
    logger.info(f"Loaded {len(embeddings_cache)} embeddings")
    
    # Load dataset files - try split directory first, then parent
    def find_data_file(filename: str) -> Optional[str]:
        # Try split directory first
        filepath = os.path.join(split_cache_dir, filename)
        if os.path.exists(filepath):
            return filepath
        
        # Try parent directory
        filepath = os.path.join(cache_dir, filename)
        if os.path.exists(filepath):
            return filepath
            
        # Try without split subdirectory (direct in cache_dir)
        return None
    
    # Load corpus
    corpus = {}
    corpus_path = find_data_file("corpus.jsonl")
    if corpus_path:
        logger.info(f"Loading corpus from: {corpus_path}")
        with open(corpus_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["_id"]] = {"title": doc.get("title", ""), "text": doc["text"]}
    else:
        logger.warning("Corpus file not found")
    
    # Load queries
    queries = {}
    queries_path = find_data_file("queries.jsonl")
    if queries_path:
        logger.info(f"Loading queries from: {queries_path}")
        with open(queries_path, 'r') as f:
            for line in f:
                query = json.loads(line)
                queries[query["_id"]] = query["text"]
    else:
        logger.warning("Queries file not found")
    
    # Load qrels
    qrels = {}
    qrels_path = find_data_file("qrels.tsv")
    if qrels_path:
        logger.info(f"Loading qrels from: {qrels_path}")
        with open(qrels_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    qid, _, doc_id, relevance = parts[:4]
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][doc_id] = int(relevance)
    else:
        logger.warning("Qrels file not found")
    
    logger.info(f"Successfully loaded:")
    logger.info(f"  - {len(embeddings_cache)} embeddings")
    logger.info(f"  - {len(corpus)} documents") 
    logger.info(f"  - {len(queries)} queries")
    logger.info(f"  - {len(qrels)} qrel entries")
    
    return embeddings_cache, corpus, queries, qrels

# def print_results(baseline_results: Dict[str, float], 
#                  dime_results: Dict[float, Dict[str, float]]):
#     """
#     Print evaluation results in a formatted table.
#     """
#     print("\n" + "="*80)
#     print("DIME MAGNITUDE EVALUATION RESULTS")
#     print("="*80)
    
#     # Print baseline
#     print("\nBASELINE PERFORMANCE:")
#     print("-" * 50)
#     print(f"MRR:           {baseline_results.get('MRR', 0):.4f}")
#     for k in [5, 10, 20, 50, 100]:
#         ndcg_key = f'NDCG@{k}'
#         recall_key = f'Recall@{k}'
#         if ndcg_key in baseline_results and recall_key in baseline_results:
#             print(f"NDCG@{k:2d}:       {baseline_results[ndcg_key]:.4f}    "
#                   f"Recall@{k:2d}:     {baseline_results[recall_key]:.4f}")
#     print(f"Avg Query Time: {baseline_results.get('avg_query_time', 0)*1000:.2f}ms")
#     print(f"Total Queries:  {baseline_results.get('num_queries', 0)}")
    
#     # Print DIME results
#     print("\nDIME RESULTS (Dimension Zeroing):")
#     print("-" * 80)
#     print(f"{'Zero %':>8} {'MRR':>8} {'NDCG@5':>8} {'NDCG@10':>8} {'NDCG@20':>8} "
#           f"{'Recall@5':>10} {'Recall@10':>11} {'Time(ms)':>10}")
#     print("-" * 80)
    
#     for zero_ratio in sorted(dime_results.keys()):
#         results = dime_results[zero_ratio]
#         print(f"{zero_ratio*100:>7.1f}% "
#               f"{results.get('MRR', 0):>7.4f} "
#               f"{results.get('NDCG@5', 0):>7.4f} "
#               f"{results.get('NDCG@10', 0):>7.4f} "
#               f"{results.get('NDCG@20', 0):>7.4f} "
#               f"{results.get('Recall@5', 0):>9.4f} "
#               f"{results.get('Recall@10', 0):>10.4f} "
#               f"{results.get('avg_query_time', 0)*1000:>9.2f}")

def get_config():
    """Define and return configuration parameters."""
    return {
        "cache_dir": "models/cache_mini_webis-touche2020",
        "dataset_name": "webis-touche2020",  # Explicitly specify dataset name
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Explicitly specify model
        "device": "cuda",  # or "cpu"
        "split": "test",
        "output_dir": "dime_results",
        "zero_ratios": [0.0, 0.2, 0.4, 0.6, 0.8],
        "k_values": [5, 10, 20, 50, 100]
    }

def main():
    # Get configuration
    config = get_config()
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    try:
        # Load cached data following code 2's pattern
        embeddings_cache, corpus, queries, qrels = load_dataset_and_embeddings(
            config["cache_dir"], 
            config.get("dataset_name"),  # Can be None for auto-detection
            config["split"],
            config.get("model_name"),    # Can be None for auto-detection
            config.get("device", "cuda")
        )
        
        # Initialize evaluator
        evaluator = DIMEEvaluator(embeddings_cache, corpus, queries, qrels)
        
        # Evaluate baseline
        logger.info("Starting baseline evaluation...")
        baseline_results = evaluator.evaluate_baseline(config["k_values"])
        
        # Evaluate with DIME
        logger.info("Starting DIME evaluation...")
        dime_results = evaluator.evaluate_with_dime(config["zero_ratios"], config["k_values"])

        # Calculate performance deltas
        delta_results = evaluator.calculate_performance_delta(baseline_results, dime_results)

        # Print formatted summary
        # evaluator.print_performance_summary(baseline_results, delta_results)
        
        # Print results
        # print_results(baseline_results, dime_results)
        
        # Save results
        results_data = {
            'baseline': baseline_results,
            'dime': dime_results,
            "delta": delta_results,
            'config': {
                'cache_dir': config["cache_dir"],
                'split': config["split"],
                'zero_ratios': config["zero_ratios"],
                'k_values': config["k_values"]
            }
        }
        
        results_path = os.path.join(config["output_dir"], f"dime_results_mini_{config['split']}_webis-touche2020.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()