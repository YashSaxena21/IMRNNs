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
from embedding_adapter import EmbeddingAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("search_adaptor_evaluation.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_embed_queries(model_name: str, queries: Dict[str, str], device: str = 'cuda') -> Dict[str, torch.Tensor]:
    logger.info(f"Loading model: {model_name}")
    
    model = SentenceTransformer(model_name)
    model.to(device)
    model.eval()
    
    logger.info(f"Embedding {len(queries)} queries...")
    
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    query_embeddings = {}
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[i:i+batch_size]
            batch_ids = query_ids[i:i+batch_size]
            
            batch_embeddings = model.encode(
                batch_texts, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                device=device
            )
            
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
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split='train')
    
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

def extract_model_name_from_cache_dir(cache_dir: str) -> str:
    cache_dirname = os.path.basename(cache_dir)
    if 'e5' in cache_dirname.lower():
        return 'sentence-transformers/all-MiniLM-L6-v2'
    return 'sentence-transformers/all-MiniLM-L6-v2'

def load_split_embeddings_and_data(cache_dir: str, dataset_name: str, split: str, 
                                 model_name: str = None, device: str = 'cuda') -> Tuple[Dict, Dict, Dict, Dict]:
    """Load embeddings and data for a specific split"""
    logger.info(f"Loading {split} split embeddings and data")
    
    if model_name is None:
        model_name = extract_model_name_from_cache_dir(cache_dir)
    
    # Load the split-specific embeddings from cache
    split_cache_dir = os.path.join(cache_dir, split)
    embeddings_path = os.path.join(split_cache_dir, "embeddings.pt")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    logger.info(f"Loading {split} embeddings from: {embeddings_path}")
    doc_embeddings_cache = torch.load(embeddings_path, map_location='cpu')
    logger.info(f"Loaded {len(doc_embeddings_cache)} {split} document embeddings")
    
    # Load dataset and get the appropriate split
    splits = load_dataset(dataset_name)
    corpus, queries, qrels = splits[split]
    
    # For the requested split, we need to embed the queries since they're not in the cache
    logger.info(f"Embedding {len(queries)} {split} queries using model...")
    query_embeddings = load_model_and_embed_queries(model_name, queries, device)
    
    # Combine document and query embeddings
    embeddings_cache = doc_embeddings_cache.copy()
    for qid, embedding in query_embeddings.items():
        embeddings_cache[f"query_{qid}"] = embedding
    
    logger.info(f"Successfully loaded {split} data:")
    logger.info(f"  - {len(doc_embeddings_cache)} document embeddings")
    logger.info(f"  - {len(query_embeddings)} query embeddings") 
    logger.info(f"  - {len(queries)} queries, {len(qrels)} qrel entries")
    
    return embeddings_cache, corpus, queries, qrels

def load_dataset_and_embeddings(cache_dir: str, dataset_name: str = None, split: str = "test", 
                              model_name: str = None, device: str = 'cuda') -> Tuple[Dict, Dict, Dict, Dict]:
    """Load dataset and embeddings for evaluation - wrapper around load_split_embeddings_and_data"""
    logger.info(f"Loading dataset and embeddings from {cache_dir} for split: {split}")
    
    if dataset_name is None:
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
    
    if model_name is None:
        model_name = extract_model_name_from_cache_dir(cache_dir)
    
    logger.info(f"Using dataset: {dataset_name}")
    logger.info(f"Using model: {model_name}")
    
    return load_split_embeddings_and_data(cache_dir, dataset_name, split, model_name, device)

class SearchAdaptorAnalyzer:
    def __init__(self, embeddings_cache: Dict[str, torch.Tensor]):
        self.embeddings_cache = embeddings_cache
        self.adapter = None
        self.embedding_dim = None
        
        if embeddings_cache:
            first_embedding = next(iter(embeddings_cache.values()))
            self.embedding_dim = first_embedding.shape[0]
            logger.info(f"Initialized Search Adaptor with {len(embeddings_cache)} embeddings, "
                       f"dimension: {self.embedding_dim}")
    
    def prepare_training_data(self, embeddings_cache: Dict, queries_data: Dict, qrels_data: Dict, split: str):
        """Prepare training data using the provided embeddings cache for the specific split"""
        logger.info(f"Preparing training data for {split} split")
        
        query_embeddings = []
        doc_embeddings = []
        labels = []
        
        logger.info(f"Available embeddings keys sample: {list(embeddings_cache.keys())[:5]}")
        logger.info(f"Available queries: {list(queries_data.keys())[:5]}")
        logger.info(f"Available qrels: {list(qrels_data.keys())[:5]}")
        
        for qid, query_text in queries_data.items():
            if qid not in qrels_data:
                continue
            
            # Try different query key formats
            query_emb = None
            for query_key in [f"query_{qid}", qid, f"q_{qid}"]:
                if query_key in embeddings_cache:
                    query_emb = embeddings_cache[query_key]
                    break
            
            if query_emb is None:
                logger.debug(f"Query embedding not found for {qid}")
                continue
                
            for doc_id, relevance in qrels_data[qid].items():
                if doc_id not in embeddings_cache:
                    logger.debug(f"Document embedding not found for {doc_id}")
                    continue
                    
                doc_emb = embeddings_cache[doc_id]
                
                query_embeddings.append(query_emb.numpy())
                doc_embeddings.append(doc_emb.numpy())
                labels.append(1 if relevance > 0 else 0)
        
        if len(labels) == 0:
            logger.warning(f"No training examples found for {split} split!")
            logger.warning("This might be due to:")
            logger.warning("1. Mismatch between query IDs in qrels and embeddings")
            logger.warning("2. Mismatch between document IDs in qrels and embeddings")
            logger.warning("3. No relevant documents found in the embeddings")
            return np.array([]), np.array([]), np.array([])
        
        logger.info(f"Prepared {len(labels)} training examples for {split}")
        return np.array(query_embeddings), np.array(doc_embeddings), np.array(labels)
    
    def train_adapter(self, train_data: Tuple, val_data: Tuple = None):
        logger.info("Training Search Adaptor...")
        
        query_embeddings, doc_embeddings, labels = train_data
        
        if len(labels) == 0:
            raise ValueError("No training data available. Cannot train adapter.")
        
        self.adapter = EmbeddingAdapter()
        self.adapter.fit(query_embeddings, doc_embeddings, labels)
        
        logger.info("Search Adaptor training completed")
    
    def transform_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.adapter is None:
            raise ValueError("Adapter must be trained first")
        
        if isinstance(embeddings, torch.Tensor):
            original_shape = embeddings.shape
            embeddings_np = embeddings.numpy()
        else:
            embeddings_np = embeddings
            original_shape = embeddings_np.shape
        
        # Handle both single embeddings and batched embeddings
        if embeddings_np.ndim == 1:
            # Single embedding - pass as 1D array
            transformed = self.adapter.transform(embeddings_np)
        elif embeddings_np.ndim == 2:
            # Batch of embeddings - process each embedding individually as 1D
            batch_size, embedding_dim = embeddings_np.shape
            transformed_list = []
            
            for i in range(batch_size):
                single_embedding = embeddings_np[i]  # Get as 1D array
                transformed_single = self.adapter.transform(single_embedding)
                transformed_list.append(transformed_single)
            
            # Stack all transformed embeddings
            transformed = np.vstack(transformed_list)
        else:
            raise ValueError(f"Unsupported embedding shape: {original_shape}")
        
        if isinstance(embeddings, torch.Tensor):
            return torch.from_numpy(transformed)
        return transformed

class SearchAdaptorEvaluator:
    def __init__(self, embeddings_cache: Dict[str, torch.Tensor], 
                 corpus: Dict[str, Dict], queries: Dict[str, str], qrels: Dict[str, Dict]):
        self.embeddings_cache = embeddings_cache
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.adaptor_analyzer = None
        
        logger.info(f"Initialized evaluator with {len(corpus)} docs, "
                   f"{len(queries)} queries, {len(qrels)} qrels")
    
    def _compute_metrics(self, results: List[Tuple[str, float]], 
                        qrel: Dict[str, int], k_values: List[int]) -> Dict[str, float]:
        metrics = {}
        
        ranked_docs = [doc_id for doc_id, _ in results]
        
        # Overall MRR (across all results)
        mrr_score = 0.0
        for rank, doc_id in enumerate(ranked_docs, 1):
            if doc_id in qrel and qrel[doc_id] > 0:
                mrr_score = 1.0 / rank
                break
        metrics['MRR'] = mrr_score
        
        for k in k_values:
            # MRR@k - only consider first k results
            mrr_k = 0.0
            for rank, doc_id in enumerate(ranked_docs[:k], 1):
                if doc_id in qrel and qrel[doc_id] > 0:
                    mrr_k = 1.0 / rank
                    break
            metrics[f'MRR@{k}'] = mrr_k
            
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
            
            for rank, doc_id in enumerate(ranked_docs[:k], 1):
                relevance = qrel.get(doc_id, 0)
                if relevance > 0:
                    dcg += (2**relevance - 1) / math.log2(rank + 1)
            
            ideal_relevances = sorted(qrel.values(), reverse=True)[:k]
            for rank, rel in enumerate(ideal_relevances, 1):
                if rel > 0:
                    idcg += (2**rel - 1) / math.log2(rank + 1)
            
            ndcg_k = dcg / idcg if idcg > 0 else 0.0
            metrics[f'NDCG@{k}'] = ndcg_k
        
        return metrics

    
    def _build_faiss_index(self, doc_embeddings: torch.Tensor) -> faiss.Index:
        embedding_dim = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        doc_embeddings_np = F.normalize(doc_embeddings, p=2, dim=1).numpy()
        index.add(doc_embeddings_np.astype('float32'))
        return index

    def _prepare_document_data(self) -> Tuple[List[str], torch.Tensor]:
        doc_ids = []
        doc_embeddings = []
        
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
        logger.info("Evaluating baseline performance...")
        
        doc_ids, doc_embeddings_tensor = self._prepare_document_data()
        index = self._build_faiss_index(doc_embeddings_tensor)
        
        all_metrics = defaultdict(list)
        query_times = []
        processed_queries = 0
        
        for qid, query_text in tqdm(self.queries.items(), desc="Evaluating queries"):
            if qid not in self.qrels:
                logger.warning(f"No qrels found for query {qid}")
                continue
            
            query_embedding = None
            for query_key in [f"query_{qid}", qid]:
                if query_key in self.embeddings_cache:
                    query_embedding = self.embeddings_cache[query_key]
                    break
            
            if query_embedding is None:
                logger.warning(f"Query embedding not found for {qid}")
                continue
            
            start_time = time.time()
            results = self._search_query(query_embedding, index, doc_ids, k=max(k_values)*2)
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            metrics = self._compute_metrics(results, self.qrels[qid], k_values)
            
            for metric_name, score in metrics.items():
                all_metrics[metric_name].append(score)
            
            processed_queries += 1
        
        if processed_queries == 0:
            raise ValueError("No queries could be processed")
        
        averaged_metrics = {}
        for metric_name, scores in all_metrics.items():
            averaged_metrics[metric_name] = mean(scores) if scores else 0.0
        
        averaged_metrics['avg_query_time'] = mean(query_times) if query_times else 0.0
        averaged_metrics['num_queries'] = processed_queries
        
        logger.info(f"Baseline evaluation completed. Processed {processed_queries} queries.")
        logger.info(f"Average query time: {averaged_metrics['avg_query_time']:.4f} seconds")
        logger.info(f"Sample metrics - MRR: {averaged_metrics.get('MRR', 0):.4f}, "
                    f"NDCG@10: {averaged_metrics.get('NDCG@10', 0):.4f}")
        
        return averaged_metrics

    def _search_query(self, query_embedding: torch.Tensor, index: faiss.Index, 
                    doc_ids: List[str], k: int = 1000) -> List[Tuple[str, float]]:
        query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        actual_k = min(k, index.ntotal)
        scores, indices = index.search(query_norm.numpy().astype('float32'), actual_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(doc_ids):
                results.append((doc_ids[idx], float(score)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def evaluate_with_search_adaptor(self, cache_dir: str, dataset_name: str, k_values: List[int] = [5, 10, 20, 50, 100]) -> Dict[str, float]:
        logger.info("Evaluating with Search Adaptor...")
        
        # Load training data from the train split - USE THE SAME CACHE DIR
        # This ensures the embeddings come from the same model/source
        train_embeddings_cache, train_corpus, train_queries, train_qrels = load_split_embeddings_and_data(
            cache_dir, dataset_name, 'train'
        )
        
        # Debug: Check embedding dimensions
        if train_embeddings_cache:
            first_train_emb = next(iter(train_embeddings_cache.values()))
            logger.info(f"Train embedding dimension: {first_train_emb.shape[0]}")
        
        if self.embeddings_cache:
            first_test_emb = next(iter(self.embeddings_cache.values()))
            logger.info(f"Test embedding dimension: {first_test_emb.shape[0]}")
        
        # Initialize adapter analyzer with training embeddings
        self.adaptor_analyzer = SearchAdaptorAnalyzer(train_embeddings_cache)
        
        # Prepare training data using the train split embeddings
        train_data = self.adaptor_analyzer.prepare_training_data(
            train_embeddings_cache, train_queries, train_qrels, 'train'
        )
        
        # Check if we have training data
        if len(train_data[2]) == 0:
            logger.error("No training data available. Cannot train adapter.")
            raise ValueError("No training data available for adapter training")
        
        # Train the adapter
        self.adaptor_analyzer.train_adapter(train_data)
        
        # Now prepare test documents for evaluation
        doc_ids, original_doc_embeddings = self._prepare_document_data()
        
        # Check dimension consistency before transformation
        logger.info(f"Original doc embeddings shape: {original_doc_embeddings.shape}")
        logger.info(f"Adapter expected dimension: {self.adaptor_analyzer.adapter._embedding_len}")
        
        # Transform test document embeddings using trained adapter
        adapted_doc_embeddings = self.adaptor_analyzer.transform_embeddings(original_doc_embeddings)
        index = self._build_faiss_index(adapted_doc_embeddings)
        
        all_metrics = defaultdict(list)
        query_times = []
        processed_queries = 0
        
        for qid, query_text in tqdm(self.queries.items(), desc="Search Adaptor evaluation"):
            if qid not in self.qrels:
                continue
            
            original_query_emb = None
            for query_key in [f"query_{qid}", qid]:
                if query_key in self.embeddings_cache:
                    original_query_emb = self.embeddings_cache[query_key]
                    break
            
            if original_query_emb is None:
                continue
            
            # Transform test query embedding using trained adapter
            adapted_query_emb = self.adaptor_analyzer.transform_embeddings(original_query_emb.unsqueeze(0)).squeeze(0)
            
            start_time = time.time()
            results_query = self._search_query(adapted_query_emb, index, doc_ids)
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            metrics = self._compute_metrics(results_query, self.qrels[qid], k_values)
            
            for metric_name, score in metrics.items():
                all_metrics[metric_name].append(score)
            
            processed_queries += 1
        
        averaged_metrics = {}
        for metric_name, scores in all_metrics.items():
            averaged_metrics[metric_name] = mean(scores) if scores else 0.0
        
        averaged_metrics['avg_query_time'] = mean(query_times) if query_times else 0.0
        averaged_metrics['num_queries'] = processed_queries
        
        logger.info(f"Search Adaptor evaluation completed. "
                   f"NDCG@10: {averaged_metrics.get('NDCG@10', 0):.4f}, "
                   f"MRR: {averaged_metrics.get('MRR', 0):.4f}")
        
        return averaged_metrics

def print_results(baseline_results: Dict[str, float], adaptor_results: Dict[str, float]):
    print("\n" + "="*80)
    print("SEARCH ADAPTOR EVALUATION RESULTS")
    print("="*80)
    
    print("\nBASELINE PERFORMANCE:")
    print("-" * 50)
    print(f"MRR:           {baseline_results.get('MRR', 0):.4f}")
    for k in [5, 10, 20, 50, 100]:
        ndcg_key = f'NDCG@{k}'
        recall_key = f'Recall@{k}'
        if ndcg_key in baseline_results and recall_key in baseline_results:
            print(f"NDCG@{k:2d}:       {baseline_results[ndcg_key]:.4f}    "
                  f"Recall@{k:2d}:     {baseline_results[recall_key]:.4f}")
    print(f"Avg Query Time: {baseline_results.get('avg_query_time', 0)*1000:.2f}ms")
    print(f"Total Queries:  {baseline_results.get('num_queries', 0)}")
    
    print("\nSEARCH ADAPTOR PERFORMANCE:")
    print("-" * 50)
    print(f"MRR:           {adaptor_results.get('MRR', 0):.4f}")
    for k in [5, 10, 20, 50, 100]:
        ndcg_key = f'NDCG@{k}'
        recall_key = f'Recall@{k}'
        if ndcg_key in adaptor_results and recall_key in adaptor_results:
            print(f"NDCG@{k:2d}:       {adaptor_results[ndcg_key]:.4f}    "
                  f"Recall@{k:2d}:     {adaptor_results[recall_key]:.4f}")
    print(f"Avg Query Time: {adaptor_results.get('avg_query_time', 0)*1000:.2f}ms")
    print(f"Total Queries:  {adaptor_results.get('num_queries', 0)}")
    
    print("\nIMPROVEMENT:")
    print("-" * 50)
    for metric in ['MRR', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'Recall@5', 'Recall@10', 'Recall@20']:
        if metric in baseline_results and metric in adaptor_results:
            baseline_val = baseline_results[metric]
            adaptor_val = adaptor_results[metric]
            if baseline_val > 0:
                improvement = ((adaptor_val - baseline_val) / baseline_val) * 100
                print(f"{metric:>12}: {improvement:+6.2f}%")

def get_config():
    return {
        "cache_dir": "models/cache_fresh_run",
        "dataset_name": "msmarco",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cuda",
        "split": "test",
        "output_dir": "search_adaptor_results",
        "k_values": [5, 10, 20, 50, 100]
    }

def main():
    config = get_config()
    os.makedirs(config["output_dir"], exist_ok=True)
    
    try:
        # Load test data for evaluation
        embeddings_cache, corpus, queries, qrels = load_dataset_and_embeddings(
            config["cache_dir"], 
            config.get("dataset_name"),
            config["split"],
            config.get("model_name"),
            config.get("device", "cuda")
        )
        
        evaluator = SearchAdaptorEvaluator(embeddings_cache, corpus, queries, qrels)
        
        logger.info("Starting baseline evaluation...")
        baseline_results = evaluator.evaluate_baseline(config["k_values"])
        
        logger.info("Starting Search Adaptor evaluation...")
        adaptor_results = evaluator.evaluate_with_search_adaptor(
            config["cache_dir"], 
            config.get("dataset_name", "webis-touche2020"), 
            config["k_values"]
        )
        
        print_results(baseline_results, adaptor_results)
        
        results_data = {
            'baseline': baseline_results,
            'search_adaptor': adaptor_results,
            'config': {
                'cache_dir': config["cache_dir"],
                'split': config["split"],
                'k_values': config["k_values"]
            }
        }
        
        results_path = os.path.join(config["output_dir"], f"search_adaptor_results_mini_{config['split']}_ms.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()