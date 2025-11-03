import os
import random
import logging
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from transformers import get_linear_schedule_with_warmup

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_DIR = os.path.join(os.getcwd(), "models", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("e5_trec-covid.log", mode="w"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MSMARCODatasetLoader:
    def __init__(self, dataset_name: str = 'trec-covid'):
        self.dataset_name = dataset_name

    def load_dataset(self, max_queries: int = 400000):
        logger.info(f"Loading {self.dataset_name} dataset with max {max_queries} queries")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
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

class InvertedIndexBM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.postings = defaultdict(list)
        self.idf = {}
        self.doc_lengths = {}
        self.avgdl = 0
        self.doc_ids = []
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'])
        
    def tokenize(self, text):
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [t for t in tokens if t not in self.stopwords and len(t) > 2]
    
    def get_top_k(self, query: str, K: int = 1000) -> List[str]:
        tokens = self.tokenize(query)
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        for term in tokens:
            if term not in self.postings:
                continue
            idf_val = self.idf[term]
            for doc_idx, tf in self.postings[term]:
                dl = self.doc_lengths[doc_idx]
                norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                scores[doc_idx] += idf_val * ((tf * (self.k1 + 1)) / (tf + norm))
        if K < len(scores):
            top_idxs = np.argpartition(-scores, K)[:K]
        else:
            top_idxs = np.argsort(-scores)
        return [self.doc_ids[i] for i in top_idxs]
        
    def fit(self, corpus):
        logger.info("Building inverted index BM25...")
        self.doc_ids = list(corpus.keys())
        
        term_doc_freq = defaultdict(set)
        
        for doc_idx, doc_id in enumerate(tqdm(self.doc_ids, desc="Indexing documents")):
            tokens = self.tokenize(corpus[doc_id]['text'])
            self.doc_lengths[doc_idx] = len(tokens)
            
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            
            for term, tf in term_freq.items():
                self.postings[term].append((doc_idx, tf))
                term_doc_freq[term].add(doc_idx)
        
        n_docs = len(self.doc_ids)
        self.avgdl = sum(self.doc_lengths.values()) / n_docs
        
        for term, doc_set in term_doc_freq.items():
            df = len(doc_set)
            self.idf[term] = np.log((n_docs - df + 0.5) / (df + 0.5))
        
        logger.info(f"Built inverted index: {n_docs} docs, {len(self.postings)} terms")
        
    def get_negatives(self, queries, qrels, num_negatives=20, top_k=100):
        logger.info("Mining BM25 negatives with inverted index...")
        
        query_negatives = {}
        
        for qid, query_text in tqdm(queries.items(), desc="Processing queries"):
            if qid not in qrels:
                continue
                
            query_tokens = self.tokenize(query_text)
            scores = np.zeros(len(self.doc_ids), dtype=np.float32)
            
            for term in query_tokens:
                if term in self.postings:
                    idf_val = self.idf[term]
                    for doc_idx, tf in self.postings[term]:
                        doc_len = self.doc_lengths[doc_idx]
                        norm_factor = self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                        scores[doc_idx] += idf_val * ((tf * (self.k1 + 1)) / (tf + norm_factor))
            
            positives = set(doc_id for doc_id, rel in qrels[qid].items() if rel > 0)
            positive_indices = set()
            for doc_id in positives:
                try:
                    positive_indices.add(self.doc_ids.index(doc_id))
                except ValueError:
                    continue
            
            if top_k < len(scores):
                top_indices = np.argpartition(-scores, top_k)[:top_k]
            else:
                top_indices = np.argsort(-scores)
            
            negative_doc_ids = []
            for idx in top_indices:
                if idx not in positive_indices:
                    negative_doc_ids.append(self.doc_ids[idx])
                    if len(negative_doc_ids) >= num_negatives:
                        break
            
            query_negatives[qid] = negative_doc_ids
        
        logger.info(f"Mined negatives for {len(query_negatives)} queries")
        return query_negatives

def mine_negatives_and_precompute(corpus, queries, qrels, negatives_path, embeddings_path):
    bm25 = InvertedIndexBM25()
    bm25.fit(corpus)
    query_negatives = bm25.get_negatives(queries, qrels, num_negatives=20, top_k=200)  # More negatives
    
    with open(negatives_path, 'w') as f:
        json.dump(query_negatives, f)
    
    logger.info("Precomputing document embeddings with E5...")
    e5_model = SentenceTransformer('intfloat/e5-large-v2')
    
    required_docs = set()
    for qid, query_rels in qrels.items():
        for doc_id, rel in query_rels.items():
            if rel > 0:
                required_docs.add(doc_id)
    
    for neg_list in query_negatives.values():
        required_docs.update(neg_list)
    
    required_docs = list(required_docs)
    total_docs = len(required_docs)
    logger.info(f"Precomputing E5 embeddings for {total_docs} documents")
    
    embeddings_dict = {}
    batch_size = 64  # Smaller batch for larger model
    
    for i in tqdm(range(0, len(required_docs), batch_size), desc="Encoding documents"):
        batch_ids = required_docs[i:i+batch_size]
        # Add E5 passage prefix
        batch_texts = [f"passage: {corpus[doc_id]['text']}" for doc_id in batch_ids]
        
        with torch.no_grad():
            embeddings = e5_model.encode(batch_texts, convert_to_tensor=True, device='cpu')
        
        for doc_id, emb in zip(batch_ids, embeddings):
            embeddings_dict[doc_id] = emb
    
    torch.save(embeddings_dict, embeddings_path)
    logger.info(f"Saved E5 embeddings to {embeddings_path}")
    
    return query_negatives, embeddings_dict


def precompute_e5_embeddings_only(corpus, queries, qrels, negatives_path, embeddings_path):
    """
    Precompute E5 embeddings using existing BM25 mined negatives.
    """
    # Load existing negatives
    logger.info(f"Loading existing negatives from {negatives_path}")
    with open(negatives_path, 'r') as f:
        query_negatives = json.load(f)
    
    logger.info("Precomputing document embeddings with E5...")
    e5_model = SentenceTransformer('intfloat/e5-large-v2')
    
    required_docs = set()
    for qid, query_rels in qrels.items():
        for doc_id, rel in query_rels.items():
            if rel > 0:
                required_docs.add(doc_id)
    
    for neg_list in query_negatives.values():
        required_docs.update(neg_list)
    
    required_docs = list(required_docs)
    total_docs = len(required_docs)
    logger.info(f"Precomputing E5 embeddings for {total_docs} documents")
    
    embeddings_dict = {}
    batch_size = 256  # Smaller batch for larger model
    
    for i in tqdm(range(0, len(required_docs), batch_size), desc="Encoding documents"):
        batch_ids = required_docs[i:i+batch_size]
        # Add E5 passage prefix
        batch_texts = [f"passage: {corpus[doc_id]['text']}" for doc_id in batch_ids]
        
        with torch.no_grad():
            embeddings = e5_model.encode(batch_texts, convert_to_tensor=True, device='cpu')
        
        for doc_id, emb in zip(batch_ids, embeddings):
            embeddings_dict[doc_id] = emb
    
    torch.save(embeddings_dict, embeddings_path)
    logger.info(f"Saved E5 embeddings to {embeddings_path}")
    
    return query_negatives, embeddings_dict


def offline_preprocessing_with_flag(corpus, queries, qrels, cache_dir, force_load_only=False, negatives_path=None):
    logger.info("Running offline preprocessing...")
    os.makedirs(cache_dir, exist_ok=True)
    
    if negatives_path is None:
        negatives_path = os.path.join(cache_dir, "negatives.json")
    
    embeddings_path = os.path.join(cache_dir, "embeddings.pt")
    
    if force_load_only:
        # Only try to load existing files
        if os.path.exists(negatives_path) and os.path.exists(embeddings_path):
            logger.info(f"Loading existing negatives from {negatives_path}")
            logger.info(f"Loading existing embeddings from {embeddings_path}")
            return negatives_path, embeddings_path
        else:
            raise FileNotFoundError(
                f"force_load_only=True but required files not found:\n"
                f"  Negatives: {negatives_path} (exists: {os.path.exists(negatives_path)})\n"
                f"  Embeddings: {embeddings_path} (exists: {os.path.exists(embeddings_path)})"
            )
    else:
        # Check if we need to compute embeddings only
        if os.path.exists(negatives_path) and not os.path.exists(embeddings_path):
            logger.info("Found existing negatives, computing embeddings only...")
            precompute_e5_embeddings_only(corpus, queries, qrels, negatives_path, embeddings_path)
            return negatives_path, embeddings_path
        elif os.path.exists(negatives_path) and os.path.exists(embeddings_path):
            logger.info("Found existing cache files, loading them...")
            return negatives_path, embeddings_path
        else:
            logger.info("Computing embeddings from scratch (ignoring any existing cache)")
            mine_negatives_and_precompute(corpus, queries, qrels, negatives_path, embeddings_path)
            return negatives_path, embeddings_path

class BiHyperNetIR(nn.Module):
    def __init__(self, output_dim=256, embeddings_cache=None):
        super().__init__()
        self.e5_model = SentenceTransformer('intfloat/e5-large-v2', device=device)
        for param in self.e5_model.parameters():
            param.requires_grad = False
            
        self.e5_projector = nn.Linear(1024, output_dim)  # E5-large has 1024 dim
        
        # Simpler hypernetworks
        self.query_hypernet = HyperNet(output_dim, 128, output_dim)
        self.doc_hypernet = HyperNet(output_dim, 128, output_dim)
        
        # Add layer normalization
        self.query_norm = nn.LayerNorm(output_dim)
        self.doc_norm = nn.LayerNorm(output_dim)
        
        # Store embeddings cache
        self.embeddings_cache = embeddings_cache
    
    def encode_and_project(self, texts, doc_ids=None, is_query=False):
        """
        Encode texts with E5 prefixes, using cache when possible
        """
        # Add E5 prefixes
        if is_query:
            prefixed_texts = [f"query: {text}" for text in texts]
        else:
            prefixed_texts = [f"passage: {text}" for text in texts]
        
        if self.embeddings_cache is not None and doc_ids is not None:
            # Try to use cached embeddings (same logic as before)
            embeddings_list = []
            texts_to_encode = []
            indices_to_encode = []
            
            for i, (text, doc_id) in enumerate(zip(prefixed_texts, doc_ids)):
                if doc_id in self.embeddings_cache:
                    cached_emb = self.embeddings_cache[doc_id].to(device)
                    projected = self.e5_projector(cached_emb.unsqueeze(0))
                    embeddings_list.append(F.normalize(projected, p=2, dim=1))
                else:
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
            
            if texts_to_encode:
                with torch.no_grad():
                    new_embeddings = self.e5_model.encode(texts_to_encode, convert_to_tensor=True, device=device)
                projected = self.e5_projector(new_embeddings)
                normalized = F.normalize(projected, p=2, dim=1)
                
                for idx, emb in zip(indices_to_encode, normalized):
                    embeddings_list.insert(idx, emb.unsqueeze(0))
            
            return torch.cat(embeddings_list, dim=0)
        else:
            with torch.no_grad():
                embeddings = self.e5_model.encode(prefixed_texts, convert_to_tensor=True, device=device)
            projected = self.e5_projector(embeddings)
            return F.normalize(projected, p=2, dim=1)
    
    def forward(self, queries, pos_embs, neg_embs):
        query_embs = self.encode_and_project(queries, is_query=True)
        batch_size = len(queries)
        
        # Project and normalize embeddings
        pos_embs_proj = F.normalize(self.e5_projector(pos_embs.to(device)), p=2, dim=1)
        neg_embs_proj = F.normalize(self.e5_projector(neg_embs.to(device)), p=2, dim=1)
        
        all_doc_embs = torch.cat([pos_embs_proj.unsqueeze(1), neg_embs_proj], dim=1)
        docs_per_query = all_doc_embs.size(1)
        all_doc_embs_flat = all_doc_embs.view(-1, 256)
        
        # Get hypernetwork outputs (scale and bias vectors)
        q_scale, q_bias = self.query_hypernet(query_embs)
        d_scale, d_bias = self.doc_hypernet(all_doc_embs_flat)
        
        # Apply simpler modulations
        modulated_queries = self.query_norm(query_embs * q_scale + q_bias)
        
        # Reshape document modulations
        d_scale_reshaped = d_scale.view(batch_size, docs_per_query, 256)
        d_bias_reshaped = d_bias.view(batch_size, docs_per_query, 256)
        
        # Apply document modulations
        modulated_docs = self.doc_norm(all_doc_embs * d_scale_reshaped + d_bias_reshaped)
        
        return modulated_queries, modulated_docs

class MultipleNegativesRankingLoss(torch.nn.Module):
    """
    Multiple Negatives Ranking Loss as described in:
    https://arxiv.org/abs/1705.00652
    
    This loss treats each positive pair in a batch as the target,
    and all other examples in the batch as negatives.
    """
    def __init__(self, scale: float = 10.0, similarity_fct=None):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct if similarity_fct is not None else self.cos_sim
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def cos_sim(self, a, b):
        """Compute cosine similarity between two tensors."""
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def forward(self, query_embeddings, doc_embeddings):
        """
        Args:
            query_embeddings: [batch_size, embedding_dim]
            doc_embeddings: [batch_size, embedding_dim] - first doc is positive for each query
        """
        scores = self.similarity_fct(query_embeddings, doc_embeddings) * self.scale
        
        # Labels: each query's positive is at the same index
        labels = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        
        return self.cross_entropy_loss(scores, labels)

class BiHyperNetTrainer:
    def __init__(self, model, train_loader, val_loader, total_steps, lr=1e-4, weight_decay=1e-3, warmup_steps=1000):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Initialize MultipleNegativesRankingLoss
        self.loss_fn = MultipleNegativesRankingLoss(scale=10.0)
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
    def train_step_alt(self, batch):
        self.optimizer.zero_grad()
        
        queries = batch['query']
        pos_embs = batch['positive_emb']
        neg_embs = batch['negative_embs']
        
        query_embs, doc_embs = self.model(queries, pos_embs, neg_embs)
        
        # For MultipleNegativesRankingLoss, we need to flatten the embeddings
        # Assuming doc_embs has shape [batch_size, num_docs, embedding_dim]
        # where the first doc is always the positive
        batch_size, num_docs, embedding_dim = doc_embs.shape
        
        # Reshape for MNRL: we'll treat each query-positive pair as a separate example
        # and use all other documents in the batch as negatives
        query_embs_flat = query_embs  # [batch_size, embedding_dim]
        pos_doc_embs = doc_embs[:, 0, :]  # [batch_size, embedding_dim] - first doc is positive
        
        # Create a batch where each positive document is paired with its query
        # This leverages the "multiple negatives" aspect by using other positives as negatives
        loss = self.loss_fn(query_embs_flat, pos_doc_embs)
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                queries = batch['query']
                pos_embs = batch['positive_emb']
                neg_embs = batch['negative_embs']
                
                query_embs, doc_embs = self.model(queries, pos_embs, neg_embs)
                
                # Use the same loss calculation as training for consistency
                batch_size, num_docs, embedding_dim = doc_embs.shape
                query_embs_flat = query_embs
                pos_doc_embs = doc_embs[:, 0, :]
                
                loss = self.loss_fn(query_embs_flat, pos_doc_embs)
                
                total_loss += loss.item()
                count += 1
        
        self.model.train()
        return total_loss / count if count > 0 else 0.0

    def train_step(self, batch):
        """
        Alternative implementation that uses all documents (positive + negatives)
        This creates more negatives per query by reshaping the batch
        """
        self.optimizer.zero_grad()
        
        queries = batch['query']
        pos_embs = batch['positive_emb']
        neg_embs = batch['negative_embs']
        
        query_embs, doc_embs = self.model(queries, pos_embs, neg_embs)
        
        # Reshape to create more training examples
        batch_size, num_docs, embedding_dim = doc_embs.shape
        
        # For each query, we'll compute similarity with ALL documents in the batch
        # and treat only its corresponding positive (at index 0) as the target
        
        # Flatten documents from all queries to create a large negative pool
        all_docs = doc_embs.view(-1, embedding_dim)  # [batch_size * num_docs, embedding_dim]
        
        # For each query, compute similarity with all documents
        scores_list = []
        labels_list = []
        
        for i in range(batch_size):
            query_emb = query_embs[i:i+1]  # [1, embedding_dim]
            
            # Compute similarity with all documents
            scores = self.loss_fn.similarity_fct(query_emb, all_docs) * self.loss_fn.scale  # [1, batch_size * num_docs]
            scores_list.append(scores)
            
            # The positive document for query i is at position i * num_docs (first doc of query i)
            positive_idx = i * num_docs
            labels_list.append(torch.tensor([positive_idx], device=query_embs.device))
        
        # Concatenate all scores and labels
        all_scores = torch.cat(scores_list, dim=0)  # [batch_size, batch_size * num_docs]
        all_labels = torch.cat(labels_list, dim=0)  # [batch_size]
        
        loss = F.cross_entropy(all_scores, all_labels)
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()

# Fix 2: Better negative sampling
class MSMarcoContrastiveDataset(Dataset):
    def __init__(self, queries, corpus, qrels, negatives_path, embeddings_path, num_negatives=20):
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels
        self.samples = []
        self.num_negatives = num_negatives
        
        # Debug: Print initial data sizes
        print(f"Total queries: {len(queries)}")
        print(f"Total corpus: {len(corpus)}")
        print(f"Total qrels: {len(qrels)}")
        
        with open(negatives_path, 'r') as f:
            query_negatives = json.load(f)
        print(f"Total query negatives: {len(query_negatives)}")
        
        self.embeddings_cache = torch.load(embeddings_path, map_location='cpu')
        print(f"Total embeddings cached: {len(self.embeddings_cache)}")
        
        # Debug: Check overlap between different data sources
        queries_with_qrels = set(queries.keys()) & set(qrels.keys())
        queries_with_negatives = set(queries.keys()) & set(query_negatives.keys())
        queries_with_both = queries_with_qrels & queries_with_negatives
        
        print(f"Queries with qrels: {len(queries_with_qrels)}")
        print(f"Queries with negatives: {len(queries_with_negatives)}")
        print(f"Queries with both qrels and negatives: {len(queries_with_both)}")
        
        # Sample a few query IDs to check format consistency
        sample_query_ids = list(queries.keys())[:3]
        sample_qrel_ids = list(qrels.keys())[:3]
        sample_neg_ids = list(query_negatives.keys())[:3]
        sample_embed_ids = list(self.embeddings_cache.keys())[:3]
        
        print(f"Sample query IDs: {sample_query_ids}")
        print(f"Sample qrel IDs: {sample_qrel_ids}")
        print(f"Sample negative IDs: {sample_neg_ids}")
        print(f"Sample embedding IDs: {sample_embed_ids}")
        
        # Check data types
        print(f"Query ID type: {type(sample_query_ids[0]) if sample_query_ids else 'None'}")
        print(f"Qrel ID type: {type(sample_qrel_ids[0]) if sample_qrel_ids else 'None'}")
        print(f"Negative ID type: {type(sample_neg_ids[0]) if sample_neg_ids else 'None'}")
        print(f"Embedding ID type: {type(sample_embed_ids[0]) if sample_embed_ids else 'None'}")
        
        # Create training samples with detailed debugging
        successful_samples = 0
        failed_no_positives = 0
        failed_no_negatives = 0
        failed_no_embeddings = 0
        
        for qid, query_text in queries.items():
            if qid in qrels and qid in query_negatives:
                # Check positives
                positives = [doc_id for doc_id, rel in qrels[qid].items() if rel > 0]
                if not positives:
                    failed_no_positives += 1
                    continue
                
                # Check negatives
                available_negatives = [doc_id for doc_id in query_negatives[qid] 
                                     if doc_id in self.embeddings_cache]
                if len(available_negatives) < self.num_negatives:
                    failed_no_negatives += 1
                    continue
                
                # Check embeddings for positives
                valid_positives = [pos_id for pos_id in positives if pos_id in self.embeddings_cache]
                if not valid_positives:
                    failed_no_embeddings += 1
                    continue
                
                # Create samples
                for pos_id in valid_positives:
                    selected_negatives = random.sample(available_negatives, self.num_negatives)
                    
                    sample = {
                        'query': query_text,
                        'positive_doc_id': pos_id,
                        'negative_doc_ids': selected_negatives
                    }
                    self.samples.append(sample)
                    successful_samples += 1
        
        print(f"\nSample creation summary:")
        print(f"Successful samples: {successful_samples}")
        print(f"Failed - no positives: {failed_no_positives}")
        print(f"Failed - insufficient negatives: {failed_no_negatives}")
        print(f"Failed - no embeddings for positives: {failed_no_embeddings}")
        
        logger.info(f"Created {len(self.samples)} contrastive training samples")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        pos_emb = self.embeddings_cache[sample['positive_doc_id']]
        
        # Resample negatives each epoch for more diversity
        available_negatives = [doc_id for doc_id in sample['negative_doc_ids'] 
                             if doc_id in self.embeddings_cache]
        
        if len(available_negatives) >= self.num_negatives:
            selected_negatives = random.sample(available_negatives, self.num_negatives)
        else:
            selected_negatives = available_negatives
        
        neg_embs = [self.embeddings_cache[doc_id] for doc_id in selected_negatives]
        
        if len(neg_embs) == 0:
            neg_embs = [pos_emb]  # Fallback
        
        return {
            'query': sample['query'],
            'positive_emb': pos_emb,
            'negative_embs': torch.stack(neg_embs)
        }

# Fix 3: Regularized HyperNet
class HyperNet(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.hypernet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim * 2),  # scale + bias
        )
        
        # Initialize with small weights to prevent large modifications initially
        for layer in self.hypernet:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)
        
    def forward(self, emb):
        hyper_output = self.hypernet(emb)
        
        scale = hyper_output[:, :self.output_dim]
        bias = hyper_output[:, self.output_dim:]
        
        
        scale = torch.sigmoid(scale) 
        bias  = torch.tanh(bias)
        print("Scale", scale, "Bias", bias) 
        
        return scale, bias

class FAISSRetriever:
    def __init__(self, model, corpus, embeddings_path=None, index_path=None):
        self.embeddings_path = embeddings_path
        self.index_path = index_path  # Path to save/load FAISS index
        self.model = model
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.index = None
        self.doc_embeddings = None
        self._build_or_load_index()
    
    def _build_or_load_index(self):
        """Build index from embeddings or load from disk"""
        
        # Try to load existing index first
        if self.index_path and os.path.exists(self.index_path):
            doc_ids_path = self.index_path.replace('.faiss', '_doc_ids.json')
            
            if os.path.exists(doc_ids_path):
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                try:
                    # Load the index
                    self.index = faiss.read_index(self.index_path)
                    
                    # Load corresponding doc_ids
                    with open(doc_ids_path, 'r') as f:
                        self.doc_ids = json.load(f)
                    
                    logger.info(f"Loaded FAISS index with {len(self.doc_ids)} documents")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load existing index: {e}. Building new index...")
        
        # Build new index if loading failed or no index exists
        self._build_index()
        
        # Save the newly built index
        if self.index_path:
            self._save_index()
    
    def _build_index(self):
        """Build FAISS index from embeddings"""
        logger.info("Building FAISS index")
        
        # Set model to eval mode and disable gradients
        self.model.eval()
        
        with torch.no_grad():
            if self.embeddings_path and os.path.exists(self.embeddings_path):
                logger.info("Using precomputed embeddings for FAISS index")
                embeddings_cache = torch.load(self.embeddings_path, map_location='cpu')
                
                embeddings_list = []
                total_docs = len(self.doc_ids)
                processed_docs = 0
                missing_docs = []
                valid_doc_ids = []
                
                for doc_id in self.doc_ids:
                    if doc_id in embeddings_cache:
                        # Project cached embeddings
                        cached_emb = embeddings_cache[doc_id].to(device)
                        emb = self.model.sbert_projector(cached_emb.unsqueeze(0))
                        embeddings_list.append(emb.cpu().numpy())
                        valid_doc_ids.append(doc_id)
                    else:
                        missing_docs.append(doc_id)
                    
                    processed_docs += 1
                    if processed_docs % 1000 == 0:
                        completion_percentage = (processed_docs / total_docs) * 100
                        logger.info(f"FAISS index building progress: {processed_docs}/{total_docs} ({completion_percentage:.1f}%)")
                
                if missing_docs:
                    logger.warning(f"Missing embeddings for {len(missing_docs)} documents")
                
                # Update doc_ids to only include valid documents
                self.doc_ids = valid_doc_ids
                self.doc_embeddings = np.vstack(embeddings_list)
                logger.info(f"FAISS index building completed: {len(self.doc_ids)} documents")
            else:
                raise FileNotFoundError(
                    f"No precomputed embeddings found at {self.embeddings_path}"
                )

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.doc_embeddings)    
        self.index = faiss.IndexFlatIP(self.doc_embeddings.shape[1])
        self.index.add(self.doc_embeddings)
        logger.info(f"FAISS index built with {len(self.doc_ids)} documents")
    
    def _save_index(self):
        """Save FAISS index and doc_ids to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save corresponding doc_ids
            doc_ids_path = self.index_path.replace('.faiss', '_doc_ids.json')
            with open(doc_ids_path, 'w') as f:
                json.dump(self.doc_ids, f)
            
            logger.info(f"Saved FAISS index to {self.index_path}")
            logger.info(f"Saved doc_ids to {doc_ids_path}")
            
            # Log file sizes
            index_size_mb = os.path.getsize(self.index_path) / (1024 * 1024)
            logger.info(f"FAISS index file size: {index_size_mb:.1f} MB")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def retrieve(self, query, k=20):
        """Retrieve top-k documents for a query using E5"""
        self.model.eval()
        with torch.no_grad():
            query_emb = self.model.encode_and_project([query], is_query=True)  # Add is_query flag
            query_emb_norm = F.normalize(query_emb, p=2, dim=1)
            
            scores, indices = self.index.search(query_emb_norm.cpu().numpy(), k)
            
            results = []
            for i, score in zip(indices[0], scores[0]):
                results.append((self.doc_ids[i], float(score)))
            
            return results

# 2. Utility functions for FAISS index management
def get_faiss_index_path(cache_dir, split_name=''):
    """Generate consistent FAISS index path"""
    if split_name:
        return os.path.join(cache_dir, f'faiss_index_{split_name}.faiss')
    else:
        return os.path.join(cache_dir, 'faiss_index.faiss')

def check_faiss_index_exists(index_path):
    """Check if FAISS index and doc_ids file exist"""
    doc_ids_path = index_path.replace('.faiss', '_doc_ids.json')
    
    index_exists = os.path.exists(index_path)
    doc_ids_exists = os.path.exists(doc_ids_path)
    
    if index_exists and doc_ids_exists:
        # Check file sizes
        index_size_mb = os.path.getsize(index_path) / (1024 * 1024)
        
        try:
            with open(doc_ids_path, 'r') as f:
                doc_ids = json.load(f)
            num_docs = len(doc_ids)
            
            print(f"FAISS index: {index_path}")
            print(f"  Index file: {'✓' if index_exists else '✗'} ({index_size_mb:.1f} MB)")
            print(f"  Doc IDs file: {'✓' if doc_ids_exists else '✗'} ({num_docs} documents)")
            return True
        except Exception as e:
            print(f"Error checking FAISS index: {e}")
            return False
    else:
        print(f"FAISS index: {index_path}")
        print(f"  Index file: {'✓' if index_exists else '✗'}")
        print(f"  Doc IDs file: {'✓' if doc_ids_exists else '✗'}")
        return False

def clean_faiss_cache(cache_dir):
    """Remove all FAISS index files from cache directory"""
    faiss_files = []
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file.endswith('.faiss') or file.endswith('_doc_ids.json'):
                faiss_files.append(os.path.join(root, file))
    
    for file_path in faiss_files:
        try:
            os.remove(file_path)
            logger.info(f"Removed {file_path}")
        except Exception as e:
            logger.error(f"Failed to remove {file_path}: {e}")
    
    logger.info(f"Cleaned {len(faiss_files)} FAISS cache files")


# Also fix the MRREvaluator to handle gradients properly
class MRREvaluator:
    def __init__(
        self,
        model,
        corpus: Dict[str, Any],
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        embeddings_path: str,
        num_candidates: int = 1000,
    ):
        # enforce that you really passed in a cache
        if embeddings_path is None:
            raise ValueError("`embeddings_path` must be provided to MRREvaluator")
        self.model = model
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.num_candidates = num_candidates

        # load once, up front
        self.doc_embeddings_cache = torch.load(embeddings_path, map_location="cpu")
        self.cached_doc_ids = list(self.doc_embeddings_cache.keys())
        
        # if embeddings_path and os.path.exists(embeddings_path):
        #     logger.info("Loading precomputed embeddings for MRR evaluation...")
        #     self.doc_embeddings_cache = torch.load(embeddings_path, map_location='cpu')
        #     self.cached_doc_ids = list(self.doc_embeddings_cache.keys())
        #     logger.info(f"Loaded {len(self.cached_doc_ids)} cached document embeddings")

    def evaluate_mrr(self, k=20, use_hypernetworks=True, max_queries=None):
        """Evaluate MRR with proper candidate selection"""
        self.model.eval()
        
        mrr_scores = []
        evaluated_queries = 0
        
        with torch.no_grad():
            for qid, query_text in tqdm(self.queries.items(), desc="Evaluating MRR"):
                if qid not in self.qrels or (max_queries is not None and evaluated_queries >= max_queries):
                    continue
                
                relevant_docs = {doc_id for doc_id, rel in self.qrels[qid].items() if rel > 0}
                if not relevant_docs:
                    continue
                
                # Better candidate selection: use random sample of all cached docs
                if self.cached_doc_ids:
                    # Ensure we include at least some relevant docs
                    candidates = list(relevant_docs & set(self.cached_doc_ids))
                    
                    # Add random negative candidates
                    remaining_docs = [doc_id for doc_id in self.cached_doc_ids 
                                    if doc_id not in relevant_docs]
                    
                    if len(remaining_docs) > 0:
                        num_negatives = min(self.num_candidates - len(candidates), len(remaining_docs))
                        negative_sample = random.sample(remaining_docs, num_negatives)
                        candidates.extend(negative_sample)
                    
                    if len(candidates) == 0:
                        continue
                        
                    # Shuffle to randomize order
                    random.shuffle(candidates)
                else:
                    # Fallback to using qrels if no cache
                    candidates = list(self.qrels[qid].keys())[:self.num_candidates]
                
                if len(candidates) == 0:
                    mrr_scores.append(0.0)
                    evaluated_queries += 1
                    continue
                
                # Get candidate embeddings
                if self.doc_embeddings_cache:
                    cand_embs = torch.stack([self.doc_embeddings_cache[doc_id] 
                                        for doc_id in candidates if doc_id in self.doc_embeddings_cache]).to(device)
                    candidates = [doc_id for doc_id in candidates if doc_id in self.doc_embeddings_cache]
                else:
                    cand_texts = [f"passage: {self.corpus[doc_id]['text']}" for doc_id in candidates]  # Add E5 prefix
                    cand_embs = self.model.e5_model.encode(cand_texts, convert_to_tensor=True, device=device)
                
                if len(cand_embs) == 0:
                    mrr_scores.append(0.0)
                    evaluated_queries += 1
                    continue
                
                # Project candidate embeddings using e5_projector
                cand_embs_proj = F.normalize(self.model.e5_projector(cand_embs), p=2, dim=1)
                
                # Get query embedding and project it
                query_text_prefixed = f"query: {query_text}"  # Add E5 prefix
                query_emb = self.model.e5_model.encode([query_text_prefixed], convert_to_tensor=True, device=device)
                query_emb_proj = F.normalize(self.model.e5_projector(query_emb), p=2, dim=1)
                
                if use_hypernetworks:
                    # Apply hypernetworks
                    q_scale, q_bias = self.model.query_hypernet(query_emb_proj)
                    d_scale, d_bias = self.model.doc_hypernet(cand_embs_proj)
                    
                    # Apply modulations
                    modulated_query = self.model.query_norm(query_emb_proj * q_scale + q_bias)
                    modulated_docs = self.model.doc_norm(cand_embs_proj * d_scale + d_bias)
                    
                    # Normalize after modulation
                    modulated_query = F.normalize(modulated_query, p=2, dim=1)
                    modulated_docs = F.normalize(modulated_docs, p=2, dim=1)
                    
                    # Compute similarities
                    similarities = torch.mm(modulated_query, modulated_docs.transpose(0, 1)).squeeze(0)
                else:
                    # Simple cosine similarity without hypernetworks
                    similarities = torch.mm(query_emb_proj, cand_embs_proj.transpose(0, 1)).squeeze(0)
                
                # Get top-k results
                _, top_idxs = torch.topk(similarities, min(k, len(candidates)))
                
                # Calculate MRR for this query
                for rank, idx in enumerate(top_idxs, 1):
                    if candidates[idx] in relevant_docs:
                        mrr_scores.append(1.0 / rank)
                        break
                else:
                    mrr_scores.append(0.0)
                
                evaluated_queries += 1
        
        self.model.train()
        logger.info(f"Evaluated MRR on {evaluated_queries} queries")
        return sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

# Fix 3: Add debugging function to check data quality
def debug_evaluation_data(queries, qrels, corpus, sample_size=10):
    """Debug function to check data quality"""
    logger.info("=== DEBUGGING EVALUATION DATA ===")
    
    sample_queries = list(queries.items())[:sample_size]
    
    for qid, query_text in sample_queries:
        logger.info(f"\nQuery ID: {qid}")
        logger.info(f"Query: {query_text[:100]}...")
        
        if qid in qrels:
            relevant_docs = {doc_id: rel for doc_id, rel in qrels[qid].items() if rel > 0}
            logger.info(f"Relevant docs: {len(relevant_docs)}")
            
            # Check if relevant docs exist in corpus
            existing_relevant = {doc_id: rel for doc_id, rel in relevant_docs.items() 
                               if doc_id in corpus}
            logger.info(f"Relevant docs in corpus: {len(existing_relevant)}")
            
            if existing_relevant:
                sample_doc_id = list(existing_relevant.keys())[0]
                sample_doc = corpus[sample_doc_id]['text']
                logger.info(f"Sample relevant doc: {sample_doc[:100]}...")
        else:
            logger.info("No qrels found for this query")
    
    logger.info("=== END DEBUG ===")


# Additional debugging function to check baseline performance
def evaluate_baseline_sbert(corpus, queries, qrels, k=20):
    """Evaluate baseline SBERT performance without any modifications"""
    logger.info("Evaluating baseline SBERT performance...")
    
    sbert_model = SentenceTransformer('intfloat/e5-large-v2', device=device)
    mrr_scores = []
    
    with torch.no_grad():
        for qid, query_text in tqdm(queries.items(), desc="Baseline evaluation"):
            if qid not in qrels:
                continue
            
            relevant_docs = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
            all_docs = list(qrels[qid].keys())[:k*5]
            
            if len(all_docs) == 0:
                mrr_scores.append(0.0)
                continue
            
            doc_texts = [corpus[doc_id]['text'] for doc_id in all_docs]
            
            # Encode with baseline SBERT
            query_emb = sbert_model.encode([query_text], convert_to_tensor=True, device=device)
            doc_embs = sbert_model.encode(doc_texts, convert_to_tensor=True, device=device)
            
            # Normalize and compute similarities
            query_emb = F.normalize(query_emb, p=2, dim=1)
            doc_embs = F.normalize(doc_embs, p=2, dim=1)
            
            similarities = torch.mm(query_emb, doc_embs.transpose(0, 1)).squeeze(0)
            
            # Get top-k results
            _, top_idxs = torch.topk(similarities, min(k, len(all_docs)))
            
            # Calculate MRR for this query
            for rank, idx in enumerate(top_idxs, 1):
                if all_docs[idx] in relevant_docs:
                    mrr_scores.append(1.0 / rank)
                    break
            else:
                mrr_scores.append(0.0)
    
    baseline_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    logger.info(f"Baseline SBERT MRR@{k}: {baseline_mrr:.4f}")
    return baseline_mrr

def collate_fn(batch):
    queries = [item['query'] for item in batch]
    pos_embs = torch.stack([item['positive_emb'] for item in batch])
    neg_embs = torch.stack([item['negative_embs'] for item in batch])
    
    return {
        'query': queries,
        'positive_emb': pos_embs,
        'negative_embs': neg_embs
    }
def check_embedding_files(cache_dirs):
    """Check if all required embedding files exist"""
    all_exist = True
    for split_name, cache_dir in cache_dirs.items():
        negatives_path = os.path.join(cache_dir, "negatives.json")
        embeddings_path = os.path.join(cache_dir, "embeddings.pt")
        
        neg_exists = os.path.exists(negatives_path)
        emb_exists = os.path.exists(embeddings_path)
        
        print(f"{split_name}:")
        print(f"  Negatives: {negatives_path} - {'✓' if neg_exists else '✗'}")
        print(f"  Embeddings: {embeddings_path} - {'✓' if emb_exists else '✗'}")
        
        if emb_exists:
            # Check file size
            size_mb = os.path.getsize(embeddings_path) / (1024 * 1024)
            print(f"  Embeddings size: {size_mb:.1f} MB")
            
            # Check number of embeddings
            try:
                emb_dict = torch.load(embeddings_path, map_location='cpu')
                print(f"  Number of embeddings: {len(emb_dict)}")
            except Exception as e:
                print(f"  Error loading embeddings: {e}")
                all_exist = False
        
        if not (neg_exists and emb_exists):
            all_exist = False
        print()
    
    return all_exist

# Example usage in main:
def main_with_checks():
    cfg = {
        'cache_dir': 'models/cache_fresh_run',
        # ... other config
    }
    
    # Check if all embedding files exist before starting
    cache_dirs = {
        'train': os.path.join(cfg['cache_dir'], 'train'),
        'val': os.path.join(cfg['cache_dir'], 'val'),
        'test': os.path.join(cfg['cache_dir'], 'test')
    }
    
    if not check_embedding_files(cache_dirs):
        logger.error("Missing embedding files. Please compute embeddings first.")
        return

def inspect_faiss_index(index_path):
    """Inspect a saved FAISS index"""
    if not os.path.exists(index_path):
        print(f"Index file not found: {index_path}")
        return
    
    try:
        index = faiss.read_index(index_path)
        doc_ids_path = index_path.replace('.faiss', '_doc_ids.json')
        
        print(f"FAISS Index: {index_path}")
        print(f"  Type: {type(index).__name__}")
        print(f"  Dimension: {index.d}")
        print(f"  Number of vectors: {index.ntotal}")
        print(f"  Is trained: {index.is_trained}")
        
        if os.path.exists(doc_ids_path):
            with open(doc_ids_path, 'r') as f:
                doc_ids = json.load(f)
            print(f"  Document IDs: {len(doc_ids)} loaded")
        else:
            print(f"  Document IDs: file not found")
            
        # File size
        size_mb = os.path.getsize(index_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"Error inspecting index: {e}")

def main():
    cfg = {
        'max_queries': 50000,
        'batch_size': 16,
        'epochs': 100,
        'lr': 1e-5,
        'weight_decay': 1e-4,
        'patience': 25,
        'save_path': 'models/bihypernet_e5_trec-covid.pt',
        'cache_dir': 'models/cache_e5_trec-covid',
        'load_embeddings_only': False,
        'use_faiss_cache': False 
    }
    
    os.makedirs(cfg['cache_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(cfg['save_path']), exist_ok=True)
    torch.cuda.empty_cache()
    
    loader = MSMARCODatasetLoader()
    splits = loader.load_dataset(cfg['max_queries'])
    
    train_corpus, train_queries, train_qrels = splits['train']
    val_corpus, val_queries, val_qrels = splits['val']
    test_corpus, test_queries, test_qrels = splits['test']
    print(len(train_queries), len(val_queries), len(test_queries))

    # Use the modified function with load-only flag
    if cfg['load_embeddings_only']:
        train_negatives_path, train_embeddings_path = offline_preprocessing_with_flag(
            train_corpus, train_queries, train_qrels, os.path.join(cfg['cache_dir'], 'train'), force_load_only=True
        )
        val_negatives_path, val_embeddings_path = offline_preprocessing_with_flag(
            val_corpus, val_queries, val_qrels, os.path.join(cfg['cache_dir'], 'val'), force_load_only=True
        )
        test_negatives_path, test_embeddings_path = offline_preprocessing_with_flag(
            test_corpus, test_queries, test_qrels, os.path.join(cfg['cache_dir'], 'test'), force_load_only=True
        )
    else:
        train_negatives_path, train_embeddings_path = offline_preprocessing_with_flag(
            train_corpus, train_queries, train_qrels, os.path.join(cfg['cache_dir'], 'train'), force_load_only=False
        )
        val_negatives_path, val_embeddings_path = offline_preprocessing_with_flag(
            val_corpus, val_queries, val_qrels, os.path.join(cfg['cache_dir'], 'val'), force_load_only=False
        )
        test_negatives_path, test_embeddings_path = offline_preprocessing_with_flag(
            test_corpus, test_queries, test_qrels, os.path.join(cfg['cache_dir'], 'test'), force_load_only=False
        )
    
    # Load embeddings cache for the model
    train_embeddings_cache = torch.load(train_embeddings_path, map_location='cpu')
    
    # Initialize model
    model = BiHyperNetIR(embeddings_cache=train_embeddings_cache).to(device)
    
    # Load pretrained weights
    # if os.path.exists(cfg['pretrained_path']):
    #     logger.info(f"Loading pretrained model from {cfg['pretrained_path']}")
    #     try:
    #         pretrained_state_dict = torch.load(cfg['pretrained_path'], map_location=device)
    #         model.load_state_dict(pretrained_state_dict)
    #         logger.info("Successfully loaded pretrained model weights")
    #     except Exception as e:
    #         logger.warning(f"Failed to load pretrained model: {e}")
    #         logger.info("Starting training from scratch")
    # else:
    #     logger.warning(f"Pretrained model not found at {cfg['pretrained_path']}")
    #     logger.info("Starting training from scratch")
    
    train_dataset = MSMarcoContrastiveDataset(
        train_queries, train_corpus, train_qrels, train_negatives_path, train_embeddings_path, num_negatives=20
    )
    val_dataset = MSMarcoContrastiveDataset(
        val_queries, val_corpus, val_qrels, val_negatives_path, val_embeddings_path, num_negatives=20
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    
    total_steps = len(train_loader) * cfg['epochs']
    trainer = BiHyperNetTrainer(model, train_loader, val_loader, total_steps, cfg['lr'], cfg['weight_decay'])
    debug_evaluation_data(test_queries, test_qrels, test_corpus)
    mrr_evaluator = MRREvaluator(
        model,
        test_corpus,
        test_queries,
        test_qrels,
        embeddings_path=test_embeddings_path,
        num_candidates=50000
    )
    
    best_mrr = 0 # Initialize to infinity for loss minimization
    patience_counter = 0
    diff_mrr = 0
    for epoch in range(1, cfg['epochs'] + 1):
        logger.info(f"Starting epoch {epoch}")
        
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            loss = trainer.train_step(batch)
            train_losses.append(loss)
            # REMOVED: MRR evaluation from here
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        val_loss = trainer.validate()
        
        
        logger.info("Evaluating MRR...")
        baseline_mrr = mrr_evaluator.evaluate_mrr(k=20, use_hypernetworks=False)
        hypernet_mrr = mrr_evaluator.evaluate_mrr(k=20, use_hypernetworks=True)
        logger.info(f"  Baseline MRR@10: {baseline_mrr:.4f}")
        logger.info(f"  HyperNet MRR@10: {hypernet_mrr:.4f}")
        
        logger.info(f"Epoch {epoch} Results:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        logger.info(f"Gradient norm: {total_norm:.4f}")

        if hypernet_mrr > baseline_mrr:
            diff_mrr = hypernet_mrr - baseline_mrr
        
        # Fixed early stopping logic: save model when validation loss improves (decreases)
        if diff_mrr > best_mrr:
            best_mrr = diff_mrr
            patience_counter = 0
            torch.save(model.state_dict(), cfg['save_path'])
            logger.info("Saved best model based on validation loss")
        else:
            patience_counter += 1
        
        if patience_counter >= cfg['patience']:
            logger.info("Early stopping triggered")
            break
    
    logger.info(f"Training completed. Best validation loss: {best_mrr:.4f}")
    
    def compute_regularization_loss(model):
        reg_loss = 0
        for name, param in model.named_parameters():
            if 'hypernet' in name and param.requires_grad:
                reg_loss += torch.norm(param, p=2)
        return 0.01 * reg_loss  # Small regularization coefficient
    
    # Modified training step with regularization
    def train_step_with_reg(trainer, batch):
        loss = trainer.train_step(batch)
        reg_loss = compute_regularization_loss(trainer.model)
        total_loss = loss + reg_loss
        return total_loss.item()

    logger.info("Building FAISS retriever for final evaluation")
    
    # Define FAISS index path
    test_faiss_index_path = get_faiss_index_path(
        os.path.join(cfg['cache_dir'], 'test'), 'test'
    )
    
    # Check if index already exists
    if cfg['use_faiss_cache'] and check_faiss_index_exists(test_faiss_index_path):
        logger.info("Using existing FAISS index")
    else:
        logger.info("Building new FAISS index")
    
    faiss_retriever = FAISSRetriever(
        model,
        test_corpus,
        embeddings_path=test_embeddings_path,
        index_path=test_faiss_index_path if cfg['use_faiss_cache'] else None
    )
    
    # Test retrieval
    sample_query = list(test_queries.values())[0]
    results = faiss_retriever.retrieve(sample_query, k=20)
    print("Query", sample_query, "Docs", results)
    logger.info(f"Sample retrieval results: {len(results)} documents")


if __name__ == '__main__':
    main()