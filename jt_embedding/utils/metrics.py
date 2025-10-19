from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr


def compute_similarity_matrix(
    embeddings1: torch.Tensor, embeddings2: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity matrix between two sets of embeddings"""
    return F.cosine_similarity(
        x1=embeddings1.unsqueeze(1), x2=embeddings2.unsqueeze(0), dim=2
    )


def retrieval_metrics(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Compute retrieval metrics (Recall@K, MRR)"""
    similarity_matrix = compute_similarity_matrix(embeddings1, embeddings2)

    metrics = {}

    # Forward retrieval (embeddings1 -> embeddings2)
    for k in k_values:
        recall_at_k = recall_at_k_metric(similarity_matrix, k)
        metrics[f"recall@{k}_forward"] = recall_at_k

    # Backward retrieval (embeddings2 -> embeddings1)
    for k in k_values:
        recall_at_k = recall_at_k_metric(similarity_matrix.t(), k)
        metrics[f"recall@{k}_backward"] = recall_at_k

    # Mean Reciprocal Rank
    mrr_forward = mean_reciprocal_rank(similarity_matrix)
    mrr_backward = mean_reciprocal_rank(similarity_matrix.t())

    metrics["mrr_forward"] = mrr_forward
    metrics["mrr_backward"] = mrr_backward
    metrics["mrr_mean"] = (mrr_forward + mrr_backward) / 2

    return metrics


def recall_at_k_metric(similarity_matrix: torch.Tensor, k: int) -> float:
    """Compute Recall@K metric"""
    batch_size = similarity_matrix.size(0)

    # Get top-k indices for each query
    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)

    # Check if correct answer is in top-k
    correct_indices = torch.arange(batch_size).unsqueeze(1).to(similarity_matrix.device)
    hits = torch.any(top_k_indices == correct_indices, dim=1)

    return hits.float().mean().item()


def mean_reciprocal_rank(similarity_matrix: torch.Tensor) -> float:
    """Compute Mean Reciprocal Rank"""
    batch_size = similarity_matrix.size(0)

    # Get ranking of correct answers
    _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)

    # Find rank of correct answer (1-indexed)
    correct_indices = torch.arange(batch_size).to(similarity_matrix.device)
    ranks = torch.zeros(batch_size).to(similarity_matrix.device)

    for i in range(batch_size):
        rank = torch.where(sorted_indices[i] == correct_indices[i])[0][0] + 1
        ranks[i] = 1.0 / rank

    return ranks.mean().item()


def alignment_uniformity_metrics(
    embeddings1: torch.Tensor, embeddings2: torch.Tensor
) -> Dict[str, float]:
    """
    Compute alignment and uniformity metrics
    Reference: Wang & Isola "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere" (2020)
    """
    # Alignment: how well positive pairs are aligned
    alignment = F.mse_loss(embeddings1, embeddings2)

    # Uniformity: how uniformly distributed embeddings are
    def uniformity_metric(embeddings):
        n = embeddings.size(0)
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity = torch.matmul(embeddings_norm, embeddings_norm.t())
        # Exclude diagonal
        mask = torch.eye(n, dtype=torch.bool).to(embeddings.device)
        similarity = similarity.masked_fill(mask, 0)
        return torch.pdist(embeddings_norm, p=2).pow(2).mul(-2).exp().mean().log()

    uniformity1 = uniformity_metric(embeddings1)
    uniformity2 = uniformity_metric(embeddings2)

    return {
        "alignment": alignment.item(),
        "uniformity1": uniformity1.item(),
        "uniformity2": uniformity2.item(),
        "uniformity_mean": (uniformity1 + uniformity2).item() / 2,
    }


def correlation_metrics(
    embeddings1: torch.Tensor, embeddings2: torch.Tensor
) -> Dict[str, float]:
    """Compute correlation metrics between embeddings"""
    # Convert to numpy
    emb1_np = embeddings1.cpu().numpy()
    emb2_np = embeddings2.cpu().numpy()

    # Compute correlations for each dimension
    correlations = []
    for i in range(emb1_np.shape[1]):
        corr, _ = pearsonr(emb1_np[:, i], emb2_np[:, i])
        if not np.isnan(corr):
            correlations.append(corr)

    # Compute overall correlation using flattened embeddings
    emb1_flat = emb1_np.flatten()
    emb2_flat = emb2_np.flatten()

    pearson_corr, _ = pearsonr(emb1_flat, emb2_flat)
    spearman_corr, _ = spearmanr(emb1_flat, emb2_flat)

    return {
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
        "mean_dim_correlation": np.mean(correlations) if correlations else 0.0,
        "std_dim_correlation": np.std(correlations) if correlations else 0.0,
    }


def evaluate_model(model, dataloader, device="cuda") -> Dict[str, float]:
    """Comprehensive model evaluation"""
    model.eval()
    all_fl_embeddings = []
    all_nl_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            fl_emb = model.encode_formal(
                batch["formal_input_ids"], batch["formal_attention_mask"]
            )
            nl_emb = model.encode_informal(
                batch["informal_input_ids"], batch["informal_attention_mask"]
            )

            all_fl_embeddings.append(fl_emb)
            all_nl_embeddings.append(nl_emb)

    # Concatenate all embeddings
    fl_embeddings = torch.cat(all_fl_embeddings, dim=0)
    nl_embeddings = torch.cat(all_nl_embeddings, dim=0)

    # Compute metrics
    metrics = {}

    # Retrieval metrics
    retrieval_results = retrieval_metrics(fl_embeddings, nl_embeddings)
    metrics.update(retrieval_results)

    # Alignment and uniformity
    alignment_results = alignment_uniformity_metrics(fl_embeddings, nl_embeddings)
    metrics.update(alignment_results)

    # Correlation metrics
    correlation_results = correlation_metrics(fl_embeddings, nl_embeddings)
    metrics.update(correlation_results)

    return metrics
