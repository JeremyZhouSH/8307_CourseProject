"""
Three-Layer MI Alignment Losses for Biomedical Summarization.

Node Layer   : entity-type-level InfoNCE + missing-entity penalty
Link Layer   : co-occurrence TransE geometric constraint
Network Layer: spectral graph embedding aligned to decoder output
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss using in-batch negatives.

    Parameters
    ----------
    anchor : [B, D]
    positive : [B, D]
    temperature : float

    Returns
    -------
    loss : scalar Tensor
    """
    if anchor.numel() == 0 or positive.numel() == 0:
        return torch.tensor(0.0, device=anchor.device, dtype=anchor.dtype)

    B = anchor.size(0)
    if B == 1:
        # With only one sample, fall back to cosine distance.
        return (1.0 - F.cosine_similarity(anchor, positive, dim=-1)).mean()

    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)

    # Similarity matrix [B, B]; diagonal = positive pairs.
    sim = torch.matmul(anchor, positive.t()) / temperature
    labels = torch.arange(B, device=anchor.device)

    loss_a = F.cross_entropy(sim, labels)
    loss_b = F.cross_entropy(sim.t(), labels)
    return (loss_a + loss_b) / 2.0


def spectral_embedding(
    adjacency: torch.Tensor,
    k: int = 8,
) -> torch.Tensor:
    """
    Compute spectral embedding of a graph via normalized Laplacian.

    Parameters
    ----------
    adjacency : [N, N] symmetric adjacency matrix (float)
    k : number of eigen-vector dimensions to retain

    Returns
    -------
    graph_vec : [k]  mean-pooled eigen-vectors (padded with zeros if N-1 < k)
    """
    N = adjacency.size(0)
    if N == 0:
        return torch.zeros(k, device=adjacency.device, dtype=adjacency.dtype)

    # Degree matrix
    degree = adjacency.sum(dim=-1)
    D_inv_sqrt = torch.diag(torch.rsqrt(degree.clamp_min(1.0)))

    I = torch.eye(N, device=adjacency.device, dtype=adjacency.dtype)
    L = I - D_inv_sqrt @ adjacency @ D_inv_sqrt

    # Eigendecomposition (symmetric => eigh)
    _, eigvecs = torch.linalg.eigh(L)

    # Skip the first eigen-vector (constant, eigen-value 0 for connected graph).
    if N > 1:
        vectors = eigvecs[:, 1:]  # [N, N-1]
    else:
        vectors = eigvecs  # [N, N]

    actual_k = min(k, vectors.size(1))
    selected = vectors[:, :actual_k]  # [N, actual_k]

    # Mean-pool over nodes.
    graph_vec = selected.mean(dim=0)  # [actual_k]

    if actual_k < k:
        pad = torch.zeros(
            k - actual_k, device=graph_vec.device, dtype=graph_vec.dtype
        )
        graph_vec = torch.cat([graph_vec, pad], dim=0)

    return graph_vec  # [k]


def _mean_pool_entity_embeddings(
    entity_emb: torch.Tensor,
    entity_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool token embeddings for each entity.

    Parameters
    ----------
    entity_emb : [N, T, D]
    entity_mask : [N, T]  (1 for real token, 0 for pad)

    Returns
    -------
    pooled : [N, D]
    """
    mask = entity_mask.unsqueeze(-1).to(entity_emb.dtype)  # [N, T, 1]
    summed = (entity_emb * mask).sum(dim=1)  # [N, D]
    denom = mask.sum(dim=1).clamp_min(1.0)  # [N, 1]
    return summed / denom


def _build_cooccurrence_pairs(
    spans: List[List[int]],
    window: int = 200,
) -> List[Tuple[int, int]]:
    """
    Build entity co-occurrence pairs based on character-distance window.

    Parameters
    ----------
    spans : list of [start, end] character positions.
    window : maximum character distance to be considered co-occurring.

    Returns
    -------
    pairs : list of (i, j) tuples with i < j.
    """
    pairs: List[Tuple[int, int]] = []
    n = len(spans)
    for i in range(n):
        start_i, end_i = spans[i]
        if start_i < 0 or end_i < 0:
            continue
        for j in range(i + 1, n):
            start_j, end_j = spans[j]
            if start_j < 0 or end_j < 0:
                continue
            # Distance between two intervals.
            dist = max(start_i, start_j) - min(end_i, end_j)
            if dist <= window:
                pairs.append((i, j))
    return pairs


class NodeLayerLoss(nn.Module):
    """
    Entity-type-level InfoNCE with missing-type penalty.
    """

    def __init__(
        self,
        missing_penalty: float = 0.5,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.missing_penalty = missing_penalty
        self.temperature = temperature

    def forward(
        self,
        src_entity_emb: torch.Tensor,      # [B, N_src, T, D]
        src_entity_mask: torch.Tensor,     # [B, N_src, T]
        src_type_lists: List[List[str]],   # [B][N_src]
        sum_entity_emb: torch.Tensor,      # [B, N_sum, T, D]
        sum_entity_mask: torch.Tensor,     # [B, N_sum, T]
        sum_type_lists: List[List[str]],   # [B][N_sum]
    ) -> torch.Tensor:
        B = src_entity_emb.size(0)
        device = src_entity_emb.device
        dtype = src_entity_emb.dtype

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        valid_samples = 0

        for b in range(B):
            src_emb = _mean_pool_entity_embeddings(
                src_entity_emb[b], src_entity_mask[b]
            )  # [N_src, D]
            sum_emb = _mean_pool_entity_embeddings(
                sum_entity_emb[b], sum_entity_mask[b]
            )  # [N_sum, D]

            src_types = src_type_lists[b]
            sum_types = sum_type_lists[b]

            # Group by type (skip PAD placeholders).
            src_by_type: dict[str, List[torch.Tensor]] = {}
            for i, t in enumerate(src_types):
                if t == "PAD":
                    continue
                src_by_type.setdefault(t, []).append(src_emb[i])

            sum_by_type: dict[str, List[torch.Tensor]] = {}
            for i, t in enumerate(sum_types):
                if t == "PAD":
                    continue
                sum_by_type.setdefault(t, []).append(sum_emb[i])

            if not src_by_type:
                continue

            # Build true in-sample InfoNCE batches over aligned types.
            common_types = [t for t in src_by_type if t in sum_by_type]
            contrastive_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if common_types:
                src_batch = torch.stack(
                    [torch.stack(src_by_type[t]).mean(dim=0) for t in common_types],
                    dim=0,
                )  # [K, D]
                sum_batch = torch.stack(
                    [torch.stack(sum_by_type[t]).mean(dim=0) for t in common_types],
                    dim=0,
                )  # [K, D]
                contrastive_loss = info_nce(src_batch, sum_batch, self.temperature)

            missing_count = sum(1 for t in src_by_type if t not in sum_by_type)
            missing_loss = self.missing_penalty * float(missing_count)

            sample_loss = contrastive_loss + missing_loss
            total_loss += sample_loss / max(len(src_by_type), 1)
            valid_samples += 1

        if valid_samples == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return total_loss / valid_samples


class LinkLayerLoss(nn.Module):
    """
    TransE geometric constraint on co-occurring entity-type pairs.
    """

    def __init__(
        self,
        hidden_dim: int,
        cooccurrence_window: int = 200,
    ) -> None:
        super().__init__()
        self.relation_vector = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.cooccurrence_window = cooccurrence_window

    def forward(
        self,
        src_entity_emb: torch.Tensor,       # [B, N_src, T, D]
        src_entity_mask: torch.Tensor,      # [B, N_src, T]
        src_span_lists: List[List[List[int]]],  # [B][N_src][2]
        src_type_lists: List[List[str]],    # [B][N_src]
        sum_entity_emb: torch.Tensor,       # [B, N_sum, T, D]
        sum_entity_mask: torch.Tensor,      # [B, N_sum, T]
        sum_span_lists: List[List[List[int]]],  # [B][N_sum][2] (accepted for API symmetry, unused)
        sum_type_lists: List[List[str]],    # [B][N_sum]
    ) -> torch.Tensor:
        B = src_entity_emb.size(0)
        device = src_entity_emb.device
        dtype = src_entity_emb.dtype

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        valid_pairs = 0

        for b in range(B):
            src_emb = _mean_pool_entity_embeddings(
                src_entity_emb[b], src_entity_mask[b]
            )  # [N_src, D]
            sum_emb = _mean_pool_entity_embeddings(
                sum_entity_emb[b], sum_entity_mask[b]
            )  # [N_sum, D]

            src_spans = src_span_lists[b]
            src_types = src_type_lists[b]
            sum_types = sum_type_lists[b]

            # Co-occurrence pairs in source text.
            pairs = _build_cooccurrence_pairs(src_spans, self.cooccurrence_window)
            if not pairs:
                continue

            # Build type-level mean embeddings for source.
            src_type_mean: dict[str, torch.Tensor] = {}
            for i, t in enumerate(src_types):
                if t == "PAD":
                    continue
                src_type_mean.setdefault(t, []).append(src_emb[i])
            for t in src_type_mean:
                src_type_mean[t] = torch.stack(src_type_mean[t]).mean(dim=0)

            # Build type-level mean embeddings for summary.
            sum_type_mean: dict[str, torch.Tensor] = {}
            for i, t in enumerate(sum_types):
                if t == "PAD":
                    continue
                sum_type_mean.setdefault(t, []).append(sum_emb[i])
            for t in sum_type_mean:
                sum_type_mean[t] = torch.stack(sum_type_mean[t]).mean(dim=0)

            # Apply TransE on type-level means for co-occurring types.
            for i, j in pairs:
                t_i = src_types[i]
                t_j = src_types[j]
                if t_i == "PAD" or t_j == "PAD":
                    continue
                if t_i in sum_type_mean and t_j in sum_type_mean:
                    predicted = sum_type_mean[t_i] + self.relation_vector
                    total_loss += F.mse_loss(predicted, sum_type_mean[t_j])
                    valid_pairs += 1

        if valid_pairs == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return total_loss / valid_pairs


class NetworkLayerLoss(nn.Module):
    """
    Align spectral graph embedding of the entity co-occurrence graph
    to the decoder's final hidden state.
    """

    def __init__(
        self,
        k: int = 8,
        hidden_dim: int = 512,
        cooccurrence_window: int = 200,
    ) -> None:
        super().__init__()
        self.k = k
        self.cooccurrence_window = cooccurrence_window
        self.projection = nn.Linear(k, hidden_dim)

    def forward(
        self,
        src_entity_emb: torch.Tensor,       # [B, N_src, T, D]
        src_entity_mask: torch.Tensor,      # [B, N_src, T]
        src_span_lists: List[List[List[int]]],  # [B][N_src][2]
        decoder_final_hidden: torch.Tensor, # [B, D]
    ) -> torch.Tensor:
        B = src_entity_emb.size(0)
        device = src_entity_emb.device
        dtype = src_entity_emb.dtype

        losses: List[torch.Tensor] = []

        for b in range(B):
            # Count valid entities (spans != [-1, -1]).
            n_entities = sum(
                1 for spans in src_span_lists[b] if spans[0] >= 0 and spans[1] >= 0
            )
            if n_entities < 3:
                # Not enough nodes to form a meaningful graph.
                continue

            # Build adjacency matrix.
            adj = torch.zeros(
                n_entities, n_entities, device=device, dtype=dtype
            )
            for i in range(n_entities):
                start_i, end_i = src_span_lists[b][i]
                for j in range(i + 1, n_entities):
                    start_j, end_j = src_span_lists[b][j]
                    dist = max(start_i, start_j) - min(end_i, end_j)
                    if dist <= self.cooccurrence_window:
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0

            graph_vec = spectral_embedding(adj, self.k)  # [k]
            projected = self.projection(graph_vec)         # [D]

            losses.append(F.mse_loss(projected, decoder_final_hidden[b]))

        if not losses:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return torch.stack(losses).mean()
