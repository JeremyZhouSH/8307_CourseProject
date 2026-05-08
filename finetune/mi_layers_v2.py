"""
Rewritten Three-Layer alignment losses for biomedical summarization.

Key upgrades over v1:
- Node layer: instance-level matching within each entity type (not only type means)
- Link layer: typed relation vectors + weighted co-occurrence edges + hard negatives
- Network layer: sign-invariant spectral signature + attention-pooled decoder state

The module keeps PyTorch-only dependencies to stay easy to integrate.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE with in-batch negatives.

    anchor: [B, D]
    positive: [B, D]
    """
    if anchor.numel() == 0 or positive.numel() == 0:
        return anchor.new_zeros(())

    if anchor.size(0) == 1:
        return (1.0 - F.cosine_similarity(anchor, positive, dim=-1)).mean()

    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = torch.matmul(anchor, positive.t()) / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)


def _mean_pool_entity_embeddings(entity_emb: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings for each entity.

    entity_emb: [N, T, D]
    entity_mask: [N, T]
    returns: [N, D]
    """
    mask = entity_mask.unsqueeze(-1).to(entity_emb.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (entity_emb * mask).sum(dim=1) / denom


def _valid_span(span: Sequence[int]) -> bool:
    return len(span) == 2 and span[0] >= 0 and span[1] >= 0 and span[1] >= span[0]


def _interval_distance(a: Sequence[int], b: Sequence[int]) -> float:
    # 0 for overlap; positive for separated intervals.
    return float(max(a[0], b[0]) - min(a[1], b[1]))


def _build_weighted_edges(spans: List[List[int]], window: int, tau: float) -> List[Tuple[int, int, float]]:
    """Build weighted co-occurrence edges with distance-decay weight."""
    edges: List[Tuple[int, int, float]] = []
    n = len(spans)
    for i in range(n):
        if not _valid_span(spans[i]):
            continue
        for j in range(i + 1, n):
            if not _valid_span(spans[j]):
                continue
            dist = _interval_distance(spans[i], spans[j])
            if dist <= window:
                # Larger weight for closer entities; clamp for numerical safety.
                w = math.exp(-max(dist, 0.0) / max(tau, 1e-6))
                edges.append((i, j, float(max(min(w, 1.0), 1e-4))))
    return edges


def _group_indices_by_type(type_list: List[str]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for idx, t in enumerate(type_list):
        if t == "PAD":
            continue
        grouped.setdefault(t, []).append(idx)
    return grouped


def _mutual_best_pairs(sim: torch.Tensor) -> List[Tuple[int, int]]:
    """Find mutual-best pairs from similarity matrix [N_src, N_sum]."""
    if sim.numel() == 0:
        return []

    src_to_sum = sim.argmax(dim=1)
    sum_to_src = sim.argmax(dim=0)

    pairs: List[Tuple[int, int]] = []
    for i in range(sim.size(0)):
        j = int(src_to_sum[i].item())
        if int(sum_to_src[j].item()) == i:
            pairs.append((i, j))

    # Fallback: at least one pair (global max) when mutual set is empty.
    if not pairs:
        flat = sim.view(-1)
        best = int(flat.argmax().item())
        i = best // sim.size(1)
        j = best % sim.size(1)
        pairs.append((i, j))
    return pairs


class NodeLayerLossV2(nn.Module):
    """Instance-aware type-aligned contrastive loss.

    Improvements:
    - Type-level alignment is performed via instance matching (mutual best) instead of pure type means.
    - Unmatched source entities are softly penalized.
    - Optional confidence weights can reduce sensitivity to noisy NER.
    """

    def __init__(self, missing_penalty: float = 0.5, temperature: float = 0.07) -> None:
        super().__init__()
        self.missing_penalty = missing_penalty
        self.temperature = temperature

    def forward(
        self,
        src_entity_emb: torch.Tensor,
        src_entity_mask: torch.Tensor,
        src_type_lists: List[List[str]],
        sum_entity_emb: torch.Tensor,
        sum_entity_mask: torch.Tensor,
        sum_type_lists: List[List[str]],
        src_entity_confidence: Optional[List[List[float]]] = None,
    ) -> torch.Tensor:
        """Compute instance-aware node loss.

        src_entity_confidence: optional [B][N_src], in [0,1]
        """
        bsz = src_entity_emb.size(0)
        device = src_entity_emb.device
        dtype = src_entity_emb.dtype

        losses: List[torch.Tensor] = []
        for b in range(bsz):
            src_emb = _mean_pool_entity_embeddings(src_entity_emb[b], src_entity_mask[b])
            sum_emb = _mean_pool_entity_embeddings(sum_entity_emb[b], sum_entity_mask[b])
            src_emb = src_emb.to(sum_emb.dtype)

            src_group = _group_indices_by_type(src_type_lists[b])
            sum_group = _group_indices_by_type(sum_type_lists[b])
            if not src_group:
                continue

            sample_loss = src_emb.new_zeros(())
            type_count = 0

            for t, src_idx in src_group.items():
                type_count += 1
                if t not in sum_group:
                    # Entire type missing in summary.
                    conf_weight = 1.0
                    if src_entity_confidence is not None and b < len(src_entity_confidence):
                        confs = [
                            float(src_entity_confidence[b][i])
                            for i in src_idx
                            if i < len(src_entity_confidence[b])
                        ]
                        if confs:
                            conf_weight = float(sum(confs) / len(confs))
                    sample_loss = sample_loss + conf_weight * self.missing_penalty
                    continue

                sum_idx = sum_group[t]
                src_t = src_emb[src_idx]  # [Ns, D]
                sum_t = sum_emb[sum_idx]  # [Nt, D]

                # Similarity for matching (detached indices, differentiable values).
                sim = torch.matmul(F.normalize(src_t.to(sum_t.dtype), dim=-1), F.normalize(sum_t, dim=-1).t())
                pairs = _mutual_best_pairs(sim.detach())

                src_sel = torch.stack([src_t[i] for i, _ in pairs], dim=0)
                sum_sel = torch.stack([sum_t[j] for _, j in pairs], dim=0)
                align_loss = info_nce(src_sel, sum_sel, self.temperature)
                sample_loss = sample_loss + align_loss

                # Unmatched source entities (within type) get a softer penalty.
                matched_src = {i for i, _ in pairs}
                unmatched = max(len(src_idx) - len(matched_src), 0)
                if unmatched > 0:
                    sample_loss = sample_loss + 0.5 * self.missing_penalty * float(unmatched)

            if type_count > 0:
                losses.append(sample_loss / float(type_count))

        if not losses:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return torch.stack(losses).mean()


class LinkLayerLossV2(nn.Module):
    """Typed-link geometric regularization with weighted edges and hard negatives."""

    def __init__(
        self,
        hidden_dim: int,
        cooccurrence_window: int = 200,
        distance_tau: float = 120.0,
        relation_buckets: int = 4096,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.cooccurrence_window = cooccurrence_window
        self.distance_tau = distance_tau
        self.margin = margin

        self.relation_table = nn.Embedding(relation_buckets, hidden_dim)
        nn.init.normal_(self.relation_table.weight, mean=0.0, std=0.01)

    @staticmethod
    def _bucket_id(type_i: str, type_j: str, n_buckets: int) -> int:
        digest = hashlib.md5(f"{type_i}->{type_j}".encode("utf-8")).hexdigest()
        return int(digest, 16) % n_buckets

    def _relation_vector(self, type_i: str, type_j: str, device: torch.device) -> torch.Tensor:
        idx = self._bucket_id(type_i, type_j, self.relation_table.num_embeddings)
        index = torch.tensor([idx], device=device, dtype=torch.long)
        return self.relation_table(index).squeeze(0)

    def forward(
        self,
        src_entity_emb: torch.Tensor,
        src_entity_mask: torch.Tensor,
        src_span_lists: List[List[List[int]]],
        src_type_lists: List[List[str]],
        sum_entity_emb: torch.Tensor,
        sum_entity_mask: torch.Tensor,
        sum_span_lists: List[List[List[int]]],
        sum_type_lists: List[List[str]],
    ) -> torch.Tensor:
        del src_entity_emb, src_entity_mask, sum_span_lists

        bsz = sum_entity_emb.size(0)
        device = sum_entity_emb.device
        dtype = sum_entity_emb.dtype

        total_loss = sum_entity_emb.new_zeros(())
        valid = 0

        for b in range(bsz):
            sum_emb = _mean_pool_entity_embeddings(sum_entity_emb[b], sum_entity_mask[b])
            sum_types = sum_type_lists[b]
            src_types = src_type_lists[b]
            src_spans = src_span_lists[b]

            sum_group = _group_indices_by_type(sum_types)
            if len(sum_group) < 2:
                continue

            sum_type_mean: Dict[str, torch.Tensor] = {}
            for t, idxs in sum_group.items():
                sum_type_mean[t] = sum_emb[idxs].mean(dim=0)

            edges = _build_weighted_edges(src_spans, self.cooccurrence_window, self.distance_tau)
            if not edges:
                continue

            available_types = list(sum_type_mean.keys())
            for i, j, w in edges:
                if i >= len(src_types) or j >= len(src_types):
                    continue
                ti, tj = src_types[i], src_types[j]
                if ti == "PAD" or tj == "PAD":
                    continue
                if ti not in sum_type_mean or tj not in sum_type_mean:
                    continue

                rel = self._relation_vector(ti, tj, device=device).to(dtype)
                pos_pred = sum_type_mean[ti] + rel
                pos_tgt = sum_type_mean[tj]
                pos_loss = F.mse_loss(pos_pred, pos_tgt)

                # Hard negative: push away one non-target type.
                neg_candidates = [t for t in available_types if t != tj]
                if neg_candidates:
                    neg_idx = int(hashlib.md5(f"{ti}-{tj}-{b}".encode("utf-8")).hexdigest(), 16) % len(neg_candidates)
                    neg_t = neg_candidates[neg_idx]
                    neg_tgt = sum_type_mean[neg_t]
                    d_pos = F.pairwise_distance(pos_pred.unsqueeze(0), pos_tgt.unsqueeze(0), p=2)
                    d_neg = F.pairwise_distance(pos_pred.unsqueeze(0), neg_tgt.unsqueeze(0), p=2)
                    rank_loss = F.relu(self.margin + d_pos - d_neg).mean()
                else:
                    rank_loss = pos_loss.new_zeros(())

                edge_loss = w * (pos_loss + 0.5 * rank_loss)
                total_loss = total_loss + edge_loss
                valid += 1

        if valid == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return total_loss / float(valid)


class NetworkLayerLossV2(nn.Module):
    """Graph-level structure regularizer with sign-invariant spectral signature."""

    def __init__(
        self,
        k: int = 8,
        hidden_dim: int = 512,
        cooccurrence_window: int = 200,
        distance_tau: float = 120.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.cooccurrence_window = cooccurrence_window
        self.distance_tau = distance_tau

        # Signature = [top-k eigenvalues, mean(|eigenvectors|) top-k] => 2k dims.
        self.projection = nn.Linear(2 * k, hidden_dim)
        self.pool_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def _spectral_signature(self, adjacency: torch.Tensor) -> torch.Tensor:
        n = adjacency.size(0)
        device, dtype = adjacency.device, adjacency.dtype
        if n == 0:
            return torch.zeros(2 * self.k, device=device, dtype=dtype)

        degree = adjacency.sum(dim=-1)
        d_inv_sqrt = torch.diag(torch.rsqrt(degree.clamp_min(1.0)))
        eye = torch.eye(n, device=device, dtype=dtype)
        lap = eye - d_inv_sqrt @ adjacency @ d_inv_sqrt

        lap_eigh = lap.float() if lap.dtype in (torch.float16, torch.bfloat16) else lap
        eigvals, eigvecs = torch.linalg.eigh(lap_eigh)
        eigvals = eigvals.to(dtype)
        eigvecs = eigvecs.to(dtype)

        # Skip the first trivial component when possible.
        if n > 1:
            eigvals = eigvals[1:]
            eigvecs = eigvecs[:, 1:]

        actual_k = min(self.k, eigvals.numel())
        val_feat = eigvals[:actual_k]
        vec_feat = eigvecs[:, :actual_k].abs().mean(dim=0)

        if actual_k < self.k:
            pad = torch.zeros(self.k - actual_k, device=device, dtype=dtype)
            val_feat = torch.cat([val_feat, pad], dim=0)
            vec_feat = torch.cat([vec_feat, pad], dim=0)

        return torch.cat([val_feat, vec_feat], dim=0)

    def _pool_decoder_state(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        # decoder_hidden: [L, D]
        attn = torch.matmul(decoder_hidden, self.pool_query.to(decoder_hidden.dtype))
        attn = torch.softmax(attn, dim=0)
        pooled = torch.sum(attn.unsqueeze(-1) * decoder_hidden, dim=0)
        return pooled

    def forward(
        self,
        src_entity_emb: torch.Tensor,
        src_entity_mask: torch.Tensor,
        src_span_lists: List[List[List[int]]],
        decoder_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        del src_entity_emb, src_entity_mask

        bsz = decoder_hidden_state.size(0)
        device = decoder_hidden_state.device
        dtype = decoder_hidden_state.dtype

        losses: List[torch.Tensor] = []

        for b in range(bsz):
            valid_spans = [s for s in src_span_lists[b] if _valid_span(s)]
            n = len(valid_spans)
            if n < 3:
                continue

            adjacency = torch.zeros(n, n, device=device, dtype=dtype)
            edges = _build_weighted_edges(valid_spans, self.cooccurrence_window, self.distance_tau)
            for i, j, w in edges:
                adjacency[i, j] = w
                adjacency[j, i] = w

            graph_sig = self._spectral_signature(adjacency)
            graph_proj = self.projection(graph_sig.to(self.projection.weight.dtype))

            dec = decoder_hidden_state[b]
            if dec.dim() == 1:
                target = dec
            else:
                target = self._pool_decoder_state(dec)

            target = target.to(graph_proj.dtype)
            mse = F.mse_loss(graph_proj, target)
            cos = 1.0 - F.cosine_similarity(graph_proj.unsqueeze(0), target.unsqueeze(0), dim=-1).mean()
            losses.append(0.5 * mse + 0.5 * cos)

        if not losses:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return torch.stack(losses).mean()
