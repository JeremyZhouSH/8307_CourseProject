from __future__ import annotations

import math

import pytest
import torch

from finetune.mi_layers import (
    NodeLayerLoss,
    LinkLayerLoss,
    NetworkLayerLoss,
    info_nce,
    spectral_embedding,
    _build_cooccurrence_pairs,
)


class TestInfoNCE:
    def test_basic_shape_and_range(self):
        anchor = torch.randn(4, 16)
        positive = torch.randn(4, 16)
        loss = info_nce(anchor, positive, temperature=0.07)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_single_sample_fallback(self):
        anchor = torch.randn(1, 8)
        positive = torch.randn(1, 8)
        loss = info_nce(anchor, positive)
        assert loss.ndim == 0
        assert 0.0 <= loss.item() <= 2.0

    def test_empty_tensor(self):
        loss = info_nce(torch.tensor([]), torch.tensor([]))
        assert loss.item() == 0.0


class TestSpectralEmbedding:
    def test_complete_graph(self):
        n = 5
        adj = torch.ones(n, n) - torch.eye(n)
        vec = spectral_embedding(adj, k=4)
        assert vec.shape == (4,)
        assert torch.isfinite(vec).all()

    def test_line_graph(self):
        n = 4
        adj = torch.zeros(n, n)
        for i in range(n - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        vec = spectral_embedding(adj, k=3)
        assert vec.shape == (3,)

    def test_padded_when_graph_small(self):
        adj = torch.zeros(2, 2)
        adj[0, 1] = adj[1, 0] = 1.0
        vec = spectral_embedding(adj, k=8)
        assert vec.shape == (8,)

    def test_empty_graph(self):
        vec = spectral_embedding(torch.zeros(0, 0), k=4)
        assert vec.shape == (4,)
        assert (vec == 0.0).all()


class TestBuildCooccurrencePairs:
    def test_basic_window(self):
        spans = [[0, 10], [50, 60], [300, 310]]
        pairs = _build_cooccurrence_pairs(spans, window=200)
        assert (0, 1) in pairs
        assert (0, 2) not in pairs
        assert (1, 2) not in pairs

    def test_skips_negative_spans(self):
        spans = [[0, 10], [-1, -1], [50, 60]]
        pairs = _build_cooccurrence_pairs(spans, window=200)
        assert (0, 2) in pairs
        assert all(i != 1 and j != 1 for i, j in pairs)


class TestNodeLayerLoss:
    @pytest.fixture
    def node_loss(self):
        return NodeLayerLoss(missing_penalty=0.5, temperature=0.07)

    def test_same_types_present(self, node_loss):
        # 2 samples, 3 entities each, token_len=4, hidden=8
        src_emb = torch.randn(2, 3, 4, 8)
        src_mask = torch.ones(2, 3, 4, dtype=torch.long)
        src_types = [["CHEMICAL", "GENE", "PAD"], ["GENE", "PAD", "PAD"]]

        sum_emb = torch.randn(2, 2, 4, 8)
        sum_mask = torch.ones(2, 2, 4, dtype=torch.long)
        sum_types = [["CHEMICAL", "GENE"], ["GENE", "PAD"]]

        loss = node_loss(src_emb, src_mask, src_types, sum_emb, sum_mask, sum_types)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_missing_type_penalty(self, node_loss):
        src_emb = torch.randn(1, 2, 4, 8)
        src_mask = torch.ones(1, 2, 4, dtype=torch.long)
        src_types = [["CHEMICAL", "GENE"]]

        sum_emb = torch.randn(1, 1, 4, 8)
        sum_mask = torch.ones(1, 1, 4, dtype=torch.long)
        sum_types = [["CHEMICAL"]]  # GENE missing

        loss = node_loss(src_emb, src_mask, src_types, sum_emb, sum_mask, sum_types)
        # Should include missing penalty for GENE.
        assert loss.item() >= 0.25  # at least partial penalty

    def test_all_pad_entities(self, node_loss):
        src_emb = torch.randn(1, 2, 4, 8)
        src_mask = torch.ones(1, 2, 4, dtype=torch.long)
        src_types = [["PAD", "PAD"]]

        sum_emb = torch.randn(1, 1, 4, 8)
        sum_mask = torch.ones(1, 1, 4, dtype=torch.long)
        sum_types = [["PAD"]]

        loss = node_loss(src_emb, src_mask, src_types, sum_emb, sum_mask, sum_types)
        assert loss.item() == 0.0


class TestLinkLayerLoss:
    @pytest.fixture
    def link_loss(self):
        return LinkLayerLoss(hidden_dim=8, cooccurrence_window=200)

    def test_basic_transe_constraint(self, link_loss):
        src_emb = torch.randn(1, 3, 4, 8)
        src_mask = torch.ones(1, 3, 4, dtype=torch.long)
        src_spans = [[[0, 10], [50, 60], [300, 310]]]
        src_types = [["CHEMICAL", "GENE", "DISEASE"]]

        sum_emb = torch.randn(1, 2, 4, 8)
        sum_mask = torch.ones(1, 2, 4, dtype=torch.long)
        sum_types = [["CHEMICAL", "GENE"]]  # DISEASE missing -> only 1 co-occurring pair

        loss = link_loss(
            src_emb, src_mask, src_spans, src_types,
            sum_emb, sum_mask, [[[]]], sum_types,
        )
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_no_cooccurrence_pairs(self, link_loss):
        src_emb = torch.randn(1, 2, 4, 8)
        src_mask = torch.ones(1, 2, 4, dtype=torch.long)
        src_spans = [[[0, 10], [500, 510]]]  # far apart
        src_types = [["CHEMICAL", "GENE"]]

        sum_emb = torch.randn(1, 2, 4, 8)
        sum_mask = torch.ones(1, 2, 4, dtype=torch.long)
        sum_types = [["CHEMICAL", "GENE"]]

        loss = link_loss(
            src_emb, src_mask, src_spans, src_types,
            sum_emb, sum_mask, [[[0, 10], [20, 30]]], sum_types,
        )
        assert loss.item() == 0.0


class TestNetworkLayerLoss:
    @pytest.fixture
    def network_loss(self):
        return NetworkLayerLoss(k=4, hidden_dim=8, cooccurrence_window=200)

    def test_basic_graph_alignment(self, network_loss):
        src_emb = torch.randn(1, 5, 4, 8)
        src_mask = torch.ones(1, 5, 4, dtype=torch.long)
        src_spans = [[[i * 100, i * 100 + 10] for i in range(5)]]

        decoder_hidden = torch.randn(1, 8)

        loss = network_loss(src_emb, src_mask, src_spans, decoder_hidden)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_too_few_entities(self, network_loss):
        src_emb = torch.randn(1, 2, 4, 8)
        src_mask = torch.ones(1, 2, 4, dtype=torch.long)
        src_spans = [[[0, 10], [20, 30]]]

        decoder_hidden = torch.randn(1, 8)

        loss = network_loss(src_emb, src_mask, src_spans, decoder_hidden)
        assert loss.item() == 0.0
