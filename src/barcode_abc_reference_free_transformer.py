# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:02:10 2026

@author: kaya-
"""

# -*- coding: utf-8 -*-
"""
barcode_abc_reference_free_transformer.py

Reference-free Transformer-generated candidate pools for ABC-based
DNA barcode library optimization.

Important terminology
---------------------
This file intentionally avoids PRNG/keystream/randomness-test terminology.
The Transformer is used only as a reference-free DNA candidate generator.
The final barcode library is selected and optimized by ABC at the library level.

Example command
---------------
python src/barcode_abc_reference_free_transformer.py --length 12 --size 64 --iters 20 --foods 12 --raw-pool-size 1024 --seed-pool-size 256 --filter-sample-size 100 --run-abc-control --out ref_transformer_abc_test

Faster test command
-------------------
python src/barcode_abc_reference_free_transformer.py --length 12 --size 64 --iters 20 --foods 12 --raw-pool-size 512 --seed-pool-size 256 --filter-sample-size 80 --run-abc-control --out ref_transformer_abc_fast
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


DNA = ["A", "C", "G", "T"]
VOCAB = {b: i for i, b in enumerate(DNA)}
INV_VOCAB = {i: b for b, i in VOCAB.items()}


# =============================================================================
# Utilities
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def gc_fraction(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / max(1, len(seq))


def max_homopolymer_run(seq: str) -> int:
    if not seq:
        return 0

    best = 1
    cur = 1

    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1

    return best


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Hamming distance requires equal-length sequences.")

    return sum(x != y for x, y in zip(a, b))


def kmer_counts(seq: str, k: int) -> Dict[str, int]:
    out: Dict[str, int] = {}

    if len(seq) < k:
        return out

    for i in range(len(seq) - k + 1):
        km = seq[i:i + k]
        out[km] = out.get(km, 0) + 1

    return out


def mutate_base(seq: str, pos: int, new_base: str) -> str:
    if seq[pos] == new_base:
        return seq

    return seq[:pos] + new_base + seq[pos + 1:]


def make_torch_generator(seed: int, device: str):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def sample_categorical(probs_1d, gen):
    p = probs_1d / probs_1d.sum()
    cdf = torch.cumsum(p, dim=0)
    u = torch.rand((1,), generator=gen, device=p.device, dtype=p.dtype)
    idx = torch.searchsorted(cdf, u, right=False).clamp(max=p.numel() - 1)
    return idx


# =============================================================================
# Configs
# =============================================================================

@dataclass
class BarcodeConfig:
    length: int = 12
    size: int = 64

    gc_min: float = 0.40
    gc_max: float = 0.60
    gc_target: float = 0.50

    homopolymer_max: int = 3
    error_radius: int = 1
    kmer_k: int = 3

    w_min_dist: float = 4.0
    w_avg_dist: float = 1.0
    w_gc: float = 1.5
    w_homopolymer: float = 1.5
    w_collision: float = 4.0
    w_kmer: float = 0.5
    w_duplicate: float = 10.0


@dataclass
class ABCConfig:
    foods: int = 12
    iters: int = 20
    limit: int = 20
    replace_rate: float = 0.10
    mutation_rate: float = 0.30
    local_trials: int = 3
    candidate_trials: int = 8
    print_every: int = 1

    greedy_init_ratio: float = 0.40
    stochastic_init_ratio: float = 0.40

    pool_use_start: float = 0.60
    pool_use_end: float = 0.15


@dataclass
class LibraryMetrics:
    fitness: float
    min_hamming: int
    avg_hamming: float
    gc_penalty: float
    homopolymer_penalty: float
    collision_pairs_radius1: int
    duplicate_count: int
    kmer_entropy: float
    valid_gc_fraction: float
    valid_hp_fraction: float


@dataclass
class PoolSummary:
    raw_pool_size_requested: int
    raw_generated_unique: int
    selected_pool_size_requested: int
    selected_pool_size: int

    transformer_candidates: int
    random_fill_candidates: int

    valid_gc_fraction_raw: float
    valid_hp_fraction_raw: float
    raw_kmer_entropy: float

    valid_gc_fraction_selected: float
    valid_hp_fraction_selected: float
    selected_kmer_entropy: float

    transformer_stream_bases: int
    runtime_generation_seconds: float
    runtime_filter_seconds: float
    runtime_total_seconds: float


# =============================================================================
# Reference-free Transformer DNA candidate generator
# =============================================================================

@dataclass
class TransformerConfig:
    vocab_size: int = 4
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.0


if TORCH_AVAILABLE:

    class CausalSelfAttention(nn.Module):
        def __init__(self, cfg: TransformerConfig):
            super().__init__()

            assert cfg.n_embd % cfg.n_head == 0

            self.n_head = cfg.n_head
            self.head_dim = cfg.n_embd // cfg.n_head

            assert self.head_dim % 2 == 0

            self.key = nn.Linear(cfg.n_embd, cfg.n_embd)
            self.query = nn.Linear(cfg.n_embd, cfg.n_embd)
            self.value = nn.Linear(cfg.n_embd, cfg.n_embd)
            self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

            base = 10000.0
            d = self.head_dim

            inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def _apply_rope(self, x, start_index: int = 0):
            B, H, T, D = x.shape

            x = x.view(B, H, T, D // 2, 2)
            x1, x2 = x[..., 0], x[..., 1]

            device = x1.device
            dtype = x1.dtype

            pos = torch.arange(
                start_index,
                start_index + T,
                device=device,
                dtype=self.inv_freq.dtype,
            )

            freqs = torch.einsum("t,f->tf", pos, self.inv_freq)

            cos = freqs.cos().to(dtype)[None, None, :, :]
            sin = freqs.sin().to(dtype)[None, None, :, :]

            xr0 = x1 * cos - x2 * sin
            xr1 = x1 * sin + x2 * cos

            return torch.stack([xr0, xr1], dim=-1).reshape(B, H, T, D)

        def forward_with_past(
            self,
            x,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            pos_offset: int = 0,
        ):
            B, T, C = x.size()
            H, D = self.n_head, self.head_dim

            k = self.key(x).view(B, T, H, D).transpose(1, 2)
            q = self.query(x).view(B, T, H, D).transpose(1, 2)
            v = self.value(x).view(B, T, H, D).transpose(1, 2)

            past_len = 0 if past_kv is None else past_kv[0].size(2)
            start = pos_offset + past_len

            q = self._apply_rope(q, start_index=start)
            k = self._apply_rope(k, start_index=start)

            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)

            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )

            y = y.transpose(1, 2).contiguous().view(B, T, H * D)

            return self.proj(y), (k, v)


    class Block(nn.Module):
        def __init__(self, cfg: TransformerConfig):
            super().__init__()

            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.attn = CausalSelfAttention(cfg)

            self.ln2 = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            )

        def forward_with_past(
            self,
            x,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            pos_offset: int = 0,
        ):
            y, present = self.attn.forward_with_past(
                self.ln1(x),
                past_kv=past_kv,
                pos_offset=pos_offset,
            )

            x = x + y
            x = x + self.mlp(self.ln2(x))

            return x, present


    class ReferenceFreeDNAStreamTransformer(nn.Module):
        """
        Reference-free DNA Transformer candidate generator.

        It is not used as a final barcode designer.
        It only generates DNA candidate material. ABC performs the final
        library-level selection and optimization.
        """

        def __init__(self, cfg: TransformerConfig):
            super().__init__()

            self.cfg = cfg

            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
            self.ln_f = nn.LayerNorm(cfg.n_embd)
            self.base_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        def forward_with_past(self, idx, past_kv=None, pos_offset: int = 0):
            x = self.tok_emb(idx)

            presents = []

            for i, blk in enumerate(self.blocks):
                pkv_i = None if past_kv is None else past_kv[i]
                x, present_i = blk.forward_with_past(
                    x,
                    past_kv=pkv_i,
                    pos_offset=pos_offset,
                )
                presents.append(present_i)

            x = self.ln_f(x)

            return self.base_head(x), presents


def prune_kv_cache(past_kv, max_size: int):
    pruned = []

    for K, V in past_kv:
        seq_len = K.size(2)

        if seq_len > max_size:
            pruned.append(
                (
                    K[:, :, -max_size:, :].contiguous(),
                    V[:, :, -max_size:, :].contiguous(),
                )
            )
        else:
            pruned.append((K, V))

    return pruned


class ReferenceFreeCandidateGenerator:
    def __init__(
        self,
        cfg: BarcodeConfig,
        seed: int,
        use_transformer: bool = True,
        temperature: float = 1.0,
        device: Optional[str] = None,
        block_size: int = 128,
    ):
        self.cfg = cfg
        self.seed = int(seed)
        self.temperature = float(temperature)

        self.rng = np.random.default_rng(seed)

        self.use_transformer = bool(use_transformer and TORCH_AVAILABLE)
        self.device = device or (
            "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        )

        self.model = None
        self.torch_gen = None
        self.block_size = int(block_size)

        if self.use_transformer:
            torch.manual_seed(seed)

            tcfg = TransformerConfig(
                block_size=self.block_size,
                dropout=0.0,
            )

            self.model = ReferenceFreeDNAStreamTransformer(tcfg).to(self.device).eval()

            self.torch_gen = make_torch_generator(seed + 999, self.device)

    def valid_soft(self, seq: str) -> bool:
        return (
            self.cfg.gc_min <= gc_fraction(seq) <= self.cfg.gc_max
            and max_homopolymer_run(seq) <= self.cfg.homopolymer_max
        )

    def random_sequence(self) -> str:
        p_gc = self.cfg.gc_target / 2.0
        p_at = (1.0 - self.cfg.gc_target) / 2.0

        probs = np.array([p_at, p_gc, p_gc, p_at], dtype=float)
        probs /= probs.sum()

        seq = ""

        for _ in range(300):
            arr = self.rng.choice(DNA, size=self.cfg.length, p=probs)
            seq = "".join(arr.tolist())

            if self.valid_soft(seq):
                return seq

        return self.repair_basic(seq)

    def repair_basic(self, seq: str) -> str:
        seq_list = list(seq)

        # Homopolymer repair.
        for _ in range(len(seq_list) * 2):
            cur = "".join(seq_list)

            if max_homopolymer_run(cur) <= self.cfg.homopolymer_max:
                break

            run = 1

            for j in range(1, len(seq_list)):
                if seq_list[j] == seq_list[j - 1]:
                    run += 1

                    if run > self.cfg.homopolymer_max:
                        choices = [b for b in DNA if b != seq_list[j]]
                        seq_list[j] = str(self.rng.choice(choices))
                        break
                else:
                    run = 1

        # GC repair.
        for _ in range(self.cfg.length * 3):
            seq2 = "".join(seq_list)
            gcf = gc_fraction(seq2)

            if self.cfg.gc_min <= gcf <= self.cfg.gc_max:
                break

            if gcf < self.cfg.gc_min:
                positions = [
                    i for i, b in enumerate(seq_list)
                    if b in ["A", "T"]
                ]

                if not positions:
                    break

                pos = int(self.rng.choice(positions))
                seq_list[pos] = str(self.rng.choice(["G", "C"]))

            else:
                positions = [
                    i for i, b in enumerate(seq_list)
                    if b in ["G", "C"]
                ]

                if not positions:
                    break

                pos = int(self.rng.choice(positions))
                seq_list[pos] = str(self.rng.choice(["A", "T"]))

        return "".join(seq_list)

    def generate_stream(self, n_bases: int) -> str:
        """
        Generate a DNA candidate stream.

        The generated stream is later split into candidate barcode windows.
        """

        if not self.use_transformer or self.model is None:
            return "".join(self.random_sequence() for _ in range(max(1, n_bases // self.cfg.length)))

        model = self.model
        model.eval()

        prompt_len = self.block_size

        idx = torch.randint(
            low=0,
            high=4,
            size=(1, prompt_len),
            generator=self.torch_gen,
            device=self.device,
            dtype=torch.long,
        )

        last_base = None
        run_len = 0
        gc_count_window = 0
        window_len = 0

        pos_offset = 0
        out_chars = []

        with torch.no_grad():
            base_logits, past = model.forward_with_past(
                idx,
                past_kv=None,
                pos_offset=pos_offset,
            )

            # Initialize local state from prompt tail.
            prompt_list = idx[0].detach().cpu().numpy().tolist()

            for b in prompt_list[-min(8, len(prompt_list)):]:
                ch = INV_VOCAB[int(b)]
                if ch in ["G", "C"]:
                    gc_count_window += 1
                window_len += 1

            for step in range(n_bases):
                logits = base_logits[:, -1, :].clone()[0]
                logits = logits / max(1e-4, self.temperature)

                p = torch.softmax(logits, dim=-1)

                # Soft GC steering over a short local window.
                target_vec = torch.tensor(
                    [
                        (1.0 - self.cfg.gc_target) / 2.0,
                        self.cfg.gc_target / 2.0,
                        self.cfg.gc_target / 2.0,
                        (1.0 - self.cfg.gc_target) / 2.0,
                    ],
                    device=p.device,
                    dtype=p.dtype,
                )

                mix_strength = 0.10
                p = (1.0 - mix_strength) * p + mix_strength * target_vec
                p = p / p.sum()

                # Homopolymer hard constraint.
                if last_base is not None and run_len >= self.cfg.homopolymer_max:
                    p[last_base] = 0.0

                if p.sum() <= 0 or not torch.isfinite(p).all():
                    p = torch.full((4,), 0.25, device=self.device)

                    if last_base is not None and run_len >= self.cfg.homopolymer_max:
                        p[last_base] = 0.0

                p = p / p.sum()

                b = int(sample_categorical(p, self.torch_gen).item())
                ch = INV_VOCAB[b]

                out_chars.append(ch)

                if last_base is None or b != last_base:
                    last_base = b
                    run_len = 1
                else:
                    run_len += 1

                tok = torch.tensor([[b]], dtype=torch.long, device=self.device)
                base_logits, past = model.forward_with_past(
                    tok,
                    past_kv=past,
                    pos_offset=pos_offset,
                )

                if past is not None:
                    seq_len = max(k.size(2) for k, v in past)

                    if seq_len > self.block_size:
                        drop = seq_len - self.block_size
                        pos_offset += drop
                        past = prune_kv_cache(past, max_size=self.block_size)

        return "".join(out_chars)


# =============================================================================
# Metrics
# =============================================================================

def gc_penalty(seq: str, cfg: BarcodeConfig) -> float:
    g = gc_fraction(seq)

    if cfg.gc_min <= g <= cfg.gc_max:
        return 0.0

    if g < cfg.gc_min:
        return cfg.gc_min - g

    return g - cfg.gc_max


def homopolymer_penalty(seq: str, cfg: BarcodeConfig) -> float:
    hp = max_homopolymer_run(seq)

    return max(0, hp - cfg.homopolymer_max) / max(1, cfg.length)


def pairwise_distance_stats(library: List[str]) -> Tuple[int, float, List[List[int]]]:
    n = len(library)

    if n < 2:
        return 0, 0.0, [[0]]

    dists = []
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    for i, j in combinations(range(n), 2):
        d = hamming(library[i], library[j])
        matrix[i][j] = d
        matrix[j][i] = d
        dists.append(d)

    return int(min(dists)), float(np.mean(dists)), matrix


def collision_count_radius(library: List[str], radius: int = 1) -> int:
    threshold = 0 if radius <= 0 else 2 * radius
    cnt = 0

    for a, b in combinations(library, 2):
        if hamming(a, b) <= threshold:
            cnt += 1

    return cnt


def library_kmer_entropy(library: List[str], k: int = 3) -> float:
    counts: Dict[str, int] = {}

    for seq in library:
        for km, c in kmer_counts(seq, k).items():
            counts[km] = counts.get(km, 0) + c

    total = sum(counts.values())

    if total <= 0:
        return 0.0

    probs = np.array([c / total for c in counts.values()], dtype=float)
    ent = -np.sum(probs * np.log2(probs + 1e-12))
    max_ent = math.log2(4 ** k)

    return float(ent / max_ent) if max_ent > 0 else 0.0


def evaluate_library(library: List[str], cfg: BarcodeConfig) -> LibraryMetrics:
    n = len(library)

    unique_n = len(set(library))
    duplicate_count = n - unique_n

    min_d, avg_d, _ = pairwise_distance_stats(library)

    gc_p = float(np.mean([gc_penalty(s, cfg) for s in library])) if library else 1.0
    hp_p = float(np.mean([homopolymer_penalty(s, cfg) for s in library])) if library else 1.0

    collision_pairs = collision_count_radius(library, cfg.error_radius)
    k_ent = library_kmer_entropy(library, cfg.kmer_k)

    valid_gc = float(
        np.mean([cfg.gc_min <= gc_fraction(s) <= cfg.gc_max for s in library])
    ) if library else 0.0

    valid_hp = float(
        np.mean([
            max_homopolymer_run(s) <= cfg.homopolymer_max
            for s in library
        ])
    ) if library else 0.0

    min_d_norm = min_d / max(1, cfg.length)
    avg_d_norm = avg_d / max(1, cfg.length)

    possible_pairs = max(1, n * (n - 1) // 2)
    collision_rate = collision_pairs / possible_pairs
    duplicate_rate = duplicate_count / max(1, n)

    fitness = (
        cfg.w_min_dist * min_d_norm
        + cfg.w_avg_dist * avg_d_norm
        - cfg.w_gc * gc_p
        - cfg.w_homopolymer * hp_p
        - cfg.w_collision * collision_rate
        + cfg.w_kmer * k_ent
        - cfg.w_duplicate * duplicate_rate
    )

    return LibraryMetrics(
        fitness=float(fitness),
        min_hamming=int(min_d),
        avg_hamming=float(avg_d),
        gc_penalty=float(gc_p),
        homopolymer_penalty=float(hp_p),
        collision_pairs_radius1=int(collision_pairs),
        duplicate_count=int(duplicate_count),
        kmer_entropy=float(k_ent),
        valid_gc_fraction=float(valid_gc),
        valid_hp_fraction=float(valid_hp),
    )


def single_sequence_score(seq: str, lib: List[str], cfg: BarcodeConfig) -> float:
    others = [x for x in lib if x != seq]

    if others:
        nearest = min(hamming(seq, x) for x in others)
        avgd = float(np.mean([hamming(seq, x) for x in others]))
    else:
        nearest = cfg.length
        avgd = cfg.length

    return (
        3.0 * nearest / cfg.length
        + 1.0 * avgd / cfg.length
        - 1.0 * gc_penalty(seq, cfg)
        - 1.0 * homopolymer_penalty(seq, cfg)
    )


def kmer_novelty(seq: str, selected_kmers: Set[str], k: int) -> float:
    kmers = list(kmer_counts(seq, k).keys())

    if not kmers:
        return 0.0

    novel = sum(1 for km in kmers if km not in selected_kmers)

    return novel / len(kmers)


# =============================================================================
# Candidate pool construction
# =============================================================================

def stream_to_candidates(
    stream: str,
    cfg: BarcodeConfig,
    stride: Optional[int] = None,
) -> List[str]:
    stride = stride or cfg.length

    out = []

    for i in range(0, len(stream) - cfg.length + 1, stride):
        seq = stream[i:i + cfg.length]

        if len(seq) == cfg.length:
            out.append(seq)

    return out


def diversity_filter_pool(
    raw_pool: List[str],
    cfg: BarcodeConfig,
    selected_size: int,
    seed: int,
    filter_sample_size: int = 100,
) -> List[str]:
    rng = np.random.default_rng(seed)

    unique_pool = list(dict.fromkeys(raw_pool))

    valid_pool = [
        s for s in unique_pool
        if cfg.gc_min <= gc_fraction(s) <= cfg.gc_max
        and max_homopolymer_run(s) <= cfg.homopolymer_max
    ]

    if len(valid_pool) < selected_size:
        valid_pool = unique_pool

    if len(valid_pool) <= selected_size:
        rng.shuffle(valid_pool)
        return valid_pool[:selected_size]

    rng.shuffle(valid_pool)

    first = max(
        valid_pool[: min(1000, len(valid_pool))],
        key=lambda s: -abs(gc_fraction(s) - cfg.gc_target) - homopolymer_penalty(s, cfg),
    )

    selected = [first]
    selected_kmers = set(kmer_counts(first, cfg.kmer_k).keys())
    remaining = [s for s in valid_pool if s != first]

    while len(selected) < selected_size and remaining:
        best = None
        best_score = -1e9

        sample_size = min(len(remaining), filter_sample_size)
        sample_indices = rng.choice(len(remaining), size=sample_size, replace=False)

        for idx in sample_indices:
            s = remaining[int(idx)]

            nearest = min(hamming(s, x) for x in selected)
            avgd = float(np.mean([hamming(s, x) for x in selected]))
            novelty = kmer_novelty(s, selected_kmers, cfg.kmer_k)

            sc = (
                4.0 * nearest / cfg.length
                + 0.8 * avgd / cfg.length
                + 0.25 * novelty
                - 1.0 * gc_penalty(s, cfg)
                - 1.0 * homopolymer_penalty(s, cfg)
            )

            if sc > best_score:
                best = s
                best_score = sc

        if best is None:
            break

        selected.append(best)
        selected_kmers.update(kmer_counts(best, cfg.kmer_k).keys())
        remaining.remove(best)

    return selected


def build_reference_free_candidate_pool(
    generator: ReferenceFreeCandidateGenerator,
    cfg: BarcodeConfig,
    raw_pool_size: int,
    selected_pool_size: int,
    seed: int,
    random_fill_ratio: float = 0.40,
    filter_sample_size: int = 100,
    stream_oversample: int = 4,
) -> Tuple[List[str], PoolSummary]:
    """
    Build an overcomplete candidate pool using the reference-free Transformer
    candidate stream, then select a compact diversity-filtered seed pool.
    """

    t0 = time.time()

    rng = np.random.default_rng(seed + 333)

    raw_pool: List[str] = []
    seen = set()

    target_random = int(round(raw_pool_size * random_fill_ratio))
    target_transformer = max(0, raw_pool_size - target_random)

    transformer_candidates = 0
    random_fill_candidates = 0
    generated_stream_bases = 0

    print("\n[Candidate pool] Building reference-free Transformer candidate stream")
    print(
        f"[Candidate pool] raw total={raw_pool_size}, "
        f"transformer_target={target_transformer}, "
        f"random_fill={target_random}"
    )

    # Generate candidates from long Transformer candidate streams.
    while len(raw_pool) < target_transformer:
        needed = target_transformer - len(raw_pool)
        chunks_to_generate = max(needed * stream_oversample, 256)
        n_bases = chunks_to_generate * cfg.length

        stream = generator.generate_stream(n_bases)
        generated_stream_bases += len(stream)

        candidates = stream_to_candidates(stream, cfg, stride=cfg.length)

        for seq in candidates:
            if seq not in seen:
                seq2 = generator.repair_basic(seq)

                if seq2 not in seen and generator.valid_soft(seq2):
                    seen.add(seq2)
                    raw_pool.append(seq2)
                    transformer_candidates += 1

                    if len(raw_pool) >= target_transformer:
                        break

        print(
            f"[Candidate pool] transformer unique={len(raw_pool)}/{target_transformer}"
        )

        # Safety fallback.
        if generated_stream_bases > raw_pool_size * cfg.length * stream_oversample * 20:
            print("[Candidate pool] Transformer stream safety limit reached; using random fill.")
            break

    # Random constrained fill for diversity and robustness.
    while len(raw_pool) < raw_pool_size:
        seq = generator.random_sequence()

        if seq not in seen:
            seen.add(seq)
            raw_pool.append(seq)
            random_fill_candidates += 1

    rng.shuffle(raw_pool)
    raw_pool = list(dict.fromkeys(raw_pool))

    runtime_generation = time.time() - t0

    raw_valid_gc = float(
        np.mean([cfg.gc_min <= gc_fraction(s) <= cfg.gc_max for s in raw_pool])
    ) if raw_pool else 0.0

    raw_valid_hp = float(
        np.mean([max_homopolymer_run(s) <= cfg.homopolymer_max for s in raw_pool])
    ) if raw_pool else 0.0

    raw_kent = library_kmer_entropy(raw_pool, cfg.kmer_k) if raw_pool else 0.0

    print("[Candidate pool] Diversity filtering candidate pool")

    t1 = time.time()

    selected_pool = diversity_filter_pool(
        raw_pool=raw_pool,
        cfg=cfg,
        selected_size=selected_pool_size,
        seed=seed + 444,
        filter_sample_size=filter_sample_size,
    )

    runtime_filter = time.time() - t1
    runtime_total = time.time() - t0

    selected_valid_gc = float(
        np.mean([cfg.gc_min <= gc_fraction(s) <= cfg.gc_max for s in selected_pool])
    ) if selected_pool else 0.0

    selected_valid_hp = float(
        np.mean([max_homopolymer_run(s) <= cfg.homopolymer_max for s in selected_pool])
    ) if selected_pool else 0.0

    selected_kent = library_kmer_entropy(selected_pool, cfg.kmer_k) if selected_pool else 0.0

    summary = PoolSummary(
        raw_pool_size_requested=int(raw_pool_size),
        raw_generated_unique=int(len(raw_pool)),
        selected_pool_size_requested=int(selected_pool_size),
        selected_pool_size=int(len(selected_pool)),
        transformer_candidates=int(transformer_candidates),
        random_fill_candidates=int(random_fill_candidates),
        valid_gc_fraction_raw=float(raw_valid_gc),
        valid_hp_fraction_raw=float(raw_valid_hp),
        raw_kmer_entropy=float(raw_kent),
        valid_gc_fraction_selected=float(selected_valid_gc),
        valid_hp_fraction_selected=float(selected_valid_hp),
        selected_kmer_entropy=float(selected_kent),
        transformer_stream_bases=int(generated_stream_bases),
        runtime_generation_seconds=float(runtime_generation),
        runtime_filter_seconds=float(runtime_filter),
        runtime_total_seconds=float(runtime_total),
    )

    print(
        "[Candidate pool] done: "
        f"raw_unique={summary.raw_generated_unique}, "
        f"selected={summary.selected_pool_size}, "
        f"selected_kmer={summary.selected_kmer_entropy:.4f}, "
        f"selected_valid_gc={summary.valid_gc_fraction_selected:.3f}, "
        f"selected_valid_hp={summary.valid_hp_fraction_selected:.3f}, "
        f"time={summary.runtime_total_seconds:.2f}s"
    )

    return selected_pool, summary


class CandidateSeedPool:
    def __init__(self, pool: List[str], cfg: BarcodeConfig, seed: int):
        self.pool = list(dict.fromkeys(pool))
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.pool)

    def __len__(self) -> int:
        return len(self.pool)

    def sample_one(self, existing: Optional[set] = None, use_prob: float = 0.50):
        if not self.pool:
            return None

        if self.rng.random() > use_prob:
            return None

        existing = existing or set()

        for _ in range(min(100, len(self.pool))):
            idx = int(self.rng.integers(0, len(self.pool)))
            cand = self.pool[idx]

            if cand not in existing:
                return cand

        return None

    def greedy_library(self, size: int, cfg: BarcodeConfig) -> List[str]:
        if not self.pool:
            return []

        pool = list(self.pool)
        self.rng.shuffle(pool)

        first = max(
            pool[: min(500, len(pool))],
            key=lambda s: -abs(gc_fraction(s) - cfg.gc_target) - homopolymer_penalty(s, cfg),
        )

        lib = [first]
        remaining = [s for s in pool if s != first]
        selected_kmers = set(kmer_counts(first, cfg.kmer_k).keys())

        while len(lib) < size and remaining:
            best = None
            best_score = -1e9

            sample_size = min(len(remaining), 300)
            sample_indices = self.rng.choice(
                len(remaining),
                size=sample_size,
                replace=False,
            )

            for idx in sample_indices:
                s = remaining[int(idx)]
                nearest = min(hamming(s, x) for x in lib)
                avgd = float(np.mean([hamming(s, x) for x in lib]))
                novelty = kmer_novelty(s, selected_kmers, cfg.kmer_k)

                sc = (
                    4.0 * nearest / cfg.length
                    + 1.0 * avgd / cfg.length
                    + 0.20 * novelty
                    - 1.0 * gc_penalty(s, cfg)
                    - 1.0 * homopolymer_penalty(s, cfg)
                )

                if sc > best_score:
                    best = s
                    best_score = sc

            if best is None:
                break

            lib.append(best)
            remaining.remove(best)
            selected_kmers.update(kmer_counts(best, cfg.kmer_k).keys())

        return lib

    def stochastic_library(self, size: int, cfg: BarcodeConfig) -> List[str]:
        if not self.pool:
            return []

        pool = list(self.pool)
        self.rng.shuffle(pool)

        first = pool[0]
        lib = [first]
        remaining = [s for s in pool if s != first]
        selected_kmers = set(kmer_counts(first, cfg.kmer_k).keys())

        while len(lib) < size and remaining:
            sample_size = min(len(remaining), 200)
            sample_indices = self.rng.choice(
                len(remaining),
                size=sample_size,
                replace=False,
            )

            scored = []

            for idx in sample_indices:
                s = remaining[int(idx)]
                nearest = min(hamming(s, x) for x in lib)
                avgd = float(np.mean([hamming(s, x) for x in lib]))
                novelty = kmer_novelty(s, selected_kmers, cfg.kmer_k)

                sc = (
                    3.5 * nearest / cfg.length
                    + 1.0 * avgd / cfg.length
                    + 0.20 * novelty
                    - 1.0 * gc_penalty(s, cfg)
                    - 1.0 * homopolymer_penalty(s, cfg)
                )

                sc += float(self.rng.normal(0.0, 0.015))
                scored.append((sc, s))

            if not scored:
                break

            scored.sort(key=lambda x: x[0], reverse=True)
            top_k = min(10, len(scored))
            chosen = scored[int(self.rng.integers(0, top_k))][1]

            lib.append(chosen)
            remaining.remove(chosen)
            selected_kmers.update(kmer_counts(chosen, cfg.kmer_k).keys())

        return lib


# =============================================================================
# ABC optimizer
# =============================================================================

class ABCBarcodeOptimizer:
    def __init__(
        self,
        barcode_cfg: BarcodeConfig,
        abc_cfg: ABCConfig,
        generator: ReferenceFreeCandidateGenerator,
        seed: int = 1234,
        seed_pool: Optional[CandidateSeedPool] = None,
        seeded_initialization: bool = True,
    ):
        self.cfg = barcode_cfg
        self.abc = abc_cfg
        self.generator = generator
        self.seed_pool = seed_pool
        self.seeded_initialization = bool(seeded_initialization)
        self.rng = np.random.default_rng(seed)

        self.foods: List[List[str]] = []
        self.metrics: List[LibraryMetrics] = []
        self.trials: List[int] = []

        self.best_library: List[str] = []
        self.best_metrics: Optional[LibraryMetrics] = None

        self.log_rows: List[Dict[str, float]] = []
        self.current_pool_use_prob = self.abc.pool_use_start

    def initialize(self):
        self.foods = [self._new_library(i) for i in range(self.abc.foods)]
        self.metrics = [evaluate_library(lib, self.cfg) for lib in self.foods]
        self.trials = [0 for _ in range(self.abc.foods)]

        self._update_best()

        if self.best_metrics is not None:
            print(
                "[ABC init] "
                f"best={self.best_metrics.fitness:.4f} "
                f"minD={self.best_metrics.min_hamming} "
                f"avgD={self.best_metrics.avg_hamming:.2f} "
                f"coll={self.best_metrics.collision_pairs_radius1}"
            )

    def _new_library(self, food_index: int = 0) -> List[str]:
        lib: List[str] = []
        seen = set()

        if (
            self.seeded_initialization
            and self.seed_pool is not None
            and len(self.seed_pool) > 0
        ):
            frac = food_index / max(1, self.abc.foods)

            if frac < self.abc.greedy_init_ratio:
                lib = self.seed_pool.greedy_library(self.cfg.size, self.cfg)
            elif frac < self.abc.greedy_init_ratio + self.abc.stochastic_init_ratio:
                lib = self.seed_pool.stochastic_library(self.cfg.size, self.cfg)
            else:
                lib = []

            lib = list(dict.fromkeys(lib))
            seen = set(lib)

        attempts = 0
        max_attempts = self.cfg.size * 200

        while len(lib) < self.cfg.size and attempts < max_attempts:
            seq = self._generate_candidate(seen)
            attempts += 1

            if seq not in seen:
                seen.add(seq)
                lib.append(seq)

        while len(lib) < self.cfg.size:
            seq = self.generator.random_sequence()

            if seq not in seen:
                seen.add(seq)
                lib.append(seq)

        return lib

    def _generate_candidate(self, existing: Optional[set] = None) -> str:
        existing = existing or set()

        if self.seed_pool is not None:
            seq = self.seed_pool.sample_one(
                existing=existing,
                use_prob=self.current_pool_use_prob,
            )

            if seq is not None:
                return seq

        return self.generator.random_sequence()

    def _update_best(self):
        best_idx = int(np.argmax([m.fitness for m in self.metrics]))
        current = self.metrics[best_idx]

        if self.best_metrics is None or current.fitness > self.best_metrics.fitness:
            self.best_metrics = current
            self.best_library = list(self.foods[best_idx])

    def _per_barcode_badness(self, lib: List[str]) -> np.ndarray:
        n = len(lib)
        bad = np.zeros(n, dtype=float)

        counts: Dict[str, int] = {}

        for s in lib:
            counts[s] = counts.get(s, 0) + 1

        for i, s in enumerate(lib):
            bad[i] += 2.0 * gc_penalty(s, self.cfg)
            bad[i] += 2.0 * homopolymer_penalty(s, self.cfg)

            if counts[s] > 1:
                bad[i] += 5.0

        for i in range(n):
            if n > 1:
                nearest = min(
                    hamming(lib[i], lib[j])
                    for j in range(n)
                    if j != i
                )
            else:
                nearest = self.cfg.length

            bad[i] += (self.cfg.length - nearest) / max(1, self.cfg.length)

        return bad

    def _mutate_sequence(self, seq: str, current_lib: List[str]) -> str:
        best = seq
        best_score = single_sequence_score(seq, current_lib, self.cfg)

        for _ in range(self.abc.local_trials):
            cand = seq

            n_mut = int(
                self.rng.integers(
                    1,
                    max(2, self.cfg.length // 4 + 1),
                )
            )

            for _ in range(n_mut):
                pos = int(self.rng.integers(0, self.cfg.length))
                choices = [b for b in DNA if b != cand[pos]]
                cand = mutate_base(cand, pos, str(self.rng.choice(choices)))

            cand = self.generator.repair_basic(cand)

            if cand in current_lib and cand != seq:
                continue

            sc = single_sequence_score(cand, current_lib, self.cfg)

            if sc > best_score:
                best = cand
                best_score = sc

        return best

    def _generate_unique(
        self,
        lib: List[str],
        exclude_index: Optional[int] = None,
    ) -> str:
        existing = set(lib)

        if exclude_index is not None:
            existing.discard(lib[exclude_index])

        best = None
        best_score = -1e9

        trials = max(1, int(self.abc.candidate_trials))

        for _ in range(trials):
            cand = self._generate_candidate(existing)

            if cand in existing:
                continue

            sc = single_sequence_score(cand, list(existing), self.cfg)

            if sc > best_score:
                best = cand
                best_score = sc

        if best is not None:
            return best

        for _ in range(500):
            cand = self.generator.random_sequence()

            if cand not in existing:
                return cand

        return self.generator.random_sequence()

    def _repair_close_pairs(self, lib: List[str]) -> List[str]:
        threshold = 2 * self.cfg.error_radius

        if threshold <= 0:
            return lib

        new_lib = list(lib)
        max_rounds = max(1, self.cfg.size // 10)

        for _ in range(max_rounds):
            close_pairs = []

            for i, j in combinations(range(len(new_lib)), 2):
                if hamming(new_lib[i], new_lib[j]) <= threshold:
                    close_pairs.append((i, j))

            if not close_pairs:
                break

            i, j = close_pairs[int(self.rng.integers(0, len(close_pairs)))]

            bad = self._per_barcode_badness(new_lib)
            idx = i if bad[i] >= bad[j] else j

            new_lib[idx] = self._generate_unique(new_lib, exclude_index=idx)

        return new_lib

    def _neighbor_library(self, lib: List[str]) -> List[str]:
        new_lib = list(lib)
        n_replace = max(1, int(round(self.cfg.size * self.abc.replace_rate)))

        scores = self._per_barcode_badness(new_lib)
        ranked = np.argsort(scores)[::-1]
        to_modify = ranked[:n_replace]

        for idx in to_modify:
            if self.rng.random() < self.abc.mutation_rate:
                new_lib[idx] = self._mutate_sequence(new_lib[idx], new_lib)
            else:
                new_lib[idx] = self._generate_unique(new_lib, exclude_index=idx)

        new_lib = self._repair_close_pairs(new_lib)

        return new_lib

    def _selection_probs(self) -> np.ndarray:
        fits = np.array([m.fitness for m in self.metrics], dtype=float)
        shifted = fits - fits.min() + 1e-9

        if shifted.sum() <= 0:
            return np.ones_like(shifted) / len(shifted)

        return shifted / shifted.sum()

    def _update_adaptive_pool_use(self, iteration: int):
        if self.abc.iters <= 1:
            self.current_pool_use_prob = self.abc.pool_use_end
            return

        progress = (iteration - 1) / max(1, self.abc.iters - 1)

        self.current_pool_use_prob = (
            self.abc.pool_use_start
            + progress * (self.abc.pool_use_end - self.abc.pool_use_start)
        )

    def run(self):
        self.initialize()

        for it in range(1, self.abc.iters + 1):
            self._update_adaptive_pool_use(it)

            # Employed bees.
            for i in range(self.abc.foods):
                cand_lib = self._neighbor_library(self.foods[i])
                cand_met = evaluate_library(cand_lib, self.cfg)

                if cand_met.fitness > self.metrics[i].fitness:
                    self.foods[i] = cand_lib
                    self.metrics[i] = cand_met
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            # Onlooker bees.
            probs = self._selection_probs()

            for _ in range(self.abc.foods):
                i = int(self.rng.choice(np.arange(self.abc.foods), p=probs))

                cand_lib = self._neighbor_library(self.foods[i])
                cand_met = evaluate_library(cand_lib, self.cfg)

                if cand_met.fitness > self.metrics[i].fitness:
                    self.foods[i] = cand_lib
                    self.metrics[i] = cand_met
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            # Scout bees.
            for i in range(self.abc.foods):
                if self.trials[i] >= self.abc.limit:
                    self.foods[i] = self._new_library(i)
                    self.metrics[i] = evaluate_library(self.foods[i], self.cfg)
                    self.trials[i] = 0

            self._update_best()

            fits = np.array([m.fitness for m in self.metrics], dtype=float)

            row = {
                "iteration": it,
                "best_fitness": float(self.best_metrics.fitness),
                "mean_fitness": float(np.mean(fits)),
                "current_best_fitness": float(np.max(fits)),
                "min_hamming": int(self.best_metrics.min_hamming),
                "avg_hamming": float(self.best_metrics.avg_hamming),
                "collision_pairs_radius1": int(self.best_metrics.collision_pairs_radius1),
                "pool_use_prob": float(self.current_pool_use_prob),
            }

            self.log_rows.append(row)

            if (
                it == 1
                or it == self.abc.iters
                or it % max(1, self.abc.print_every) == 0
            ):
                print(
                    f"[ABC] iter={it:04d}/{self.abc.iters} "
                    f"best={row['best_fitness']:.4f} "
                    f"minD={row['min_hamming']} "
                    f"avgD={row['avg_hamming']:.2f} "
                    f"coll={row['collision_pairs_radius1']} "
                    f"poolP={row['pool_use_prob']:.2f}"
                )

        return self.best_library, self.best_metrics


# =============================================================================
# Save functions
# =============================================================================

def metrics_to_dict(m: LibraryMetrics) -> Dict[str, float]:
    return asdict(m)


def save_library_csv(path: str, library: List[str], cfg: BarcodeConfig) -> None:
    rows = []

    for i, seq in enumerate(library):
        nearest = min(
            [
                hamming(seq, other)
                for j, other in enumerate(library)
                if j != i
            ],
            default=0,
        )

        rows.append(
            {
                "index": i + 1,
                "barcode": seq,
                "length": len(seq),
                "gc_fraction": gc_fraction(seq),
                "max_homopolymer": max_homopolymer_run(seq),
                "nearest_hamming": nearest,
                "valid_gc": int(cfg.gc_min <= gc_fraction(seq) <= cfg.gc_max),
                "valid_homopolymer": int(
                    max_homopolymer_run(seq) <= cfg.homopolymer_max
                ),
            }
        )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_pairwise_matrix_csv(path: str, library: List[str]) -> None:
    _, _, matrix = pairwise_distance_stats(library)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["barcode"] + [f"B{i + 1}" for i in range(len(library))])

        for i, row in enumerate(matrix):
            writer.writerow([f"B{i + 1}"] + row)


def save_log_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return

    keys = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(
    path: str,
    method: str,
    cfg: BarcodeConfig,
    abc_cfg: ABCConfig,
    metrics: LibraryMetrics,
    runtime_s: float,
    args: argparse.Namespace,
    pool_summary: Optional[PoolSummary],
) -> None:
    data = {
        "method": method,
        "barcode_config": asdict(cfg),
        "abc_config": asdict(abc_cfg),
        "best_metrics": metrics_to_dict(metrics),
        "runtime_seconds": runtime_s,
        "args": vars(args),
        "candidate_pool_summary": (
            asdict(pool_summary) if pool_summary is not None else None
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_available": TORCH_AVAILABLE,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_pool_summary_csv(path: str, summary: Optional[PoolSummary]) -> None:
    if summary is None:
        return

    data = asdict(summary)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
        writer.writeheader()
        writer.writerow(data)


# =============================================================================
# ABC-only control
# =============================================================================

def run_abc_only_control(cfg: BarcodeConfig, abc_cfg: ABCConfig, seed: int):
    print("\n[Control] ABC-only control run")

    t0 = time.time()

    gen = ReferenceFreeCandidateGenerator(
        cfg=cfg,
        seed=seed,
        use_transformer=False,
    )

    opt = ABCBarcodeOptimizer(
        barcode_cfg=cfg,
        abc_cfg=abc_cfg,
        generator=gen,
        seed=seed + 111,
        seed_pool=None,
        seeded_initialization=False,
    )

    lib, met = opt.run()
    runtime_s = time.time() - t0

    return lib, met, opt.log_rows, runtime_s


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Reference-free Transformer candidate generation + ABC optimization "
            "for DNA barcode library design"
        )
    )

    p.add_argument("--length", type=int, default=12)
    p.add_argument("--size", type=int, default=64)

    p.add_argument("--gc-min", type=float, default=0.40)
    p.add_argument("--gc-max", type=float, default=0.60)
    p.add_argument("--gc-target", type=float, default=0.50)

    p.add_argument("--homopolymer-max", type=int, default=3)
    p.add_argument("--error-radius", type=int, default=1)
    p.add_argument("--kmer-k", type=int, default=3)

    p.add_argument("--foods", type=int, default=12)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--replace-rate", type=float, default=0.10)
    p.add_argument("--mutation-rate", type=float, default=0.30)
    p.add_argument("--local-trials", type=int, default=3)
    p.add_argument("--candidate-trials", type=int, default=8)
    p.add_argument("--print-every", type=int, default=1)

    p.add_argument("--greedy-init-ratio", type=float, default=0.40)
    p.add_argument("--stochastic-init-ratio", type=float, default=0.40)

    p.add_argument("--pool-use-start", type=float, default=0.60)
    p.add_argument("--pool-use-end", type=float, default=0.15)

    p.add_argument("--seed", type=int, default=20260428)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--transformer-block-size", type=int, default=128)

    p.add_argument("--raw-pool-size", type=int, default=1024)
    p.add_argument("--seed-pool-size", type=int, default=256)
    p.add_argument("--filter-sample-size", type=int, default=100)
    p.add_argument("--random-fill-ratio", type=float, default=0.40)
    p.add_argument("--stream-oversample", type=int, default=4)

    p.add_argument("--no-transformer", action="store_true")
    p.add_argument("--run-abc-control", action="store_true")

    p.add_argument("--out", type=str, default="reference_free_transformer_abc_results")

    return p.parse_args()


def main():
    args = parse_args()

    if not TORCH_AVAILABLE and not args.no_transformer:
        print("[WARN] Torch is not available. Falling back to constrained random generation.")
        args.no_transformer = True

    seed_everything(args.seed)
    safe_mkdir(args.out)

    cfg = BarcodeConfig(
        length=args.length,
        size=args.size,
        gc_min=args.gc_min,
        gc_max=args.gc_max,
        gc_target=args.gc_target,
        homopolymer_max=args.homopolymer_max,
        error_radius=args.error_radius,
        kmer_k=args.kmer_k,
    )

    abc_cfg = ABCConfig(
        foods=args.foods,
        iters=args.iters,
        limit=args.limit,
        replace_rate=args.replace_rate,
        mutation_rate=args.mutation_rate,
        local_trials=args.local_trials,
        candidate_trials=args.candidate_trials,
        print_every=args.print_every,
        greedy_init_ratio=args.greedy_init_ratio,
        stochastic_init_ratio=args.stochastic_init_ratio,
        pool_use_start=args.pool_use_start,
        pool_use_end=args.pool_use_end,
    )

    print("=" * 80)
    print("Reference-free Transformer candidate generation + ABC barcode optimization")
    print("=" * 80)
    print("[BarcodeConfig]", cfg)
    print("[ABCConfig]", abc_cfg)
    print(f"[Torch] available={TORCH_AVAILABLE}")
    print(f"[Output] {args.out}")
    print(f"[Transformer enabled] {not args.no_transformer}")
    print(f"[Raw pool size] {args.raw_pool_size}")
    print(f"[Selected seed pool size] {args.seed_pool_size}")

    generator = ReferenceFreeCandidateGenerator(
        cfg=cfg,
        seed=args.seed,
        use_transformer=not args.no_transformer,
        temperature=args.temperature,
        block_size=args.transformer_block_size,
    )

    selected_pool, pool_summary = build_reference_free_candidate_pool(
        generator=generator,
        cfg=cfg,
        raw_pool_size=args.raw_pool_size,
        selected_pool_size=args.seed_pool_size,
        seed=args.seed,
        random_fill_ratio=args.random_fill_ratio,
        filter_sample_size=args.filter_sample_size,
        stream_oversample=args.stream_oversample,
    )

    seed_pool = CandidateSeedPool(
        pool=selected_pool,
        cfg=cfg,
        seed=args.seed + 444,
    )

    comparison_rows = []

    if args.run_abc_control:
        control_lib, control_met, control_log, control_runtime = run_abc_only_control(
            cfg=cfg,
            abc_cfg=abc_cfg,
            seed=args.seed + 777,
        )

        comparison_rows.append(
            {
                "method": "ABC-only control",
                **metrics_to_dict(control_met),
                "runtime_seconds": control_runtime,
                "candidate_pool_runtime_seconds": 0.0,
                "total_runtime_seconds": control_runtime,
            }
        )

        save_log_csv(
            os.path.join(args.out, "run_log_abc_only_control.csv"),
            control_log,
        )

    print("\n[Proposed] Reference-free Transformer-generated candidate pool + ABC")

    t0 = time.time()

    opt = ABCBarcodeOptimizer(
        barcode_cfg=cfg,
        abc_cfg=abc_cfg,
        generator=generator,
        seed=args.seed + 111,
        seed_pool=seed_pool,
        seeded_initialization=True,
    )

    best_lib, best_met = opt.run()

    runtime_s = time.time() - t0
    total_runtime_s = runtime_s + pool_summary.runtime_total_seconds

    comparison_rows.append(
        {
            "method": "Reference-free Transformer-generated pool + ABC",
            **metrics_to_dict(best_met),
            "runtime_seconds": runtime_s,
            "candidate_pool_runtime_seconds": pool_summary.runtime_total_seconds,
            "total_runtime_seconds": total_runtime_s,
        }
    )

    save_library_csv(
        os.path.join(args.out, "best_library.csv"),
        best_lib,
        cfg,
    )

    save_pairwise_matrix_csv(
        os.path.join(args.out, "pairwise_distances.csv"),
        best_lib,
    )

    save_log_csv(
        os.path.join(args.out, "run_log.csv"),
        opt.log_rows,
    )

    save_summary_json(
        os.path.join(args.out, "best_summary.json"),
        "Reference-free Transformer-generated pool + ABC",
        cfg,
        abc_cfg,
        best_met,
        runtime_s,
        args,
        pool_summary,
    )

    save_comparison_csv(
        os.path.join(args.out, "comparison_summary.csv"),
        comparison_rows,
    )

    save_pool_summary_csv(
        os.path.join(args.out, "candidate_pool_summary.csv"),
        pool_summary,
    )

    print("\n" + "=" * 80)
    print("FINAL BEST METRICS")
    print("=" * 80)

    for k, v in metrics_to_dict(best_met).items():
        print(f"{k:28s}: {v}")

    print(f"runtime_seconds             : {runtime_s:.2f}")
    print(f"candidate_pool_runtime_s    : {pool_summary.runtime_total_seconds:.2f}")
    print(f"total_runtime_seconds       : {total_runtime_s:.2f}")

    print("\nFiles saved:")
    print(f"  {os.path.join(args.out, 'best_library.csv')}")
    print(f"  {os.path.join(args.out, 'best_summary.json')}")
    print(f"  {os.path.join(args.out, 'run_log.csv')}")
    print(f"  {os.path.join(args.out, 'comparison_summary.csv')}")
    print(f"  {os.path.join(args.out, 'pairwise_distances.csv')}")
    print(f"  {os.path.join(args.out, 'candidate_pool_summary.csv')}")

    if args.run_abc_control:
        print(f"  {os.path.join(args.out, 'run_log_abc_only_control.csv')}")


if __name__ == "__main__":
    main()
