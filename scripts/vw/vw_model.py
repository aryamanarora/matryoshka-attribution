"""The 1L transformer of Turner, Wu & Batson (2026) and its virtual-weight decomposition.

Architecture is quoted from the note's Appendix > Training details > Transformer:

  "a one-layer, decoder-only transformer with residual width d_m = 256, 4 attention heads
   of dimension d_head = 64, a width-1024 MLP with ReLU activation, a 4,096-token
   vocabulary, and a 1,024-token context window, totaling ~2.9M parameters (~0.79M
   excluding the embedding and unembedding matrices). The model contains no normalization
   layers and no bias terms anywhere.

   Attention is causal with standard softmax; scores are scaled by 1/sqrt(64). Position
   information enters solely through a fixed (non-learned) additive sinusoidal position
   embedding (an interleaved sin/cos table with maximum wavelength 2^16) added to the
   residual stream immediately after the token embedding. Weights were initialized from a
   Gaussian with standard deviation 0.6/sqrt(d_m) ~= 0.0375."

`n_params()` checks the two published counts (2,883,584 and 786,432) as an assertion, so a
silent architecture drift fails loudly rather than producing plausible numbers.

WHAT IS UNDERDETERMINED BY THE SOURCE: the learning rate and any LR schedule are not
stated anywhere in the note (β, weight decay, clipping, batch, steps and token count all
are). scripts/vw/vw_train.py sweeps it and pins the choice by the note's own published
losses -- train ~3.33 at the final step, test 3.38.

THE VIRTUAL WEIGHT DECOMPOSITION. Because there is no normalization, the logits split
exactly into three additive paths, and the direct one is a fixed matrix product:

    logits = x @ W_U  +  attn_out @ W_U  +  mlp_out @ W_U ,   x = W_E[tok] + P[pos]
    direct = W_TL[tok] + W_TL[4096 + pos] ,   W_TL := [W_E ; P] @ W_U

`W_TL` is the note's **Tokens->Logits** family: d_v' x d_v = 5120 x 4096 = 20,971,520
virtual weights, indexed by a [vocabulary, position] source axis and a logit target. It is
the one family that needs no transcoder, whose helpfulness has an exact closed form (see
vw_scores.py), and which the note singles out as the cleanest evidence for interference --
"without the risk that the story is complicated by an imperfect extraction of features
from superposition".
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL, N_HEADS, D_HEAD, D_MLP, VOCAB, CTX = 256, 4, 64, 1024, 4096, 1024
MAX_WAVELENGTH = 2 ** 16
INIT_STD = 0.6 / math.sqrt(D_MODEL)


def sinusoid(ctx=CTX, d=D_MODEL, max_wavelength=MAX_WAVELENGTH):
    """Interleaved sin/cos, longest wavelength 2^16 -- the Vaswani table with base 2^16.

    The note fixes the interleaving and the maximum wavelength and nothing else, so we use
    the standard construction at that base; `d/2` frequencies geometrically spaced from 1
    down to 1/max_wavelength, sin in the even channels and cos in the odd ones."""
    pos = torch.arange(ctx, dtype=torch.float32)[:, None]
    inv = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(max_wavelength) / d))
    pe = torch.zeros(ctx, d)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1::2] = torch.cos(pos * inv)
    return pe


class TinyLM(nn.Module):
    def __init__(self, vocab=VOCAB, ctx=CTX, d_model=D_MODEL, n_heads=N_HEADS,
                 d_head=D_HEAD, d_mlp=D_MLP, seed=0):
        super().__init__()
        self.vocab, self.ctx, self.d_model, self.n_heads, self.d_head = vocab, ctx, d_model, n_heads, d_head
        g = torch.Generator().manual_seed(seed)

        def p(*shape):
            return nn.Parameter(torch.randn(*shape, generator=g) * INIT_STD)

        self.W_E = p(vocab, d_model)
        self.W_Q, self.W_K, self.W_V = (p(n_heads, d_model, d_head) for _ in range(3))
        self.W_O = p(n_heads, d_head, d_model)
        self.W_in, self.W_out = p(d_model, d_mlp), p(d_mlp, d_model)
        self.W_U = p(d_model, vocab)
        self.register_buffer("P", sinusoid(ctx, d_model), persistent=False)

    def n_params(self, check=True):
        tot = sum(x.numel() for x in self.parameters())
        core = tot - self.W_E.numel() - self.W_U.numel()
        if check:                       # the note's "~2.9M" / "~0.79M"
            assert (tot, core) == (2_883_584, 786_432), (tot, core)
        return tot, core

    def paths(self, tok):
        """Return (direct, attn, mlp) contributions to the logits; they sum to the logits.

        No normalization anywhere, so this split is exact, not a linearization."""
        B, T = tok.shape
        x = self.W_E[tok] + self.P[:T]
        q = torch.einsum("btm,hmd->bhtd", x, self.W_Q)
        k = torch.einsum("btm,hmd->bhtd", x, self.W_K)
        v = torch.einsum("btm,hmd->bhtd", x, self.W_V)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(self.d_head))
        attn = torch.einsum("bhtd,hdm->btm", z, self.W_O)
        mlp = F.relu((x + attn) @ self.W_in) @ self.W_out
        return x @ self.W_U, attn @ self.W_U, mlp @ self.W_U

    def forward(self, tok):
        return sum(self.paths(tok))

    def attention(self, tok):
        """The (B, H, T, T) pattern -- needed by the OV/QK families, kept here so there is
        one definition of the attention computation."""
        B, T = tok.shape
        x = self.W_E[tok] + self.P[:T]
        q = torch.einsum("btm,hmd->bhtd", x, self.W_Q)
        k = torch.einsum("btm,hmd->bhtd", x, self.W_K)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=tok.device).triu(1), -torch.inf)
        return s.softmax(-1)

    @torch.no_grad()
    def W_TL(self):
        """The Tokens->Logits virtual weight family, [W_E ; P] @ W_U -- (5120, 4096)."""
        return torch.cat([self.W_E, self.P], 0) @ self.W_U


def source_index(tok, ctx_offset=VOCAB):
    """The two [vocabulary, position] source rows active at each position: (B, T, 2)."""
    B, T = tok.shape
    pos = ctx_offset + torch.arange(T, device=tok.device).expand(B, T)
    return torch.stack([tok, pos], -1)


def loss_from_logits(logits, tok):
    """Next-token cross entropy, mean over the T-1 predicted positions."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                           tok[:, 1:].reshape(-1).long())
