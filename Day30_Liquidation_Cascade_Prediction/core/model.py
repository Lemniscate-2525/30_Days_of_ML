import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight

def precompute_rope(dim, sq_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(sq_len, dtype=torch.float32)
    return torch.polar(torch.ones_like(torch.outer(t, inv_freq)), torch.outer(t, inv_freq))

def apply_rope(x, freqs_cis):
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.shape[1], 1, x_.shape[-1]).to(x.device)
    return torch.view_as_real(x_ * freqs_cis).flatten(3).type_as(x)

class SwiGLU(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, h_dim, bias=False)
        self.w2 = nn.Linear(dim, h_dim, bias=False)
        self.w3 = nn.Linear(h_dim, dim, bias=False)

    def forward(self, x): return self.w3(F.silu(self.w1(x)) * self.w2(x))

class LlamaBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.ffn = SwiGLU(dim, dim * 4)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def forward(self, x, freqs):
        h = self.norm1(x)
        B, T, C = h.shape
        q = self.wq(h).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(h).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(h).view(B, T, self.n_heads, self.head_dim)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        h = x + self.wo(attn.transpose(1, 2).contiguous().view(B, T, C))
        return h + self.ffn(self.norm2(h))

class LlamaModel(nn.Module):
    def __init__(self, vocab_size_in, d_model_in, num_layers_in, num_heads_in, seq_len_in):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size_in, d_model_in)
        self.layers = nn.ModuleList([LlamaBlock(d_model_in, num_heads_in) for _ in range(num_layers_in)])
        self.norm = RMSNorm(d_model_in)
        self.out = nn.Linear(d_model_in, 1)
        self.freqs = precompute_rope(d_model_in // num_heads_in, seq_len_in)

    def forward(self, x):
        h = self.tok_emb(x)
        freqs = self.freqs[:x.shape[1]].to(x.device)
        for layer in self.layers:
            h = layer(h, freqs)
        return self.out(self.norm(h)[:, -1, :]).squeeze(-1)
