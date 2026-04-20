import torch
import torch.nn as nn
import torch.nn.functional as F

# Llama Architecture : 
class RMSNorm(nn.Module):

    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
        return norm * self.weight
      
# ROPE Embeddings : 
def precompute_rope_angles(dim, max_seq_len, theta = 10000.0):

    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype = torch.float32)

    freqs = torch.outer(t, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis

def apply_rope(x, freqs_cis):

    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.shape[1], 1, x_.shape[-1]).to(x.device)

    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)

class SwiGLU(nn.Module):  # Swish Gelu Activation
    def __init__(self, dim, hidden_dim):

        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias = False)
        self.w2 = nn.Linear(dim, hidden_dim, bias = False)
        self.w3 = nn.Linear(hidden_dim, dim, bias = False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class LlamaAttention(nn.Module):

    def __init__(self, dim, n_heads):

        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, dim, bias = False)
        self.wk = nn.Linear(dim, dim, bias = False)
        self.wv = nn.Linear(dim, dim, bias = False)
        self.wo = nn.Linear(dim, dim, bias = False)

    def forward(self, x, freqs_cis):

        B, T, C = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

class LlamaBlock(nn.Module):
    def __init__(self, dim, n_heads):

        super().__init__()

        self.attention = LlamaAttention(dim, n_heads)
        self.feed_forward = SwiGLU(dim, dim * 4)

        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, freqs_cis):

        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class LlamaModel(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, seq_len):

        super().__init__()

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([LlamaBlock(dim, n_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        
        self.output = nn.Linear(dim, 1)
        self.freqs_cis = precompute_rope_angles(dim // n_heads, seq_len)

    def forward(self, tokens):

        x = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:tokens.shape[1]].to(x.device)
        
        for layer in self.layers:
            x = layer(x, freqs_cis)
            
        x = self.norm(x)
        final_token_state = x[:, -1, :] 

        return self.output(final_token_state).squeeze(-1)
