import torch
import torch.nn as nn


class SinGatedLinear(nn.Module):
    """f(x, y) = (W @ x) * alpha * sin(y)"""
    def __init__(self, in_features, out_features, alpha=1.0, learnable_alpha=True, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer('alpha', torch.tensor(float(alpha)))

    def forward(self, x, y):
        Wx = self.linear(x)
        gate = self.alpha * torch.sin(y)
        return Wx * gate


class SinGatedAttention(nn.Module):
    """
    A = Attention(x)             -- pure gate signal, no residual
    out = SinGatedLinear(x, A)   = (W @ x) * alpha * sin(A)
    """
    def __init__(self, d_model, n_heads=4, alpha=1.0, learnable_alpha=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gated = SinGatedLinear(d_model, d_model, alpha=alpha, learnable_alpha=learnable_alpha)

    def forward(self, x, attn_mask=None):
        A, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        out = self.gated(x, A)
        return out


class SinGatedLM(nn.Module):
    """
    Embedding
     -> Linear
     -> MHA (standard self-attention, causal)
     -> Linear
     -> SinGatedAttention
     -> Linear (vocab projection / LM head)
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, max_seq_len=256,
                 alpha=1.0, learnable_alpha=True):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.linear1 = nn.Linear(d_model, d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.linear2 = nn.Linear(d_model, d_model)
        self.sin_gated_attn = SinGatedAttention(d_model, n_heads=n_heads,
                                                 alpha=alpha, learnable_alpha=learnable_alpha)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        # idx: (batch, seq)
        B, T = idx.shape
        device = idx.device

        pos = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = self.embedding(idx) + self.pos_embedding(pos)  # (B, T, d_model)

        # causal mask: True = masked out (cannot attend)
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        x = self.linear1(x)

        attn_out, _ = self.mha(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out  # Residual + attention output feeds forward 

        x = self.linear2(x)

        x = self.sin_gated_attn(x, attn_mask=causal_mask)

        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx


class PlainLM(nn.Module):
    """
    Baseline for comparison:
    Embedding -> Linear -> MHA -> Linear -> Linear -> Linear (vocab)
    Same depth/param budget as SinGatedLM, but SinGatedAttention replaced
    with a plain Linear (no attention, no sin gating) so we isolate the
    effect of that one block.
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, max_seq_len=256):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.linear2 = nn.Linear(d_model, d_model)
        self.mha2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)  # replaces SinGatedAttention
        self.linear3 = nn.Linear(d_model, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.embedding(idx) + self.pos_embedding(pos)
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        x = self.linear1(x)
        attn_out, _ = self.mha(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = self.linear2(x)
        attn_out, _ = x + self.mha2(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = self.linear3(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx


if __name__ == "__main__":
    # quick shape sanity check
    model = SinGatedLM(vocab_size=65, d_model=64, n_heads=4, max_seq_len=128)
    x = torch.randint(0, 65, (4, 32))
    out = model(x)
    print("logits shape:", out.shape)  # (4, 32, 65)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")