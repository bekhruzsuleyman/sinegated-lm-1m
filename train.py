import torch
import torch.nn as nn
import time
from model import SinGatedLM, PlainLM

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ---- data ----
with open("input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 128
batch_size = 32

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, eval_iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits = model(xb)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), yb.view(-1)
            )
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_model(model, max_iters=3000, lr=3e-4, eval_interval=250, label=""):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    t0 = time.time()
    for it in range(max_iters):
        xb, yb = get_batch("train")
        logits = model(xb)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), yb.view(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - t0
            print(f"[{label}] iter {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | {elapsed:.1f}s")
            history.append((it, losses["train"], losses["val"]))
    return history


if __name__ == "__main__":
    d_model_sin = 292
    d_model_plain = 360
    n_heads = 4

    model = SinGatedLM(vocab_size=vocab_size, d_model=d_model_sin, n_heads=n_heads,
                        max_seq_len=block_size, alpha=1.0, learnable_alpha=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SinGatedLM params: {n_params:,}")

    history_sg = train_model(model, max_iters=3000, eval_interval=100, label="SinGatedLM")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=300, temperature=0.8, top_k=40)
    print("\n--- SinGatedLM sample ---")
    print(decode(generated[0].tolist()))

    torch.save(model.state_dict(), "singated_lm.pt")

    # # ---- baseline for comparison ----
    torch.manual_seed(1337)  # same init seed, same data order
    baseline = PlainLM(vocab_size=vocab_size, d_model=d_model_plain, n_heads=n_heads,
                        max_seq_len=block_size)
    n_params_baseline = sum(p.numel() for p in baseline.parameters())
    print(f"\nPlainLM params: {n_params_baseline:,}")

    history_base = train_model(baseline, max_iters=3000, eval_interval=100, label="PlainLM")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = baseline.generate(context, max_new_tokens=300, temperature=0.8, top_k=40)
    print("\n--- PlainLM sample ---")
    print(decode(generated[0].tolist()))

    torch.save(baseline.state_dict(), "plain_lm.pt")

    import json
    with open("history.json", "w") as f:
        json.dump({"singated": history_sg, "plain": history_base}, f)