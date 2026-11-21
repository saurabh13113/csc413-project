"""Train the chorale Transformer with teacher forcing.

Usage:
    python scripts/train.py --name base                     # full run on train, eval on val
    python scripts/train.py --name overfit --overfit bwv269 --dropout 0 --epochs 300 --warmup 20
    python scripts/train.py --name small --d-model 128 --layers 4   # any hyperparameter is a flag

Writes runs/<name>/: best.pt (lowest val NLL), history.json, curve.png, config.json.
`evaluate(model, loader, device)` is the shared NLL metric; baselines should import it.
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
import torch
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chorales.dataset import IGNORE_INDEX, ChoraleDataset, collate_fn  # noqa: E402
from chorales.model import ChoraleTransformer, n_params, param_breakdown  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"
FEATURES = ("voice", "beat_pos", "fermata")


def forward_loss(model, batch, device):
    """Summed cross-entropy and number of target tokens for one batch."""
    b = {k: batch[k].to(device) for k in ("input_ids", "target_ids", *FEATURES)}
    logits = model(b["input_ids"], b["voice"], b["beat_pos"], b["fermata"])
    loss = nn.functional.cross_entropy(logits.flatten(0, 1).float(), b["target_ids"].flatten(),
                                       ignore_index=IGNORE_INDEX, reduction="sum")
    return loss, (b["target_ids"] != IGNORE_INDEX).sum()


@torch.no_grad()
def evaluate(model, loader, device):
    """Mean per-token NLL (nats) over every target in the loader."""
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss, n = forward_loss(model, batch, device)
        total, count = total + loss.item(), count + n.item()
    return total / count


def plot(history, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot([h["epoch"] for h in history], [h["train_nll"] for h in history], label="train")
    ax.plot([h["epoch"] for h in history], [h["val_nll"] for h in history], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("per-token NLL (nats)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=300, help="linear warmup steps, then cosine decay")
    ap.add_argument("--augment-mode", default="sample", choices=["sample", "all"])
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--overfit", metavar="CHORALE_ID", help="train and evaluate on this one train chorale")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)

    if args.overfit:
        train_ds = ChoraleDataset("train", args.max_len, train=False)  # no augmentation
        train_ds.items = [it for it in train_ds.items if it[0] == args.overfit]
        if not train_ds.items:
            raise SystemExit(f"{args.overfit} is not in the train split")
        val_ds = train_ds
    else:
        train_ds = ChoraleDataset("train", args.max_len, args.augment_mode, seed=args.seed)
        val_ds = ChoraleDataset("val", args.max_len)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, args.batch_size, collate_fn=collate_fn)

    model = ChoraleTransformer(args.d_model, args.layers, args.heads, args.d_ff, args.dropout, args.max_len).to(device)
    print(f"params {n_params(model):,}", param_breakdown(model))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / args.warmup,
                                              0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1)))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    config = {**vars(args), "params": n_params(model), "device": str(device)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    history, best = [], float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_dl:
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss, n = forward_loss(model, batch, device)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss / n).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, count = total + loss.item(), count + n.item()
        if epoch % args.eval_every and epoch != args.epochs:
            continue
        val = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "step": epoch * len(train_dl), "train_nll": total / count,
                        "val_nll": val, "lr": sched.get_last_lr()[0]})
        print(f"epoch {epoch:4d}  train {total / count:.4f}  val {val:.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
        if val < best:
            best = val
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "val_nll": val}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
        plot(history, out / "curve.png")
    print(f"best val NLL {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
