
"""Train and evaluate N-gram Markov models for Bach chorale generation.

Usage:
    python scripts/train_markov.py
    python scripts/train_markov.py --orders 1 2 3 4
    python scripts/train_markov.py --orders 2 3 --alpha 0.1
    python scripts/train_markov.py --name markov_experiment

Writes runs/<name>/:
    best.pkl
    order_<n>.pkl
    history.json
    config.json

Metrics:
    Mean per-token negative log-likelihood (nats).
    Top-1 token prediction accuracy.
"""

import argparse
import json
import math
import random
import time
import torch
import numpy as np
from pathlib import Path

from chorales.dataset import ChoraleDataset
from models.factory import ModelFactory


RUNS = Path(__file__).resolve().parents[1] / "runs"
FEATURES = ("voice", "beat_pos", "fermata")


def evaluate(model, dataset):
    """Calculate mean per-token NLL and top-1 accuracy."""

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for chorale in dataset.chorales.values():
        sequence = chorale["ids"]

        for i in range(1, len(sequence)):
            context = sequence[:i]
            target = sequence[i]

            probabilities = model.predict(context)
            probability = probabilities[target]
            prediction = max(probabilities, key=probabilities.get)

            total_loss -= math.log(max(probability, 1e-12))
            total_correct += int(prediction == target)
            total_tokens += 1

    return (
        total_loss / total_tokens if total_tokens else 0.0,
        total_correct / total_tokens if total_tokens else 0.0,
    )



def main():

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--name", default="markov")
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--vocab-size", type=int, default=51)
    ap.add_argument("--backoff", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if any(order < 0 for order in args.orders):
        ap.error("Markov orders must be non-negative")

    if args.alpha <= 0:
        ap.error("Alpha must be positive")

    # Reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Output directory.
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)

    train_ds = ChoraleDataset("train", train=False)
    val_ds = ChoraleDataset("val", train=False)


    config = {
        **vars(args),
        "train_size": len(train_ds),
        "val_size": len(val_ds)
    }

    (out / "config.json").write_text(
        json.dumps(config, indent=2)
    )

    history = []
    best_nll = float("inf")

    # Train each Markov configuration.
    for order in args.orders:

        print(f"\nTraining order-{order} Markov model...")

        start_time = time.time()

        model = ModelFactory.create(
            "markov",
            order=order,
            vocab_size=args.vocab_size,
            alpha=args.alpha,
            backoff=args.backoff,
        )

        train_sequences = [chorale["ids"] for chorale in train_ds.chorales.values()]
        # debug(train_ds, train_sequences)
        model.fit(train_sequences)

        training_time = time.time() - start_time

        # Evaluate on training and validation data.
        eval_start = time.time()

        train_nll, train_acc = evaluate(model, train_ds)
        val_nll, val_acc = evaluate(model, val_ds)

        evaluation_time = time.time() - eval_start

        # Save model.
        model_path = out / f"order_{order}.pt"
        model.save(model_path)

        result = {
            "order": order,
            "alpha": args.alpha,
            "backoff": args.backoff,
            "params": model.n_params(),
            "param_breakdown": model.param_breakdown(),
            "train_nll": train_nll,
            "train_accuracy": train_acc,
            "val_nll": val_nll,
            "val_accuracy": val_acc,
            "training_time": training_time,
            "evaluation_time": evaluation_time,
        }

        history.append(result)

        print(f"\nModel: {model.name} (order {order})")
        print(f"Parameters: {model.n_params():,}")

        print(
            f"  train NLL {train_nll:.4f}  "
            f"accuracy {train_acc * 100:.2f}%"
        )

        print(
            f"  val   NLL {val_nll:.4f}  "
            f"accuracy {val_acc * 100:.2f}%"
        )

        print(
            f"  train {training_time:.2f}s  "
            f"eval {evaluation_time:.2f}s"
        )

        # Save progress.
        (out / "history.json").write_text(
            json.dumps(history, indent=2)
        )

        # Keep the best validation model.
        if val_nll < best_nll:

            best_nll = val_nll

            model.save(out / "best.pt")

            best_config = {
                "model_name": model.name,
                "model_config": {
                    "order": order,
                    "vocab_size": args.vocab_size,
                    "alpha": args.alpha,
                    "backoff": args.backoff,
                },
                "val_nll": val_nll,
            }

            (out / "best_config.json").write_text(
                json.dumps(best_config, indent=2)
            )

    # Final report.
    best = min(history, key=lambda x: x["val_nll"])

    print("\nTraining complete!")

    print(
        f"\nBest model: order {best['order']}"
        f"\nTrain NLL: {best['train_nll']:.4f}"
        f"\nTrain accuracy: {best['train_accuracy'] * 100:.2f}%"
        f"\nValidation NLL: {best['val_nll']:.4f}"
        f"\nValidation accuracy: {best['val_accuracy'] * 100:.2f}%"
        f"\nStored transitions: {best['params']:,}"
    )

    print(f"\nBest checkpoint: {out / 'best.pkl'}")
    print(f"Results saved to {out}")


def debug(train_ds, train_sequences):
    print("train_ds type:", type(train_ds))
    print("chorales type:", type(train_ds.chorales))
    print("Number of chorales:", len(train_ds.chorales))

    first = next(iter(train_ds.chorales.items()))
    print("First chorale ID:", first[0])
    print("First chorale data type:", type(first[1]))
    print("First chorale keys:", first[1].keys())
    print("IDs type:", type(first[1]["ids"]))
    print("IDs first 10:", first[1]["ids"][:10])

    print("train_sequences type:", type(train_sequences))
    print("First sequence type:", type(train_sequences[0]))
    print("First sequence:", train_sequences[0][:10])

if __name__ == "__main__":
    main()
