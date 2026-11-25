"""Sample chorales from a trained checkpoint and write them as MIDI.

Supports:
    - Transformer
    - LSTM
    - Markov

Usage:
    python scripts/sample.py runs/big/best.pt --n 5
    python scripts/sample.py runs/big/best.pt --n 5 --template bwv254 --temperature 0.9

--template copies a real chorale's time signature, pickup and fermata (phrase-end) positions,
and length, so the model only has to fill in the notes. Without it: 4/4, no fermatas, --steps.
Writes runs/<run>/samples/sample_<i>.mid and samples.json (token ids, for the rule checker).
"""
import argparse
import json
from pathlib import Path

import torch

from chorales.encode import write_midi
from chorales.load import PROCESSED_DIR
from models.base import ChoraleModel
from chorales.vocab import HOLD, REST, is_pitch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=128, help="max 16th-note steps without a template")
    ap.add_argument("--template", metavar="CHORALE_ID")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChoraleModel.load(args.checkpoint, device=device)

    kw = {"time_signature": "4/4", "max_steps": args.steps}
    if args.template:
        e = json.loads((Path(PROCESSED_DIR) / "encoded.json").read_text())[args.template]
        kw = {"time_signature": e["time_signature"], "max_steps": (len(e["ids"]) - 2) // 4,
              "fermata_template": e["fermata"][1:-1:4], "start_beat_pos": e["beat_pos"][1]}

    out = Path(args.checkpoint).parent / "samples"
    out.mkdir(exist_ok=True)
    samples = []
    for i in range(args.n):
        ids = model.generate(temperature=args.temperature, seed=args.seed + i, **kw)
        samples.append(ids)
        path = write_midi(ids, out / f"sample_{i}.mid", kw["time_signature"])
        body = ids[1:-1]
        print(f"{path}  steps {len(body) // 4}  pitches {sum(map(is_pitch, body))}  "
              f"holds {body.count(HOLD)}  rests {body.count(REST)}")
    results = {
        "model": model.name,
        "args": vars(args),
        **kw,
        "samples": samples,
    }
    results["args"]["checkpoint"] = str(args.checkpoint)
    (out / "samples.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main()
