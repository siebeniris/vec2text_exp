"""
Evaluation script for a trained inversion model.

Usage:
    python eval_inversion.py \
        --model_path saves/inverters/coco_clip \
        --victim_name clip \
        --batch_size 32 \
        --num_beams 4 \
        --max_samples 1000 \
        --output saves/inverters/coco_clip/eval_results.json
"""

import argparse
import json
import os

import nltk
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _tokenize(text: str):
    return nltk.word_tokenize(text.lower())


def bleu(predictions, references):
    """Corpus BLEU-1..4. references is a list of lists of strings (multiple refs allowed)."""
    smooth = SmoothingFunction().method1
    refs = [[_tokenize(r) for r in refs_i] for refs_i in references]
    hyps = [_tokenize(p) for p in predictions]
    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n)
        scores[f"bleu{n}"] = corpus_bleu(refs, hyps, weights=weights, smoothing_function=smooth)
    return scores


def rouge(predictions, references):
    """ROUGE-1/2/L, averaged over all samples (best ref when multiple)."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, refs_i in zip(predictions, references):
        best = {k: 0.0 for k in agg}
        for ref in refs_i:
            s = scorer.score(ref, pred)
            for k in agg:
                best[k] = max(best[k], s[k].fmeasure)
        for k in agg:
            agg[k].append(best[k])
    return {k: float(np.mean(v)) for k, v in agg.items()}


def exact_match(predictions, references):
    """Fraction of predictions that exactly match at least one reference."""
    hits = sum(
        any(pred.strip().lower() == ref.strip().lower() for ref in refs_i)
        for pred, refs_i in zip(predictions, references)
    )
    return hits / len(predictions)


def meteor(predictions, references):
    """Average METEOR score (supports multiple references per sample)."""
    scores = [
        meteor_score([_tokenize(ref) for ref in refs_i], _tokenize(pred))
        for pred, refs_i in zip(predictions, references)
    ]
    return float(np.mean(scores))


def token_f1(predictions, references):
    """Average token-level F1 (best ref), as in SQuAD."""
    scores = []
    for pred, refs_i in zip(predictions, references):
        pred_toks = set(_tokenize(pred))
        best = 0.0
        for ref in refs_i:
            ref_toks = set(_tokenize(ref))
            common = pred_toks & ref_toks
            if not common:
                continue
            p = len(common) / len(pred_toks)
            r = len(common) / len(ref_toks)
            best = max(best, 2 * p * r / (p + r))
        scores.append(best)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(victim_name: str, max_samples: int):
    """Returns (embeddings [N, D], captions [[str, ...], ...])."""
    repo_root = os.getcwd()
    embed_dir = os.path.join(repo_root, "data", "embeds", "victim_embeddings", victim_name, "test")
    caption_path = os.path.join(repo_root, "data", "victim_embeds_data", "test_dict.json")

    embeddings = np.load(os.path.join(embed_dir, "test.npy"))
    with open(os.path.join(embed_dir, "test_image_ids.json")) as f:
        image_ids = json.load(f)
    with open(caption_path) as f:
        caption_dict = json.load(f)
        caption_dict = {idx:c[0] for idx, c in caption_dict.items()}

    if max_samples > 0:
        image_ids = image_ids[:max_samples]
        embeddings = embeddings[:max_samples]

    # Normalise image_ids to strings for dict lookup
    references = []
    for iid in image_ids:
        entry = caption_dict.get(str(iid)) or caption_dict.get(int(iid))
        if entry is None:
            references.append([""])
        elif isinstance(entry, str):
            references.append([entry])
        elif isinstance(entry, dict):
            caps = entry.get("caption", entry.get("captions", [""]))
            references.append([caps] if isinstance(caps, str) else caps)
        else:
            references.append(entry if isinstance(entry, list) else [""])

    return embeddings, references


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_inversions(model, tokenizer, embeddings_np, batch_size, num_beams, max_new_tokens):
    model.eval()
    device = next(model.parameters()).device
    predictions = []

    for start in range(0, len(embeddings_np), batch_size):
        batch_emb = torch.tensor(
            embeddings_np[start: start + batch_size], dtype=torch.float32, device=device
        )
        inputs = {"frozen_embeddings": batch_emb}
        out_ids = model.generate(
            inputs=inputs,
            generation_kwargs={
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
                "early_stopping": True,
            },
        )
        decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        predictions.extend(decoded)

        done = min(start + batch_size, len(embeddings_np))
        print(f"  {done}/{len(embeddings_np)}", end="\r", flush=True)

    print()
    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to trained inverter checkpoint (e.g. saves/inverters/coco_clip)")
    parser.add_argument("--victim_name", required=True,
                        help="Victim embedding name (e.g. clip, nomic). "
                             "Must match a folder in data/embeds/victim_embeddings/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Cap number of test samples (-1 = all)")
    parser.add_argument("--output", default=None,
                        help="Path to save JSON results. Defaults to <model_path>/eval_results.json")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.model_path, "eval_results.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ---- Load model --------------------------------------------------------
    print(f"Loading model from {args.model_path} ...")
    config = InversionConfig.from_pretrained(args.model_path)
    model = InversionModel.from_pretrained(args.model_path, config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tokenizer = model.tokenizer
    print(f"Model loaded on {device}. Embedding dim: {model.embedder_dim}")

    # ---- Load test data ----------------------------------------------------
    print(f"Loading test data for victim '{args.victim_name}' ...")
    embeddings, references = load_test_data(args.victim_name, args.max_samples)
    print(f"  {len(embeddings)} test samples, embedding dim {embeddings.shape[1]}")

    # ---- Generate ----------------------------------------------------------
    print(f"Generating with beam_size={args.num_beams}, batch_size={args.batch_size} ...")
    predictions = generate_inversions(
        model, tokenizer, embeddings,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )

    # ---- Metrics -----------------------------------------------------------
    print("Computing metrics ...")
    results = {}
    results.update(bleu(predictions, references))
    results.update(rouge(predictions, references))
    results["meteor"] = meteor(predictions, references)
    results["exact_match"] = exact_match(predictions, references)
    results["token_f1"] = token_f1(predictions, references)
    results["num_samples"] = len(predictions)

    print("\n=== Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- Save --------------------------------------------------------------
    output = {
        "config": {
            "model_path": args.model_path,
            "victim_name": args.victim_name,
            "num_beams": args.num_beams,
            "max_samples": args.max_samples,
        },
        "metrics": results,
        "predictions": predictions,
        "references": [refs[0] for refs in references],  # first reference per sample
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
