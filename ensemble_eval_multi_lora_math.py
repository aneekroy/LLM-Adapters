import argparse, copy, datetime, glob, json, os, re, sys
from collections import Counter, defaultdict
from statistics import median

import torch, wandb
from tqdm.auto import tqdm
from datasets import disable_caching, load_dataset
from transformers import (GenerationConfig, AutoModelForCausalLM,
                          AutoTokenizer, LlamaTokenizer)

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel  # noqa: E402

disable_caching()
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)


# ───────────────────────────────── argument parsing ──────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=["AddSub", "MultiArith", "SingleEq", "gsm8k", "AQuA", "SVAMP"],
        required=True,
    )
    p.add_argument(
        "--model",
        choices=[
            "LLaMA-7B",
            "BLOOM-7B",
            "GPT-j-6B",
            "Llama-3.2-1B",
            "Llama-3.2-3B",
        ],
        required=True,
    )
    p.add_argument("--base_model", required=True)
    p.add_argument(
        "--lora_weights",
        required=True,
        help="Comma-separated list or glob of LoRA adapter directories",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--ensemble_rule",
        choices=["vote", "median"],
        default="vote",
        help="How to aggregate numeric predictions. 'vote' counts exact matches within tolerance; 'median' uses median value.",
    )
    p.add_argument("--tolerance", type=float, default=1e-3, help="Numeric equality tolerance")
    p.add_argument("--load_8bit", action="store_true")
    return p.parse_args()


# ───────────────────────────────── helper functions ─────────────────────────────────

def expand_paths(spec: str):
    out = []
    for part in spec.split(","):
        if any(ch in part for ch in "*?["):
            out.extend(sorted(glob.glob(part)))
        else:
            out.append(part.rstrip("/"))
    if not out:
        raise ValueError(f"No adapter paths expand from spec: {spec}")
    return out


def generate_prompt(instr: str) -> str:
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instr}\n\n### Response:\n"
    )


def load_data(name: str):
    path = f"dataset/{name}/test.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file {path}")
    return json.load(open(path))


def batches(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i : i + bs]


def load_base(args):
    Tok = LlamaTokenizer if "LLaMA" in args.model else AutoTokenizer
    tok = Tok.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    tok.pad_token_id, tok.padding_side = 0, "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return tok, mdl


@torch.no_grad()
def model_generate(model, tok, instructions, max_new=256):
    prompts = [generate_prompt(i) for i in instructions]
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)
    cfg = GenerationConfig(temperature=0.1, top_p=0.75, top_k=40, num_beams=4)
    out = model.generate(
        **enc,
        generation_config=cfg,
        max_new_tokens=max_new,
        return_dict_in_generate=True,
    )
    txt = tok.batch_decode(out.sequences, skip_special_tokens=True)
    return [t.split("### Response:")[-1].strip() for t in txt]


NUM_REGEX = re.compile(r"-?\d+\.?\d*")


def extract_answer(args, text: str):
    """Return prediction depending on dataset type. For AQuA returns option letter, else float('inf') if none."""
    text = text.replace(",", "")
    if args.dataset.lower() == "aqua":
        m = re.search(r"[ABCDE]", text)
        return m.group(0) if m else ""
    # numeric datasets
    nums = NUM_REGEX.findall(text)
    if not nums:
        return float("inf")
    try:
        return float(nums[-1])
    except ValueError:
        return float("inf")


def numeric_equal(a: float, b: float, tol: float):
    return abs(a - b) <= tol


# ──────────────────────────────────── main ─────────────────────────────────────

def main():
    args = parse_args()

    run_id = f"{args.model}-{os.path.basename(args.base_model)}-{args.dataset}-{datetime.datetime.now():%Y%m%d_%H%M%S}"
    wandb.init(project="lora_math_ensemble_eval", name=run_id, config=vars(args), reinit=True)

    data = load_data(args.dataset)
    labels = [d["answer"] for d in data]
    # cast numeric labels to float where needed
    if args.dataset.lower() != "aqua":
        labels = [float(x) if not isinstance(x, (int, float)) else x for x in labels]

    tokenizer, base_model = load_base(args)
    adapter_paths = expand_paths(args.lora_weights)
    M = len(adapter_paths)

    # votes[a_idx][i] = prediction of adapter a_idx on example i
    votes = [[None] * len(data) for _ in range(M)]

    for a_idx, path in enumerate(adapter_paths):
        print(f"\n>> [{a_idx+1}/{M}] {path}")
        model = PeftModel.from_pretrained(base_model, path, torch_dtype=torch.float16, device_map={"": 0})
        seen = 0
        for batch in tqdm(list(batches(data, args.batch_size)), leave=False):
            instr = [d["instruction"] for d in batch]
            outs = model_generate(model, tokenizer, instr)
            preds = [extract_answer(args, o) for o in outs]
            for j, p in enumerate(preds):
                votes[a_idx][seen + j] = p
            seen += len(batch)

            # running accuracy for this adapter
            if args.dataset.lower() == "aqua":
                corr = sum(
                    p == lbl for p, lbl in zip(votes[a_idx][:seen], labels[:seen])
                )
            else:
                corr = sum(
                    numeric_equal(p, lbl, args.tolerance)
                    for p, lbl in zip(votes[a_idx][:seen], labels[:seen])
                )
            wandb.log({f"adapter_{a_idx+1}_acc": corr / seen, "samples_seen": seen})
        # free vram
        del model
        torch.cuda.empty_cache()

    # ── ensemble aggregation ───────────────────────────────────────────────
    final_preds = []
    for i in range(len(data)):
        pred_list = [votes[a][i] for a in range(M)]

        if args.dataset.lower() == "aqua":
            # majority vote on letters
            tally = Counter(pred_list)
            final_preds.append(tally.most_common(1)[0][0])
        else:
            # numeric aggregation
            if args.ensemble_rule == "median":
                numerics = [p for p in pred_list if isinstance(p, (int, float)) and p != float("inf")]
                final_preds.append(median(numerics) if numerics else float("inf"))
            else:  # vote within tolerance
                bins: defaultdict[float, int] = defaultdict(int)
                for p in pred_list:
                    if not isinstance(p, (int, float)) or p == float("inf"):
                        continue
                    placed = False
                    for key in list(bins.keys()):
                        if numeric_equal(p, key, args.tolerance):
                            bins[key] += 1
                            placed = True
                            break
                    if not placed:
                        bins[p] = 1
                if bins:
                    best = max(bins.items(), key=lambda x: x[1])[0]
                    final_preds.append(best)
                else:
                    final_preds.append(float("inf"))

    # accuracy
    if args.dataset.lower() == "aqua":
        acc = sum(p == l for p, l in zip(final_preds, labels)) / len(labels)
    else:
        acc = sum(
            numeric_equal(p, l, args.tolerance) for p, l in zip(final_preds, labels)
        ) / len(labels)

    print(f"\n==== ENSEMBLE (M={M}) | ACC={acc:.4f} ====\n")
    wandb.log("ensemble_accuracy", acc)

    # save detailed predictions
    out = copy.deepcopy(data)
    for rec, pred in zip(out, final_preds):
        rec["prediction"] = pred
        if args.dataset.lower() == "aqua":
            rec["correct"] = pred == rec["answer"]
        else:
            rec["correct"] = numeric_equal(pred, float(rec["answer"]), args.tolerance)

    os.makedirs("experiment", exist_ok=True)
    out_path = f"experiment/{run_id}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print("Saved →", out_path)

    wandb.finish()


if __name__ == "__main__":
    main()
