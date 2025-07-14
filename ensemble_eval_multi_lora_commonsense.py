import argparse, copy, datetime, glob, json, os, re, sys
from collections import Counter

import torch, wandb
from tqdm.auto import tqdm
from datasets import load_dataset, disable_caching
from transformers import (GenerationConfig, AutoTokenizer,
                          AutoModelForCausalLM, LlamaTokenizer)

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel                                            # noqa: E402

disable_caching()
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
          else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',
        choices=["boolq", "piqa", "social_i_qa", "hellaswag",
                 "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
        required=True)
    p.add_argument('--model',
        choices=['LLaMA-7B', 'LLaMA-13B', 'BLOOM-7B', 'GPT-j-6B',
                 'Llama-3.2-1B', 'Llama-3.2-3B', 'Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct'],
        required=True)
    p.add_argument('--base_model', required=True)
    p.add_argument('--lora_weights', required=True,
        help='Comma-separated list or glob of adapter dirs')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--ensemble_rule', choices=['vote'], default='vote')
    p.add_argument('--load_8bit', action='store_true')
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def expand_paths(spec: str):
    out = []
    for part in spec.split(','):
        out.extend(sorted(glob.glob(part))) if any(ch in part for ch in "*?[") \
            else out.append(part.rstrip('/'))
    if not out:
        raise ValueError(f"No adapter paths expand from spec: {spec}")
    return out

def generate_prompt(instr: str) -> str:
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instr}\n\n### Response:\n")

def load_data(name: str):
    f = f'dataset/{name}/test.json'
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing dataset file {f}")
    return json.load(open(f))

def batches(data, bs):            # simple chunker
    return [data[i:i+bs] for i in range(0, len(data), bs)]

def load_base(args):
    Tok = LlamaTokenizer if "LLaMA" in args.model else AutoTokenizer
    tok = Tok.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    tok.pad_token_id, tok.padding_side = 0, "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True).eval()
    return tok, mdl

@torch.no_grad()
def model_generate(m, tok, instructions, max_new=256):
    prompts = [generate_prompt(i) for i in instructions]
    enc = tok(prompts, return_tensors="pt", padding=True,
              truncation=True, max_length=512).to(m.device)
    cfg = GenerationConfig(temperature=0.1, top_p=0.75, top_k=40, num_beams=4)
    out = m.generate(**enc, generation_config=cfg,
                     max_new_tokens=max_new,
                     return_dict_in_generate=True)
    txt = tok.batch_decode(out.sequences, skip_special_tokens=True)
    return [t.split("### Response:")[-1].strip() for t in txt]

# ────────────────── answer extraction (NEW LOGIC) ────────────────────────────
PLACEHOLDER_PAT = {
    'boolq':        r'\b(?:true|false)\b',
    'piqa':         r'\bsolution[12]\b',
    'social_i_qa':  r'\banswer[123]\b',
    'ARC-Challenge':r'\banswer[1-5]\b',
    'ARC-Easy':     r'\banswer[1-5]\b',
    'openbookqa':   r'\banswer[1-5]\b',
    'hellaswag':    r'\bending[1-4]\b',
    'winogrande':   r'\boption[12]\b'
}

def normalize(s: str):
    return re.sub(r'\W+', '', s.lower())

def get_options(example, dataset):
    """Return list of (placeholder, text) tuples for this example."""
    if dataset == 'boolq':
        return [('true', 'true'), ('false', 'false')]
    if dataset == 'piqa':
        return [('solution1', example.get('sol1', example.get('solution1', ''))),
                ('solution2', example.get('sol2', example.get('solution2', '')))]
    if dataset == 'social_i_qa':
        return [(f'answer{i}', example[f'answer{i}']) for i in range(1,4)]
    if dataset in ['ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        return [(f'answer{i}', example.get(f'answer{i}', ''))
                for i in range(1,6) if f'answer{i}' in example]
    if dataset == 'hellaswag':
        return [(f'ending{i}', example.get(f'ending{i}', ''))
                for i in range(1,5)]
    if dataset == 'winogrande':
        return [('option1', example.get('option1','')),
                ('option2', example.get('option2',''))]
    return []

def extract_answer(dataset: str, pred_text: str, example):
    txt = pred_text.strip().lower()
    # 1) try placeholder regex
    pat = PLACEHOLDER_PAT.get(dataset)
    if pat:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            return m.group(0).lower()
    # 2) try real option strings
    for placeholder, opt_text in get_options(example, dataset):
        if opt_text and normalize(opt_text) in normalize(txt):
            return placeholder           # return canonical id
    return ""                            # fallback: empty

# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    run_id = f"{args.model}-{os.path.basename(args.base_model)}-" \
             f"{args.dataset}-{datetime.datetime.now():%Y%m%d_%H%M%S}"
    wandb.init(project="lora_ensemble_eval", name=run_id,
               config=vars(args), reinit=True)

    data   = load_data(args.dataset)
    labels = [d['answer'] for d in data]
    bs     = args.batch_size
    batches_list = batches(data, bs)

    tokenizer, base = load_base(args)
    adapter_paths   = expand_paths(args.lora_weights)
    M = len(adapter_paths)
    votes = [ [] for _ in range(M) ]

    for a_idx, path in enumerate(adapter_paths):
        print(f"\n>> [{a_idx+1}/{M}] {path}")
        model = PeftModel.from_pretrained(base, path,
                                          torch_dtype=torch.bfloat16,
                                          device_map={'':0})
        seen = 0
        for batch in tqdm(batches_list, leave=False):
            instr = [d['instruction'] for d in batch]
            outs  = model_generate(model, tokenizer, instr)
            preds = [extract_answer(args.dataset, o, ex)
                     for o, ex in zip(outs, batch)]
            votes[a_idx].extend(preds)
            seen += len(batch)
            wandb.log({f"adapter_{a_idx+1}_acc":
                       sum(p==l for p,l in zip(votes[a_idx], labels[:seen]))/seen,
                       "samples_seen": seen})
        model.unload()      # free VRAM

    # ── majority vote ensemble ──────────────────────────────────────────
    final = []
    for i in range(len(labels)):
        tally = Counter(votes[a][i] for a in range(M))
        final.append(tally.most_common(1)[0][0])

    acc = sum(p==l for p,l in zip(final, labels)) / len(labels)
    print(f"\n==== ENSEMBLE (M={M}) | ACC={acc:.4f} ====")
    wandb.log({"ensemble_accuracy": acc})
    wandb.finish()

    # detailed json
    out = copy.deepcopy(data)
    for rec, pred in zip(out, final):
        rec["prediction"] = pred
        rec["correct"]    = (pred == rec["answer"])
    os.makedirs("experiment", exist_ok=True)
    out_path = f"experiment/{run_id}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print("Saved →", out_path)

if __name__ == "__main__":
    main()