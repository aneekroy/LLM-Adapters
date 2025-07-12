#!/usr/bin/env python
"""
active_learning.py – adds an uncertainty-sampling loop around finetune.py
Usage example (2 AL rounds, 10 % → 30 % of data):
python active_learning.py \
    --base_model /path/llama \
    --data_path /path/data.json \
    --output_dir ./exp \
    --rounds 2 --init_frac 0.1 --acq_frac 0.2
"""
import os, math, json, copy, torch, transformers, fire
from datasets import load_dataset
from finetune import train as finetune_train   # ← your existing function

# ───────────── acquisition helpers ──────────────
def _batched_loss(model, tokenizer, ds, cutoff_len, batch_size=8):
    """Return per-example negative-log-likelihood (higher ⇒ more uncertain)."""
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         collate_fn=collator, shuffle=False)
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            # mean loss per token × sequence length  → NLL per sequence
            seq_loss = (out.loss.detach() * batch["input_ids"].ne(tokenizer.pad_token_id).sum(1))
            losses.extend(seq_loss.cpu().tolist())
    return losses

def _select_top_k(pool_ds, scores, k):
    idx = torch.tensor(scores).topk(k).indices.tolist()
    sel = pool_ds.select(idx)
    rem = pool_ds.select(sorted(set(range(len(pool_ds))) - set(idx)))
    return sel, rem
# ────────────────────────────────────────────────

def active_learning(base_model:str,
                    data_path:str,
                    output_dir:str="./al-run",
                    rounds:int=5,
                    init_frac:float=0.1,
                    acq_frac:float=0.1,
                    cutoff_len:int=256,
                    **finetune_kwargs):
    """Pool-based AL with uncertainty sampling (loss)."""
    full = load_dataset("json", data_files=data_path)["train"]
    assert 0 < init_frac < 1 and 0 < acq_frac < 1

    # Step 0 – seed labelled set
    init_n = math.ceil(len(full)*init_frac)
    labelled, pool = full.train_test_split(test_size=len(full)-init_n, shuffle=True, seed=42).values()
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    for r in range(rounds):
        print(f"\n=== Active-learning round {r} | labelled={len(labelled)} ===")
        round_out = os.path.join(output_dir, f"round_{r}")
        os.makedirs(round_out, exist_ok=True)

        # ─── train on current labelled set ───
        tmp_json = os.path.join(round_out, "labelled.json")
        with open(tmp_json, "w") as f: json.dump(labelled.to_list(), f, indent=2)
        finetune_train(base_model=base_model,
                       data_path=tmp_json,
                       output_dir=round_out,
                       cutoff_len=cutoff_len,
                       **finetune_kwargs)

        # ─── stop if pool is empty ───
        if len(pool) == 0 or r == rounds-1:
            print("AL loop finished.")
            break

        # ─── compute uncertainty on pool ───
        ckpt = round_out    # latest adapter directory
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True)
        model = finetune_kwargs.get("prepare_model_for_int8_training", lambda x, **_:x)(model)
        adapters_state = torch.load(os.path.join(ckpt,"adapter_model.bin"))
        from peft import set_peft_model_state_dict
        model = set_peft_model_state_dict(model, adapters_state)

        scores = _batched_loss(model, tokenizer,
                               pool.map(lambda x: finetune_kwargs.get("generate_and_tokenize_prompt")(x)),
                               cutoff_len=cutoff_len)
        k = min(math.ceil(len(full)*acq_frac), len(pool))
        acquired, pool = _select_top_k(pool, scores, k)
        labelled = transformers.concatenate_datasets([labelled, acquired])

if __name__ == "__main__":
    fire.Fire(active_learning)