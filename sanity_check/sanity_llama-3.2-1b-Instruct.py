import torch, textwrap
from transformers import LlamaTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import (GenerationConfig, AutoTokenizer,
                          AutoModelForCausalLM, LlamaTokenizer)
model_dir = "/home/models/Llama-3.2-1B-Instruct"

# 1) tokenizer  (LlamaTokenizer keeps BOS/EOS consistent)
tok = AutoTokenizer.from_pretrained(model_dir, legacy=True)
tok.pad_token_id = tok.eos_token_id            # required for generate()

# 2) load base in bf16 on GPU-0  (falls back to fp32 on cards w/o bf16)
mdl = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,            # <<< main fix
        device_map="auto",
        trust_remote_code=True).eval()

prompt = "### Instruction:\nList three prime numbers.\n\n### Response:\n"
ids = tok(prompt, return_tensors="pt").to(mdl.device)

# Greedy
out = mdl.generate(**ids, do_sample=False, max_new_tokens=32)
print("\nGREEDY:\n", textwrap.fill(tok.decode(out[0], skip_special_tokens=True), 100))

# Low-risk sampling – pass flags ONCE
out = mdl.generate(**ids,
                   do_sample=True, temperature=0.7, top_p=1.0,
                   max_new_tokens=32)
print("\nSAMPLED:\n", textwrap.fill(tok.decode(out[0], skip_special_tokens=True), 100))