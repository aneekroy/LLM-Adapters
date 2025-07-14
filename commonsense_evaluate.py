import copy
import json
import os
import re
import sys
import argparse

import fire

import torch
import datetime
import wandb

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()

    # ── Weights & Biases setup ────────────────────────────────────────────────
    run_name = f"{args.model}-{args.adapter}-{args.dataset}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="commonsense_eval",
        name=run_name,
        config=vars(args),
        reinit=True
    )

    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        print(outputs)
        return outputs

    save_file = f'experiment/{args.model}-{args.adapter}-{args.dataset}.json'
    create_dir('experiment/')

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)
    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions)

        # for data, output in zip(batch, outputs):
        #     label = data.get('answer')
        #     flag = False
        #     predict = extract_answer(args, output)
        #     if label == predict:
        #         correct += 1
        #         flag = True
        #     new_data = copy.deepcopy(data)
        #     new_data['output_pred'] = output
        #     new_data['pred'] = predict
        #     new_data['flag'] = flag
        #     output_data.append(new_data)
        #     print(data["instruction"])
        #     print(output)
        #     print('prediction:', predict)
        #     print('label:', label)

        for data, output in zip(batch, outputs):
            gold_text = get_gold_answer_text(args.dataset, data)
            pred_text = extract_answer(args, output, data)

            flag = gold_text and (normalize(pred_text) == normalize(gold_text))
            if flag:
                correct += 1

            new_data = copy.deepcopy(data)
            new_data.update({
                "output_pred": output,
                "pred": pred_text,
                "gold_text": gold_text,
                "flag": flag,
            })
            output_data.append(new_data)

        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        print('---------------')
        wandb.log({
            "batch": idx + 1,
            "running_accuracy": correct / current,
            "samples_seen": current
        }, step=current)
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print('\n')
    final_acc = correct / current if current else 0.0
    wandb.log({"final_accuracy": final_acc})
    wandb.finish()
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', "LLaMA-13B",'BLOOM-7B', 'GPT-j-6B','Llama-3.2-1B','Llama-3.2-3B'], required=True)
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel'],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit
    if "LLaMA" in args.model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float32,
            device_map={"":0}
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float32,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float32,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]

########################################################################################

def normalize(txt: str) -> str:
    return re.sub(r'\W+', '', txt.lower())


# Helper: list of (tag, actual_text) for each dataset
def get_options(data: dict, dataset: str):
    if dataset == "boolq":
        return [("true", "true"), ("false", "false")]
    if dataset == "piqa":
        return [("solution1", data.get("solution1", data.get("sol1", ""))),
                ("solution2", data.get("solution2", data.get("sol2", "")))]
    if dataset == "social_i_qa":
        return [(f"answer{i}", data.get(f"answer{i}", "")) for i in range(1, 4)]
    if dataset in ("ARC-Challenge", "ARC-Easy", "openbookqa"):
        return [(f"answer{i}", data.get(f"answer{i}", ""))
                for i in range(1, 6)]
    if dataset == "hellaswag":
        return [(f"ending{i}", data.get(f"ending{i}", ""))
                for i in range(1, 5)]
    if dataset == "winogrande":
        return [("option1", data.get("option1", "")),
                ("option2", data.get("option2", ""))]
    return []


# Resolve ground-truth tag → its real answer text
def get_gold_answer_text(dataset: str, data: dict) -> str:
    gold_tag = data.get("answer", "")
    for tag, txt in get_options(data, dataset):
        if tag == gold_tag:
            return txt.strip()
    # boolq already stores "true"/"false" directly
    return gold_tag.strip()


# Extract model prediction as real answer text
def extract_answer(args, sentence: str, example: dict) -> str:
    dataset = args.dataset
    sent_norm = normalize(sentence)

    # 1) Try placeholder regex (keeps old behaviour)
    tag_patterns = {
        "boolq": r"\b(?:true|false)\b",
        "piqa": r"\bsolution[12]\b",
        "social_i_qa": r"\banswer[123]\b",
        "ARC-Challenge": r"\banswer[1-5]\b",
        "ARC-Easy": r"\banswer[1-5]\b",
        "openbookqa": r"\banswer[1-5]\b",
        "hellaswag": r"\bending[1-4]\b",
        "winogrande": r"\boption[12]\b",
    }
    pat = tag_patterns.get(dataset)
    if pat:
        m = re.search(pat, sentence, flags=re.IGNORECASE)
        if m:
            tag = m.group(0).lower()
            for t, txt in get_options(example, dataset):
                if t == tag:
                    return txt.strip()

    # 2) Fallback: scan actual option texts
    for tag, txt in get_options(example, dataset):
        if txt and normalize(txt) in sent_norm:
            return txt.strip()

    return ""  # no match


########################################################################################
if __name__ == "__main__":
    main()
