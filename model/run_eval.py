import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd
import argparse
import json
import re
from prompts import build_prompt_messages

def parse_evaluation(xml_string):
    root = ET.fromstring(xml_string)
    return {child.tag: child.text for child in root}

def get_eval_resume(example):
    return parse_evaluation(example["evaluation"])["eval_resume"]

def get_eval_selfintro(example):
    return parse_evaluation(example["evaluation"]).get("eval_selfintro", "")

def get_ground_truth(example):
    eval_resume = get_eval_resume(example)
    eval_selfintro = get_eval_selfintro(example)
    return {
        "eval_resume": eval_resume,
        "eval_selfintro": eval_selfintro,
    }

def remove_asterisks(text):
    return text.replace("**", "")


def parse_key_value(text):
    """
    key: value 형태로 JSON 변환
    """
    data = {}
    lines = text.strip().split('\n')
    current_key = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ':' in line and not line.startswith(" "):  # 키:값 시작
            key, val = line.split(':', 1)
            current_key = key.strip()
            data[current_key] = val.strip()
        elif current_key:
            data[current_key] += ' ' + line  # 다음 줄 붙이기
    return data

# 전체 함수
def convert_markdown_to_json(text):
    clean_text = remove_asterisks(text)
    parsed = parse_key_value(clean_text)
    return parsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_model_path", type=str, default="./model-output/qlora-output_v2_explicit_guideline_curriculum/stage_2")
    parser.add_argument("--prompt_version", type=str, default="v2_explicit_guidelines_ko_improved_formated")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default="small_resume_dataset_final")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model.eval()

    dataset = load_from_disk(args.dataset_path)
    test_dataset = dataset["test"]

    if args.num_samples:
        test_dataset = test_dataset.shuffle(seed=42).select(range(args.num_samples))

    messages_list = [build_prompt_messages(example, args.prompt_version, None) for example in test_dataset]
    ground_truthes = [get_ground_truth(example) for example in test_dataset]
    resumes = [example["resume"] for example in test_dataset]
    selfintros = [example.get("selfintro", "") for example in test_dataset]

    resume_results = []
    selfintro_results = []

    for messages, gt, resume, selfintro in tqdm(zip(messages_list, ground_truthes, resumes, selfintros), total=len(messages_list), desc="Generating"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False
            )

        prompt_len = input_ids.size(1)
        generated_text = tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True).strip()

        print(f"Generated text: {generated_text}")

        json_output = convert_markdown_to_json(generated_text)
        print(f"Parsed JSON output: {json_output}")
        resume_results.append({
            "eval_resume": gt["eval_resume"],
            "generated_text": json_output.get("eval_resume", ""),
            "resume": resume
        })
        selfintro_results.append({
            "eval_resume": gt["eval_selfintro"], 
            "generated_text": json_output.get("eval_selfintro", ""), 
            "selfintro": selfintro
        })

    if args.output_csv is None:
        args.output_csv = f"eval_results_{args.prompt_version or 'default'}"

    pd.DataFrame(resume_results).to_csv(f"{args.output_csv}_resume.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selfintro_results).to_csv(f"{args.output_csv}_selfintro.csv", index=False, encoding="utf-8-sig")

    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()


