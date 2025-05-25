import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd

# EXAONE tokenizer & model load
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# QLoRA 가중치 적용 (optional)
model = PeftModel.from_pretrained(base_model, "./qlora-exaone-3.5-model")
model.eval()

# Dataset 불러오기
dataset = load_from_disk("small_resume_dataset")
# dataset["test"] = dataset["test"].shuffle(seed=42).select(range(10))

# 평가 결과 파싱 함수
def parse_evaluation(xml_string):
    root = ET.fromstring(xml_string)
    return {child.tag: child.text for child in root}

# selfintro 원본 보관용
def get_selfintro(example):
    return parse_evaluation(example["evaluation"])["eval_selfintro"]

# Chat template message 구성 함수
def build_messages(example):
    return [
        {"role": "system", "content": "You are a helpful assistant for resume evaluation."},
        {"role": "user", "content": f"[Self-Introduction]\n{example['selfintro']}"}
    ]

# Prompt 생성 및 Tokenization
messages_list = [build_messages(example) for example in dataset["test"]]
selfintro_bases = [get_selfintro(example) for example in dataset["test"]]

# 결과 저장용
results = []

for messages, gt in tqdm(zip(messages_list, selfintro_bases), total=len(messages_list), desc="Generating"):
    # Chat 템플릿 적용
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
    results.append({"selfintro": gt, "generated_text": generated_text})

# CSV 저장
pd.DataFrame(results).to_csv("exaone-3.5_simple_eval.csv", index=False, encoding="utf-8-sig")
