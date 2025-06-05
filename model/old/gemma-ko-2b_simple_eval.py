from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
import torch
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd


def parse_evaluation(xml_string):
    root = ET.fromstring(xml_string)
    result_dict = {child.tag: child.text for child in root}
    return result_dict

def format_example(example):
    prompt = f"""[Self-Introduction]
{example['selfintro']}

[Evaluation]
"""
    # example_evaluation = parse_evaluation(example['evaluation'])
    # full_text = prompt + example_evaluation['eval_selfintro']
    return {"text": prompt}

def get_selfintro(example):
    example_evaluation = parse_evaluation(example['evaluation'])
    return example_evaluation['eval_selfintro']

dataset = load_from_disk("small_resume_dataset")
# dataset["test"] = dataset["test"].shuffle(seed=42).select(range(10))  # 테스트 데이터 500개로 제한

tokenizer = AutoTokenizer.from_pretrained("beomi/gemma-ko-2b", use_fast=True)

def tokenize(example):
    return tokenizer(example["text"], truncation=True)

selfintro_bases = [get_selfintro(selfintro) for selfintro in dataset["test"]]
dataset["test"] = dataset["test"].map(format_example)
dataset["test"] = dataset["test"].map(tokenize, remove_columns=dataset["test"].column_names)
dataset["test"].set_format(type="torch", columns=["input_ids", "attention_mask"])

# 원본 모델 불러오기 (4bit quantization 없이 평가 시)
model = AutoModelForCausalLM.from_pretrained(
    "beomi/gemma-ko-2b",
    device_map="auto",
    torch_dtype=torch.float16,
)

# LoRA 가중치 로드 (QLoRA로 fine-tuned 된 모델 경로)
model = PeftModel.from_pretrained(model, "./qlora-final-model")

model.eval()

results = []

for i, example in tqdm(enumerate(dataset["test"]), desc="Generating"):
    selfintro = selfintro_bases[i]
    
    input_ids = example["input_ids"].unsqueeze(0).to(model.device)
    attention_mask = example["attention_mask"].unsqueeze(0).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,  # greedy decoding
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
        )
    
    prompt_len = input_ids.size(1)
    generated_text = tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)
    generated_text = generated_text.strip().split("\n\n")[0]
    results.append({"selfintro": selfintro, "generated_text": generated_text})


# results 리스트를 DataFrame으로 변환
df = pd.DataFrame(results)

# CSV 파일로 저장 (utf-8 인코딩, 한글 깨짐 방지)
df.to_csv("gemma-ko-2b_simple_test.csv", index=False, encoding="utf-8-sig")