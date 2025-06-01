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

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# # QLoRA 가중치 적용 (optional)
# model = PeftModel.from_pretrained(base_model, "./qlora-exaone-3.5-model")
# model.eval()

# Dataset 불러오기
dataset = load_from_disk("small_resume_dataset_final")
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(500))

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
        {"role": "user", "content": f"""[Job-Post]\n{example['job_analysis']}\n
         [Resume]\n{example['resume_analysis']}\n
         [keywords]\n{example['keywords_analysis']}\n
         [Self-Introduction]\n{example['selfintro']}
         위 Job-Post, Resume, Keywords를 바탕으로 Self-Introduction를 평가해주세요. 피드백은 반드시 아래 예시처럼 한 문단으로 간결하게 작성해 주세요. 항목을 나누지 말고, 문장 수는 3~4문장 이내로 제한해 주세요.
         
         Example:
         - 자기소개서는 전반적으로 논리적인 구조를 갖추고 있으나 표현력이 부족하고 다소 평범한 느낌을 주는 문장이 있었습니다. 각 경험에 대한 설명이 잘 이루어져 있지만, 구어체 표현이 있어 다소 격식이 떨어지는 인상을 주며 정확한 기술적 용어의 사용이 부족했기 때문에 감점이 있었습니다. 또한, 지원 동기와 입사 후 포부에서는 팀워크과 협업 강조가 나타나지만, 구체적인 계획이나 목표가 부족하여 심도 있는 인상을 주지 못했습니다. 그럼에도 불구하고 기본적으로 성실한 태도를 갖추고 있습니다.
         
         이 예시의 길이와 문장 구조를 그대로 따라 주세요.         
         """},
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
pd.DataFrame(results).to_csv("exaone-3.5_baseline_eval_250531.csv", index=False, encoding="utf-8-sig")
