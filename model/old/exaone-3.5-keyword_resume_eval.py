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
model = PeftModel.from_pretrained(base_model, "./qlora_models_resume_model/qlora-exaone-3.5-model")
model.eval()

# Dataset 불러오기
dataset = load_from_disk("small_resume_dataset_final")
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(500))

# 평가 결과 파싱 함수
def parse_evaluation(xml_string):
    root = ET.fromstring(xml_string)
    return {child.tag: child.text for child in root}

# selfintro 원본 보관용
def get_eval_resume(example):
    return parse_evaluation(example["evaluation"])["eval_resume"]

# Chat template message 구성 함수
def build_messages(example):
    return [
        {"role": "system", "content": "You are a helpful assistant for resume evaluation."},
        {"role": "user", "content": f"""[Job-Post]\n{example['job_analysis']}\n
         [Resume]\n{example['resume_analysis']}\n
         위 Job-Post를 바탕으로 Resume를 평가해주세요. 피드백은 반드시 아래 예시처럼 한 문단으로 간결하게 작성해 주세요. 항목을 나누지 말고, 문장 수는 3~4문장 이내로 제한해 주세요.
         
         Example:
         - 지원자는 HTML, CSS, JavaScript, React.js 및 Next.js에 대한 확실한 이해를 보유하고 있으며, 졸업 프로젝트와 인턴 경험을 통해 실무 능력을 쌓아왔습니다. 이력서에서 요구되는 자격 요건을 완벽히 충족하고, 우대 사항 중 Tailwind CSS 경험도 보유하고 있어 큰 장점을 가지고 있습니다. 공백 기간 없이 관련 경험이 모두 연속적으로 나열되어 있으며, 기술과 경력이 잘 결합되어 있습니다. 다만, Zustand의 경험이 학교 프로젝트 상황에서 한정되는 부분에서 약간의 감점이 있었습니다.         
         
         이 예시의 길이와 문장 구조를 그대로 따라 주세요.         
         """},
    ]

# Prompt 생성 및 Tokenization
messages_list = [build_messages(example) for example in dataset["test"]]
eval_resumes_bases = [get_eval_resume(example) for example in dataset["test"]]
resumes = [example["resume"] for example in dataset["test"]]

# 결과 저장용
results = []

for messages, gt, resume in tqdm(zip(messages_list, eval_resumes_bases, resumes), total=len(messages_list), desc="Generating"):
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
    results.append({"eval_resume": gt, "generated_text": generated_text, "resume": resume})

# CSV 저장
pd.DataFrame(results).to_csv("exaone-3.5_keyword_eval_resume_250531.csv", index=False, encoding="utf-8-sig")
