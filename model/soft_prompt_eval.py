import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PromptTuningConfig
from peft.tuners.prompt_tuning.model import PromptEmbedding
from datasets import load_from_disk
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from prompts import build_prompt_messages

# 1. 설정
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
peft_model_path = "./model-output/qlora-output_v2_explicit_guideline"
prompt_encoder_path = "./prompt-tuned/prompt_encoder.bin"
dataset_path = "small_resume_dataset_final"
prompt_version = "prompt_for_soft_prompt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_samples = 3

# 2. 모델 및 프롬프트 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, peft_model_path)

prompt_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=2,
    tokenizer_name_or_path=model_name,
    num_transformer_submodules=1,
    token_dim=base_model.config.hidden_size
)
prompt_encoder = PromptEmbedding(prompt_config, model.get_input_embeddings())
prompt_encoder.load_state_dict(torch.load(prompt_encoder_path, map_location="cpu"))
prompt_encoder = prompt_encoder.to(model.device).to(torch.bfloat16)

class ModelWithPrompt(torch.nn.Module):
    def __init__(self, model, prompt_encoder, prompt_config):
        super().__init__()
        self.model = model
        self.prompt_encoder = prompt_encoder
        self.prompt_config = prompt_config

    def generate(self, input_ids, attention_mask, **kwargs):
        batch_size = input_ids.size(0)
        prompt_embeds = self.prompt_encoder(torch.arange(self.prompt_config.num_virtual_tokens, device=input_ids.device))
        prompt_embeds = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        token_embeds = self.model.get_input_embeddings()(input_ids)
        token_embeds = token_embeds.to(dtype=prompt_embeds.dtype)
        inputs_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)
        prompt_attn = torch.ones((batch_size, self.prompt_config.num_virtual_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
        extended_attention_mask = torch.cat([prompt_attn, attention_mask], dim=1)
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            eos_token_id=kwargs.get("eos_token_id", tokenizer.eos_token_id),
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            do_sample=False
        )

wrapped_model = ModelWithPrompt(model, prompt_encoder, prompt_config)
wrapped_model.eval()

# 3. 데이터 로드 및 전처리
def parse_evaluation(xml_string):
    root = ET.fromstring(xml_string)
    return {child.tag: child.text for child in root}

def get_ground_truth(example):
    parsed = parse_evaluation(example["evaluation"])
    return {
        "eval_resume": parsed.get("eval_resume", ""),
        "eval_selfintro": parsed.get("eval_selfintro", "")
    }

def remove_asterisks(text):
    return text.replace("**", "")

def parse_key_value(text):
    result = {}
    current_key = None
    for line in text.strip().split('\n'):
        if ':' in line and not line.startswith(' '):
            key, value = line.split(':', 1)
            current_key = key.strip()
            result[current_key] = value.strip()
        elif current_key:
            result[current_key] += ' ' + line.strip()
    return result

def convert_markdown_to_json(text):
    return parse_key_value(remove_asterisks(text))

# 4. 데이터셋 준비 및 추론
raw_dataset = load_from_disk(dataset_path)["test"].shuffle(seed=42).select(range(n_samples))
messages_list = [build_prompt_messages(ex, prompt_version, None) for ex in raw_dataset]
ground_truthes = [get_ground_truth(ex) for ex in raw_dataset]
resumes = [ex["resume"] for ex in raw_dataset]
selfintros = [ex.get("selfintro", "") for ex in raw_dataset]

resume_results, selfintro_results = [], []

for messages, gt, resume, selfintro in tqdm(zip(messages_list, ground_truthes, resumes, selfintros), total=len(messages_list)):
    input_tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    input_ids = input_tokenized
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        generated_ids = wrapped_model.generate(input_ids=input_ids, attention_mask=attention_mask)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        print(f"Generated text: {generated_text}")

    try:
        json_output = convert_markdown_to_json(generated_text)
    except:
        json_output = {}

    resume_results.append({
        "eval_resume": gt["eval_resume"],
        "generated_text": json_output.get("eval_resume", generated_text),
        "resume": resume
    })
    selfintro_results.append({
        "eval_resume": gt["eval_selfintro"],
        "generated_text": json_output.get("eval_selfintro", generated_text),
        "selfintro": selfintro
    })

# 5. 저장
pd.DataFrame(resume_results).to_csv("inference_results_resume.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(selfintro_results).to_csv("inference_results_selfintro.csv", index=False, encoding="utf-8-sig")

print("Saved to inference_results_*.csv")

