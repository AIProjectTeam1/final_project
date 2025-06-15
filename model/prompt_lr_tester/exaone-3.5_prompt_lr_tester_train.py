from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk
from transformers import BitsAndBytesConfig
from peft import PeftModel, PromptTuningConfig
from peft.tuners.prompt_tuning.model import PromptEmbedding
import torch
import xml.etree.ElementTree as ET
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 데이터셋 로드
full_dataset = load_from_disk("small_resume_dataset_final")
dataset = {
    "train": full_dataset["train"],
    "validation": full_dataset["validation"]
}

# 2. 모델 및 토크나이저
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
base_model_path = "./qlora-output_v2_explicit_guideline"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

if tokenizer.pad_token is not None and base_model.config.vocab_size < len(tokenizer):
    base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(
    base_model,
    base_model_path,
    is_trainable=False
)

# 3. 프롬프트 임베딩 설정
prompt_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=8,
    tokenizer_name_or_path=model_name,
    num_transformer_submodules=1,
    token_dim=base_model.config.hidden_size
)
prompt_encoder = PromptEmbedding(prompt_config, model.get_input_embeddings()).to(device)

# 4. 전처리 함수
def parse_evaluation(xml_string):
    root = ET.fromstring(xml_string)
    return {child.tag: child.text for child in root}

def format_dict_to_string(data):
    if isinstance(data, dict):
        return "\n".join(f"- {k}: {v}" for k, v in data.items())
    return str(data)

def format_and_tokenize(example):
    eval_dict = parse_evaluation(example['evaluation'])

    job_text = format_dict_to_string(example['job_analysis'])
    resume_text = format_dict_to_string(example['resume_analysis'])
    keywords_text = format_dict_to_string(example['keywords_analysis'])

    messages = [
        {"role": "system", "content": "당신은 숙련된 인사(HR) 어시스턴트입니다. 지원자의 이력서와 자기소개서를 직무 요구사항에 맞추어 꼼꼼히 평가해 주세요."},
        {"role": "user", "content": f"""아래 문서를 바탕으로 평가 항목별로 구체적인 피드백을 작성해 주세요.\n
        [직무 공고]\n{job_text}\n
        [이력서]\n{resume_text}\n
        [키워드]\n{keywords_text}\n
        [자기소개서]\n{example['selfintro']}\n
        **평가 지침**
        - eval_resume: 이력서에서 언급된 기술 및 경험이 직무 요건과 얼마나 부합하는지 평가해 주세요. 기술별 언급이 포함되면 좋습니다. 구체적 근거와 함께 3~5문장으로 작성해 주세요.
        - eval_selfintro: 자기소개서의 구성, 표현력, 지원 동기, 입사 후 포부, 어투 등을 평가해 주세요. 장점과 개선점을 함께 언급하며 3~5문장으로 작성해 주세요.
        - summary: 이력서와 자기소개서를 종합하여 전체적인 인상 및 채용 적합성에 대한 의견을 2~4문장으로 요약해 주세요.

        아래 형식으로 응답해주세요.
        eval_resume: 이력서 평가 내용
        eval_selfintro: 자기소개서 평가 내용
        summary: 전체적인 인상 및 채용 적합성 요약
        """ },
        {"role": "assistant", "content": json.dumps({
            "eval_resume": eval_dict.get("eval_resume", ""),
            "eval_selfintro": eval_dict.get("eval_selfintro", ""),
            "summary": eval_dict.get("summary", "")
        }, ensure_ascii=False)}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer(text, truncation=True, max_length=2000)

for split in ["train", "validation"]:
    dataset[split] = dataset[split].map(
        format_and_tokenize,
        remove_columns=[col for col in dataset[split].column_names if col not in ['input_ids', 'attention_mask']]
    )

# 5. 커스텀 collator
class PromptCollator:
    def __init__(self, tokenizer, prompt_config, pad_to_multiple_of=64):
        self.tokenizer = tokenizer
        self.prompt_config = prompt_config
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        input_ids = [ex["input_ids"] for ex in batch]
        attention_mask = [ex["attention_mask"] for ex in batch]

        batch_input = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        batch_input = {k: v.to(device) for k, v in batch_input.items()}

        prompt_mask = torch.full(
            (batch_input["input_ids"].size(0), prompt_config.num_virtual_tokens),
            -100, dtype=batch_input["input_ids"].dtype, device=device
        )
        labels = torch.cat([prompt_mask, batch_input["input_ids"]], dim=1)

        prompt_attn = torch.ones_like(prompt_mask, dtype=batch_input["attention_mask"].dtype)
        attention_mask = torch.cat([prompt_attn, batch_input["attention_mask"]], dim=1)

        return {
            "input_ids": batch_input["input_ids"],
            "attention_mask": attention_mask,
            "labels": labels
        }

data_collator = PromptCollator(tokenizer, prompt_config)

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="./prompt-tuned_final",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-4,
    num_train_epochs=1,
    fp16=True,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    save_total_limit=2,
    save_safetensors=False,
    remove_unused_columns=False,
    dataloader_pin_memory=False
)

# 7. 모델 래퍼
class ModelWithPrompt(torch.nn.Module):
    def __init__(self, model, prompt_encoder, prompt_config):
        super().__init__()
        self.model = model
        self.prompt_encoder = prompt_encoder
        self.prompt_config = prompt_config

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        prompt_embeds = self.prompt_encoder(torch.arange(self.prompt_config.num_virtual_tokens, device=device)).to(dtype=self.model.dtype)
        prompt_embeds = prompt_embeds.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        token_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

wrapped_model = ModelWithPrompt(model, prompt_encoder, prompt_config)

# 8. Trainer 설정
trainer = Trainer(
    model=wrapped_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 9. 학습 수행
print("\nStarting training...")
trainer.train()
print("\nTraining finished.")

# 10. 모델 및 프롬프트 저장
output_dir = training_args.output_dir
trainer.save_model(output_dir)
torch.save(prompt_encoder.state_dict(), f"{output_dir}/prompt_encoder.bin")
print(f"\nModel and prompt encoder saved to {output_dir}")
