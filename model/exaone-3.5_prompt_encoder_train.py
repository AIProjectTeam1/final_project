from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk
from transformers import BitsAndBytesConfig
from peft import PeftModel, PromptTuningConfig
from peft.tuners.prompt_tuning.model import PromptEmbedding
import torch
import xml.etree.ElementTree as ET

# 현재 장치 설정 (GPU 사용 가능 여부 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 데이터셋 로드
dataset = load_from_disk("small_resume_dataset_final")

# 2. 모델/토크나이저 로드
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
base_model_path = "./qlora-output_v2_explicit_guideline" # LoRA 모델 경로 확인

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# 토크나이저에 패딩 토큰이 없는 경우 추가
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("Added [PAD] token to tokenizer.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto", # 모델의 각 부분을 자동으로 GPU에 배치
    trust_remote_code=True,
)

# 토크나이저에 패딩 토큰을 추가했다면, 모델 임베딩 레이어 크기 조정
if tokenizer.pad_token is not None and base_model.config.vocab_size < len(tokenizer):
    base_model.resize_token_embeddings(len(tokenizer))
    print(f"Model embedding layer resized to {len(tokenizer)} for new pad token.")

# LoRA 모델 로드 (LoRA 가중치는 고정)
model = PeftModel.from_pretrained(
    base_model,
    base_model_path,
    is_trainable=False
)

# 모델의 파라미터가 실제로 GPU에 있는지 확인
# device_map="auto"를 사용하면 일반적으로 이 단계는 필요 없지만, 디버깅에 유용합니다.
print(f"PEFT model's first parameter device: {next(model.parameters()).device}")


# 3. Prompt Embedding 정의 (동적 삽입용)
prompt_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=2,
    tokenizer_name_or_path=model_name,
    num_transformer_submodules=1,
    token_dim=base_model.config.hidden_size
)
# PromptEmbedding 객체 생성 시, model.get_input_embeddings()를 사용하여 초기화
prompt_encoder = PromptEmbedding(prompt_config, model.get_input_embeddings())
# !!! 가장 중요한 변경: prompt_encoder를 명시적으로 GPU로 이동 !!!
prompt_encoder = prompt_encoder.to(device)
print(f"Prompt encoder device: {next(prompt_encoder.parameters()).device}")


# 4. 전처리 함수 (통합)
def parse_evaluation(xml_string):
    root = ET.fromstring(xml_string)
    return {child.tag: child.text for child in root}

def format_and_tokenize_example(example):
    eval_dict = parse_evaluation(example['evaluation'])
    messages = [
        {"role": "system", "content": "You are a helpful assistant for resume evaluation."},
        {"role": "user", "content": f"""[Job-Post]\n{example['job_analysis']}\n
          [Resume]\n{example['resume_analysis']}\n
          [keywords]\n{example['keywords_analysis']}\n
          [Self-Introduction]\n{example['selfintro']}
          위 Job-Post, Resume, Keywords를 바탕으로 Self-Introduction를 평가해주세요. 피드백은 반드시 한 문단으로 간결하게 작성해 주세요. 항목을 나누지 말고, 문장 수는 3~4문장 이내로 제한해 주세요."""},
        {"role": "assistant", "content": eval_dict.get('eval_selfintro', '')}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    tokenized_output = tokenizer(formatted_text, truncation=True, max_length=1016)
    return tokenized_output

print("Applying format_and_tokenize_example to dataset splits...")
for split in ["train", "validation"]:
    dataset[split] = dataset[split].map(
        format_and_tokenize_example,
        # input_ids와 attention_mask를 제외한 모든 원본 컬럼 제거
        remove_columns=[col for col in dataset[split].column_names if col not in ['input_ids', 'attention_mask']]
    )
print("Dataset preparation complete.")


# 5. Prompt 삽입용 커스텀 collator
# 이 Collator는 input_ids를 패딩하고, Prompt 영역을 -100으로 처리한 labels를 생성합니다.
# 실제 prompt_embeds와 token_embeds를 결합하는 로직은 ModelWithPrompt로 이동합니다.
class PromptCollator:
    def __init__(self, tokenizer, prompt_config, pad_to_multiple_of=64):
        self.tokenizer = tokenizer
        self.prompt_config = prompt_config
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        input_ids = [example["input_ids"] for example in batch]
        attention_mask = [example["attention_mask"] for example in batch]

        # Padding
        batch_input = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        # 패딩된 텐서들을 현재 장치(device)로 이동
        batch_input = {k: v.to(device) for k, v in batch_input.items()}

        # Labels 구성: Prompt 부분은 -100으로 설정하여 손실 계산에서 제외
        # prompt_mask는 labels와 동일한 모양으로 생성
        prompt_mask_for_labels = torch.full(
            (batch_input["input_ids"].size(0), self.prompt_config.num_virtual_tokens),
            -100,
            dtype=batch_input["input_ids"].dtype,
            device=device
        )
        labels = torch.cat([prompt_mask_for_labels, batch_input["input_ids"]], dim=1)

        # Attention mask도 확장 (prompt_mask는 1로 채움)
        prompt_attention_mask = torch.ones(
            batch_input["attention_mask"].size(0), self.prompt_config.num_virtual_tokens,
            dtype=batch_input["attention_mask"].dtype,
            device=device
        )
        attention_mask_expanded = torch.cat([prompt_attention_mask, batch_input["attention_mask"]], dim=1)

        return {
            "input_ids": batch_input["input_ids"], # 콜레이터는 input_ids만 반환
            "attention_mask": attention_mask_expanded, # 확장된 attention_mask 반환
            "labels": labels # 확장된 labels 반환
        }

# DataCollator 초기화
data_collator = PromptCollator(tokenizer, prompt_config)


# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="./prompt-tuned-from-lora-frozen",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=1,
    fp16=True,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    save_total_limit=2,
    remove_unused_columns=False,
    # --- HERE IS THE FIX ---
    dataloader_pin_memory=False, # <--- Add this line
    # -----------------------
)

# 7. 모델 래핑 (inputs_embeds 대응)
# ModelWithPrompt에서 inputs_embeds를 구성합니다.
class ModelWithPrompt(torch.nn.Module):
    def __init__(self, model, prompt_encoder, prompt_config):
        super().__init__()
        self.model = model # PeftModel
        self.prompt_encoder = prompt_encoder # 학습 가능한 prompt_encoder (GPU에 있음)
        self.prompt_config = prompt_config

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # input_ids는 DataCollator로부터 패딩된 원본 토큰 ID
        # attention_mask와 labels는 DataCollator에서 프롬프트만큼 확장된 상태로 넘어옴

        # Prompt embedding 생성 (GPU에 있는 prompt_encoder 사용)
        prompt_embeds = self.prompt_encoder(
            torch.arange(self.prompt_config.num_virtual_tokens, device=device)
        )
        prompt_embeds = prompt_embeds.unsqueeze(0).expand(input_ids.size(0), -1, -1)

        # 기존 토큰 임베딩
        # model.get_input_embeddings()는 PeftModel에서 기반 모델의 임베딩 레이어를 참조
        token_embeds = self.model.get_input_embeddings()(input_ids)

        # 입력 합치기: prompt_embeds + token_embeds
        inputs_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)

        # 모델에 inputs_embeds, attention_mask, labels 전달
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs

wrapped_model = ModelWithPrompt(model, prompt_encoder, prompt_config)

# 8. Trainer 정의
trainer = Trainer(
    model=wrapped_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer, # tokenizer는 여전히 필요 (내부 로깅 등)
    data_collator=data_collator
)

# 9. 학습
print("Starting training...")
trainer.train()
print("Training complete.")

# 10. 저장
trainer.save_model("./prompt-tuned-from-lora-frozen")
print("Model saved to ./prompt-tuned-from-lora-frozen")