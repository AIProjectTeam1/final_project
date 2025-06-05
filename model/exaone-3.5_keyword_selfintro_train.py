from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig
import torch
import xml.etree.ElementTree as ET

# 1. Load dataset
dataset = load_from_disk("small_resume_dataset_final")

# 2. Load model & tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 3. Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# 4. Parse XML evaluation
def parse_evaluation(xml_string):
    root = ET.fromstring(xml_string)
    return {child.tag: child.text for child in root}

# 5. Format example as chat messages and apply template
def format_example(example):
    eval_dict = parse_evaluation(example['evaluation'])
    messages = [
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
        {"role": "assistant", "content": eval_dict.get('eval_selfintro', '')}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": formatted_text}


# 6. Apply formatting and tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True)

for split in ["train", "validation"]:
    dataset[split] = dataset[split].map(format_example)
    dataset[split] = dataset[split].map(tokenize, remove_columns=dataset[split].column_names)

# 7. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=64)

# 8. Training arguments
training_args = TrainingArguments(
    output_dir="./qlora-output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    save_total_limit=2,
    gradient_checkpointing=True,
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 10. Train
trainer.train()

# 11. Save model
trainer.save_model("./qlora-exaone-3.5-model")
