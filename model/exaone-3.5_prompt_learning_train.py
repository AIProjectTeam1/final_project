from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import PeftModel, PromptTuningConfig, get_peft_model
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig
import torch
import xml.etree.ElementTree as ET

# 1. Load dataset
dataset = load_from_disk("small_resume_dataset_final")

# 2. Load tokenizer & LoRA fine-tuned model
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# QLoRA로 학습된 모델을 불러옴
base_model_path = "./qlora-exaone-3.5-model"
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

# 3. Prompt Tuning만 추가 적용
prompt_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=8,
    tokenizer_name_or_path=model_name,
)

model = get_peft_model(model, prompt_config)

# 4. XML 파싱
def parse_evaluation(xml_string):
    root = ET.fromstring(xml_string)
    return {child.tag: child.text for child in root}

# 5. Format example
def format_example(example):
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
    return {"text": formatted_text}

# 6. Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True)

for split in ["train", "validation"]:
    dataset[split] = dataset[split].map(format_example)
    dataset[split] = dataset[split].map(tokenize, remove_columns=dataset[split].column_names)

# 7. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=64)

# 8. Training arguments
training_args = TrainingArguments(
    output_dir="./prompt-tuned-from-qlora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
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
trainer.save_model("./prompt-tuned-from-qlora")
