from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig
import torch
import xml.etree.ElementTree as ET
import numpy as np

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
         위 Job-Post, Resume, Keywords를 바탕으로 Self-Introduction를 평가해주세요. 피드백은 반드시 한 문단으로 간결하게 작성해 주세요. 항목을 나누지 말고, 문장 수는 3~4문장 이내로 제한해 주세요."""},
        {"role": "assistant", "content": eval_dict.get('eval_selfintro', '')}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": formatted_text}

# 6. Apply formatting and tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True)

# 7. Curriculum Learning을 위한 데이터셋 분할
def get_curriculum_splits(dataset, num_stages=3):
    """
    selfintro_score를 기준으로 데이터셋을 단계별로 분할
    Stage 1: 높은 점수와 낮은 점수 (명확한 피드백)
    Stage 2: 중간 점수 (애매한 피드백)
    """
    scores = np.array([item['selfintro_score'] for item in dataset])
    
    # 점수 기준으로 데이터 분할
    high_threshold = np.percentile(scores, 75)  # 상위 25%
    low_threshold = np.percentile(scores, 25)   # 하위 25%
    
    # Stage 1: 명확한 피드백 (높은 점수 + 낮은 점수)
    clear_indices = np.where((scores >= high_threshold) | (scores <= low_threshold))[0]
    
    # Stage 2: 애매한 피드백 (중간 점수)
    ambiguous_indices = np.where((scores < high_threshold) & (scores > low_threshold))[0]
    
    return [clear_indices, ambiguous_indices]

# 8. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=64)

# 9. Training arguments
training_args = TrainingArguments(
    output_dir="./qlora-curriculum-output",
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

# 10. Curriculum Learning 단계별 학습
stages = get_curriculum_splits(dataset["train"])

# Stage 1: 명확한 피드백으로 학습
print("\n=== Starting Stage 1: Clear Feedback (High and Low Scores) ===")
stage1_dataset = dataset["train"].select(stages[0])
stage1_dataset = stage1_dataset.map(format_example)
stage1_dataset = stage1_dataset.map(tokenize, remove_columns=stage1_dataset.column_names)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=stage1_dataset,
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("./qlora-curriculum-output/stage_1")

# Stage 2: 애매한 피드백으로 학습
print("\n=== Starting Stage 2: Ambiguous Feedback (Middle Scores) ===")
stage2_dataset = dataset["train"].select(stages[1])
stage2_dataset = stage2_dataset.map(format_example)
stage2_dataset = stage2_dataset.map(tokenize, remove_columns=stage2_dataset.column_names)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=stage2_dataset,
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("./qlora-curriculum-final-model") 