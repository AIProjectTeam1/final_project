from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_from_disk, load_dataset
from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig
import torch
from transformers import BitsAndBytesConfig

# 1. Load dataset
# dataset = load_dataset("Youseff1987/resume-matching-dataset-v2")
dataset = load_from_disk("small_resume_dataset")

# 2. Load model & tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # 보통 nf4 또는 fp4 중 선택
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "beomi/gemma-ko-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
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

# 4. Prompt formatting function
def format_example(example):
    prompt = f"""<s>[Job Post]
{example['jobpost']}

[Resume]
{example['resume']}

[Self-Introduction]
{example['selfintro']}

[Evaluation]
"""
    full_text = prompt + example['evaluation'] + "</s>"
    return {"text": full_text}

# 5. Apply formatting and tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=1024)

for split in ["train", "validation", "test"]:
    dataset[split] = dataset[split].map(format_example)
    dataset[split] = dataset[split].map(tokenize, remove_columns=dataset[split].column_names)

# 6. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./qlora-output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    save_total_limit=2
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 9. Start training
trainer.train()

# 10. Save model
trainer.save_model("./qlora-final-model")
