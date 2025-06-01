from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_from_disk
import torch
import xml.etree.ElementTree as ET

# 1. Load dataset from disk (Hugging Face .arrow format expected)
dataset = load_from_disk("small_resume_dataset_final")
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(6500))



# 2. Load model & tokenizer with 4-bit quantization configuration
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


# 3. Prepare model for QLoRA (k-bit training) and apply LoRA
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


# 4. Parse XML evaluation string into a dict
def parse_evaluation(xml_string: str) -> dict:
    """
    Input: XML string containing <eval_resume>...</eval_resume> (and possibly other tags)
    Output: A dictionary where each child tag maps to its text content.
    """
    root = ET.fromstring(xml_string)
    return {child.tag: child.text for child in root}


# 5. Format each example: combine Job Post + Resume → Eval_Resume
def format_example(example: dict) -> dict:
    """
    - example['jobpost']   : string containing the job posting
    - example['resume']    : string containing the candidate's resume
    - example['evaluation']: string containing XML with <eval_resume>…</eval_resume>
    
    After parsing XML, we extract <eval_resume> as eval_resume_text.
    We then construct a chat-style prompt where the user message is:
      [Job Post]
      ...
      [Resume]
      ...
    and the assistant message is the eval_resume_text. We pass messages into
    tokenizer.apply_chat_template(...) to get a single formatted string.
    """
    eval_dict = parse_evaluation(example["evaluation"])
    eval_resume_text = eval_dict.get("eval_resume", "").strip()

    prompt = f"""[Job Post]
{example["jobpost"].strip()}

[Resume]
{example["resume"].strip()}"""

    messages = [
        {"role": "system",    "content": "You are a helpful assistant for resume evaluation."},
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": eval_resume_text}
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": formatted_text}


# 6. Tokenize the "text" field to produce input_ids & attention_mask
def tokenize(example: dict) -> dict:
    return tokenizer(
        example["text"],
        truncation=True
    )


# 7. Apply formatting and tokenization to train/validation splits
for split in ["train", "validation"]:
    # 7-1) Map format_example: create "text" column, remove all original columns
    dataset[split] = dataset[split].map(
        format_example,
        remove_columns=dataset[split].column_names
    )
    # 7-2) Map tokenize: tokenize "text" → (input_ids, attention_mask), then remove "text"
    dataset[split] = dataset[split].map(
        tokenize,
        remove_columns=["text"]
    )


# 8. Data collator for causal language modeling (no MLM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=64
)


# 9. Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./qlora-output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
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


# 10. Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)


# 11. Train
trainer.train()


# 12. Save the fine-tuned model
trainer.save_model("./qlora-exaone-3.5-resume")
