import torch
import xml.etree.ElementTree as ET
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import wandb
from prompts import prompt_templates

def parse_evaluation(xml_string):
    root = ET.fromstring(xml_string)
    return {child.tag: child.text for child in root}

def format_example(example, tokenizer, prompt_fn):
    eval_dict = parse_evaluation(example['evaluation'])
    messages = prompt_fn(example, eval_dict)
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": formatted_text}

def tokenize(example, tokenizer):
    return tokenizer(example["text"], truncation=True)

def train(config):
    # wandb.init() 호출 (이미 init 되어있으면 안 함)
    if not wandb.run:
        # config가 dict나 Namespace 형태가 아니라면 vars() 호출 방어
        config_for_wandb = config
        if hasattr(config, "__dict__"):
            config_for_wandb = vars(config)
        elif not isinstance(config, dict):
            config_for_wandb = dict(config)
        wandb.init(project="resume_eval", config=config_for_wandb)

    # 이제 wandb.config로 접근 가능
    cfg = wandb.config

    # Load dataset
    dataset = load_from_disk("small_resume_dataset_final")
    dataset["train"] = dataset["train"].select(range(1000))

    # Load model & tokenizer
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

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)

    prompt_fn = prompt_templates[cfg.prompt_version]

    def example_format_fn(example):
        return format_example(example, tokenizer, prompt_fn)

    def example_tokenize_fn(example):
        return tokenize(example, tokenizer)

    for split in ["train", "validation"]:
        dataset[split] = dataset[split].map(example_format_fn)
        dataset[split] = dataset[split].map(example_tokenize_fn, remove_columns=dataset[split].column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=64)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir if hasattr(cfg, 'output_dir') else "./qlora-output",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps if hasattr(cfg, 'gradient_accumulation_steps') else 8,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        fp16=True,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb" if (hasattr(cfg, 'use_wandb') and cfg.use_wandb) else "none",
        save_total_limit=2,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_version", type=str, default="prompt_v2_all_eval")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--output_dir", type=str, default="./qlora-output")
    args = parser.parse_args()

    if args.use_wandb:
        # wandb.init()는 train에서 처리하므로 여기서는 config만 넘김
        config = args
    else:
        config = args

    train(config)

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
