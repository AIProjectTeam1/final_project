import argparse
import torch
import xml.etree.ElementTree as ET
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import PeftModel, PromptTuningConfig, get_peft_model
from prompts import prompt_templates
import wandb

def main(args):
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # 1. Load dataset
    dataset = load_from_disk(args.dataset_path)

    # 2. Load tokenizer & LoRA fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )

    # 3. Prompt Tuning 설정 추가 (LoRA freeze)
    prompt_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=args.num_virtual_tokens,
        tokenizer_name_or_path=args.model_name,
    )

    model = get_peft_model(model, prompt_config)

    # 4. LoRA 파라미터 freeze
    for name, param in model.named_parameters():
        param.requires_grad = "prompt_embeddings" in name

    # 5. XML 파싱 함수
    def parse_evaluation(xml_string):
        root = ET.fromstring(xml_string)
        return {child.tag: child.text for child in root}

    # 6. 예시 포맷 구성
    def format_example(example, prompt_version="prompt_for_soft_prompt"):
        eval_dict = parse_evaluation(example['evaluation'])
        prompt_fn = prompt_templates[prompt_version]
        messages = prompt_fn(example, eval_dict)
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": formatted_text}

    # 7. Tokenize
    def tokenize(example):
        return tokenizer(example["text"], truncation=True)

    # 8. Dataset 전처리
    for split in ["train", "validation"]:
        dataset[split] = dataset[split].map(format_example)
        dataset[split] = dataset[split].map(tokenize, remove_columns=dataset[split].column_names)

    # 9. Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=64)

    # 10. TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="epoch",
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        report_to="wandb",
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
    )

    # 11. Trainer 구성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 12. Train!
    trainer.train()

    # 13. Save model + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt tuning training script with LoRA frozen")

    parser.add_argument("--dataset_path", type=str, default="small_resume_dataset_final", help="Dataset path")
    parser.add_argument("--model_name", type=str, default="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", help="Base model name or path")
    parser.add_argument("--base_model_path", type=str, default="./model-output/qlora-output_v2_explicit_guideline", help="QLoRA fine-tuned model path")

    parser.add_argument("--output_dir", type=str, default="./prompt-tuned-from-qlora", help="Output directory")
    parser.add_argument("--wandb_project", type=str, default="resume_eval", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="prompt-tuning", help="Wandb run name")

    parser.add_argument("--num_virtual_tokens", type=int, default=8, help="Number of virtual tokens for prompt tuning")

    parser.add_argument("--train_batch_size", type=int, default=2, help="Train batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Eval batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=0.0004869, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")

    parser.add_argument("--eval_steps", type=int, default=100, help="Eval steps")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to save")

    args = parser.parse_args()
    main(args)
