import torch
import xml.etree.ElementTree as ET
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import wandb
from prompts import prompt_templates

from transformers import TrainerCallback, TrainerState, TrainerControl

class WandbEvalLossLogger(TrainerCallback):
    def __init__(self, stage=0, prev_steps=0):
        self.stage = stage
        self.prev_steps = prev_steps

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            wandb.log({
                "eval_loss": metrics["eval_loss"],
                "stage": self.stage
            }, step=self.prev_steps + state.global_step)  # 누적 step으로 로그 찍기



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

def get_curriculum_splits(dataset, num_stages=2):
    import numpy as np
    scores = np.array([item['selfintro_score'] for item in dataset])
    high_threshold = np.percentile(scores, 75)
    low_threshold = np.percentile(scores, 25)
    clear_indices = np.where((scores >= high_threshold) | (scores <= low_threshold))[0]
    ambiguous_indices = np.where((scores < high_threshold) & (scores > low_threshold))[0]
    return [clear_indices, ambiguous_indices]

def train(config):
    # Load dataset
    dataset = load_from_disk("small_resume_dataset_final")
    # dataset["train"] = dataset["train"].select(range(1000))

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
        r=config.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)

    prompt_fn = prompt_templates[config.prompt_version]

    def example_format_fn(example):
        return format_example(example, tokenizer, prompt_fn)

    def example_tokenize_fn(example):
        return tokenize(example, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=64)

    training_args = TrainingArguments(
        output_dir=config.output_dir if hasattr(config, 'output_dir') else "./qlora-output",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps if hasattr(config, 'gradient_accumulation_steps') else 8,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb" if (hasattr(config, 'use_wandb') and config.use_wandb) else "none",
        save_total_limit=2,
        gradient_checkpointing=True,
    )

    # Curriculum Learning 여부에 따른 분기
    if hasattr(config, "use_curriculum") and config.use_curriculum:
        print("\n=== Using Curriculum Learning ===")
        stages = get_curriculum_splits(dataset["train"])

        prev_steps = 0  # 누적 step 저장

        for stage_idx, indices in enumerate(stages):
            print(f"\n--- Training Stage {stage_idx + 1} ---")
            stage_dataset = dataset["train"].select(indices)
            stage_dataset = stage_dataset.map(example_format_fn)
            stage_dataset = stage_dataset.map(example_tokenize_fn, remove_columns=stage_dataset.column_names)

            val_dataset = dataset["validation"].map(example_format_fn)
            val_dataset = val_dataset.map(example_tokenize_fn, remove_columns=val_dataset.column_names)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=stage_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[WandbEvalLossLogger(stage=stage_idx + 1, prev_steps=prev_steps)],
            )

            trainer.train()
            trainer.save_model(f"{training_args.output_dir}/stage_{stage_idx + 1}")

            # 현재까지의 학습 step을 누적
            prev_steps += trainer.state.global_step
    else:
        print("\n=== Training Without Curriculum ===")
        dataset["train"] = dataset["train"].map(example_format_fn)
        dataset["train"] = dataset["train"].map(example_tokenize_fn, remove_columns=dataset["train"].column_names)
        dataset["validation"] = dataset["validation"].map(example_format_fn)
        dataset["validation"] = dataset["validation"].map(example_tokenize_fn, remove_columns=dataset["validation"].column_names)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[WandbEvalLossLogger()],
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
    parser.add_argument("--use_curriculum", action='store_true')

    args = parser.parse_args()

    

    if args.use_wandb:
        from datetime import datetime

        now = datetime.now().strftime("%Y%m%d-%H%M%S")

        name = f"train_{args.prompt_version}" + \
            ("_curriculum" if args.use_curriculum else "_nocurriculum") + \
            f"_{now}"
        
        config_dict = vars(args)
        wandb.init(config=config_dict, project="resume_eval", name=name)
        config = wandb.config
    else:
        config = args

    train(config)

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
