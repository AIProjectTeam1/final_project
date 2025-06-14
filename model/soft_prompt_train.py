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

# 커스텀 Trainer 클래스 정의
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs 딕셔너리에서 'num_items_in_batch' 인자를 제거 (혹시 있다면)
        # 이 인자는 model.forward()에서 사용되지 않음
        # 이전에 발생한 TypeError 해결을 위함
        if "num_items_in_batch" in inputs:
            del inputs["num_items_in_batch"]

        # 원래의 Trainer.compute_loss 로직을 따름
        # 모델의 forward 메서드를 호출하기 전에 필요한 처리
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs) # 여기서 문제가 발생했었음. 이제 inputs에 num_items_in_batch 없음.

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For `Trainer`, either the model should return a loss, "
                    "or provide its `labels` input parameter and override `compute_loss`."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if self.label_smoother is not None:
                loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                loss = None

        return (loss, outputs) if return_outputs else loss


def main(args):
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # 1. Load dataset
    dataset = load_from_disk(args.dataset_path)

    # 2. Load tokenizer & LoRA fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # 토크나이저에 패딩 토큰이 없는 경우 추가
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # 모델 리사이즈 (임베딩 레이어의 크기 조정)
        # 이 부분은 base_model 로드 후에 해야 함
        # model.resize_token_embeddings(len(tokenizer)) # 이 코드는 get_peft_model 적용 전에 필요할 수 있음

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )

    # 토크나이저에 패딩 토큰을 추가했다면, 모델 임베딩 레이어 크기 조정
    if tokenizer.pad_token is not None and base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))


    model = PeftModel.from_pretrained(
        base_model,
        args.base_model_path,
        is_trainable=False  # LoRA 파라미터는 고정
    )

    # 3. Prompt Tuning 설정 추가 (LoRA freeze)
    prompt_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=args.num_virtual_tokens,
        # tokenizer_name_or_path=args.model_name, # 이 인자는 제거해도 됨
    )

    model = get_peft_model(model, prompt_config)
    model.print_trainable_parameters()
    print("Trainable parameters after PEFT model creation:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}, shape: {param.shape}")
    print("-" * 30)


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
        # data_collator에서 패딩을 처리하므로, 여기서는 패딩하지 않아도 됨
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

    # 11. Trainer 구성 (커스텀 Trainer 사용)
    trainer = CustomTrainer( # <-- 여기를 CustomTrainer로 변경
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

    parser.add_argument("--train_batch_size", type=int, default=1, help="Train batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=0.0004869, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")

    parser.add_argument("--eval_steps", type=int, default=100, help="Eval steps")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to save")

    args = parser.parse_args()
    main(args)