## 🚀 How to Run

### 1. Run a Single Training Experiment

Use `multi_prompt_train.py` to run a single training job with your desired hyperparameters:

```bash
python model/multi_prompt_train.py \
  --prompt_version v2_explicit_guidelines_ko_improved \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lora_r 8
```

#### Available Arguments

| Argument                     | Description                                               | Example Values                        |
|-----------------------------|-----------------------------------------------------------|---------------------------------------|
| `--prompt_version`          | Prompt template version to use                            | `v2_minimal`, `v2_all_eval`, etc.     |
| `--learning_rate`           | Learning rate for training                                | `1e-5`, `5e-5`, `1e-4`, etc.          |
| `--num_train_epochs`        | Number of training epochs                                 | `3`, `5`, etc.                        |
| `--batch_size`              | Batch size per GPU                                        | `2`                                   |
| `--gradient_accumulation_steps` | Number of steps for gradient accumulation             | `4`, `8`, `16`                        |
| `--lora_r`                  | Rank parameter for LoRA fine-tuning                       | `4`, `8`, `16`                        |

### 2. Run a Sweep with W&B

#### Setup

- Make sure you have a [Weights & Biases](https://wandb.ai/) account.
- Login via CLI:

```bash
wandb login
```

#### Sweep Configuration (`model/sweep.yaml`)

Example using Bayesian optimization:

```yaml
method: bayes
metric:
  name: eval_loss
  goal: minimize

parameters:
  prompt_version:
    values: ["v2_all_eval_formatted", "v2_minimal", "v2_explicit_guidelines_ko_improved"]
  learning_rate:
    min: 0.00001
    max: 0.0005
  num_train_epochs:
    values: [3, 5]
  batch_size:
    value: 2
  gradient_accumulation_steps:
    values: [4, 8, 16]
  lora_r:
    values: [4, 8, 16]
```

#### Launch the Sweep

Instead of using the default `wandb agent`, launch the sweep using your own script:

```bash
python model/sweep_runner.py
```

The `sweep_runner.py` script internally registers the sweep and uses `wandb.agent` to execute training using configurations defined in `sweep.yaml`.

## Sweep Results

[https://wandb.ai/codream00-sungkyunkwan-university/resume_eval_ko_sweep/sweeps/3yle2pch?nw=nwusercodream00](https://wandb.ai/codream00-sungkyunkwan-university/resume_eval_ko_sweep/sweeps/3yle2pch?nw=nwusercodream00)
