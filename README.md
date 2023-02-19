# Fine-tuning (Large) Language Models

**Key Features 🔑**

1️⃣ Full use of computational resources (GPU Utilization)

- bf16, model parallelism, most optimal GPU utilization
- Support [Parameter Efficient Fine-tuning (PEFT)](https://github.com/huggingface/peft) with underlying PLM loaded in lower bits (int8())

2️⃣ Dynamically construct ANY kind of Training & Evaluation Dataset Combination

- Designate any # of instances for each training & validation dataset
- Allows instruction tuning & dynamic validation during training (beats T0 by +00%)
- Support all pre-training objective & fine-tuning objective (e.g. MLM, SSM, etc.)

3️⃣ Evaluate LLMs (even 175B LMs) on any kind of evaluation datasets (MMLU, BigBench, etc.) using any kind of verbalizers

- Supports multiple types of verbalizer techniques
- Supports all kinds of metrics (ROUGE, BLEU, MAUVE, etc.)

4️⃣ Log the training run via **wandb**
