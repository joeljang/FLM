# Fine-tuning (Large) Language Models (under progress ⏳)

### Key Features 🔑

1️⃣ Full use of computational resources (GPU Utilization)

- bf16, model parallelism, most optimal GPU utilization
- Support [Parameter Efficient Fine-tuning (PEFT)](https://github.com/huggingface/peft) with underlying PLM loaded in lower bits (int8())

2️⃣ Dynamically construct ANY kind of Training & Evaluation Dataset Combination

- Designate any # of instances for each training & validation dataset
- Allows instruction tuning & dynamic validation during training (beats T0 by +00%)
- Support all pre-training objective & fine-tuning objective (e.g. MLM, SSM, etc.)

3️⃣ Evaluate LLMs (even 175B LMs) on any kind of evaluation datasets (MMLU, BigBench, etc.) with [FlexGen](https://github.com/FMInference/FlexGen) support

- Support inference with both decoder-only LMs & encoder-decoder LMs
- Supports multiple types of verbalizer techniques (calibrazation, etc.)
- Supports all kinds of generative metrics (ROUGE, BLEU, MAUVE, etc.)

4️⃣ Log the training run via **wandb**

### 0. Install Dependencies
```
# install torch with the correct cuda version, check nvcc --version
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 --upgrade
# install Hugging Face Libraries
pip install "transformers==4.26.0" "datasets==2.9.0" "accelerate==0.16.0" "evaluate==0.4.0" --upgrade
# install deepspeed and ninja for jit compilations of kernels
pip install "deepspeed==0.8.0" ninja --upgrade
# install additional dependencies needed for training
pip install rouge-score nltk py7zr tensorboard scikit-learn
pip install sentencepiece
pip install wandb
```
Also, get promptsource (currently getting the version that supports xP3)
```
cd third_party
git clone -b tr13 https://github.com/Muennighoff/promptsource.git
cd ..
```

### 1. Dataset Preparation
Run the following code for preparing pretraining data 
```
python make_dataset_pt.py --config dataset_configs/pretrain/ko.json
```

Run the following code for preparing fine-tuning & evaluation data
```
python make_dataset_ft.py --config dataset_configs/finetune/basic.json
```

### 2. Train or Evaluate any LMs in huggingface

Before training or evaluating, we will tokenize the train/eval dataset before the actaul training. After filling out the configurations in 'run_configs' directory, run the following code for preprocessing the train/eval data
```
python preprocess_dataset.py --config run_configs/train/basic.json
```

Finally, run the run.py file! Happy training & evaluating :) ('localhost:{num1, num2' to designate the CUDA_VISIBLE_DEVICES=num1, num2. Code will use all GPUs as default)
```
deepspeed --include localhost:0 run.py --config run_configs/train/basic.json
```