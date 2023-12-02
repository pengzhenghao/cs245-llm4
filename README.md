# CS245 Project for LLM4

Members:
* Zhenghao Peng (105951555)
* Haoting Ni (905545789)
* Shihao Yang (205945583)


## Getting started


### Setup conda environment

```bash
# Create conda environment:
conda create -n llm4 python=3.11

# Activate conda environment:
conda activate llm4
```

### Install dependencies

Install Pytorch first:
```bash
pip install torch

# To address the compatibility issue for running Pytorch
# with different GPU, we suggest to install Pytorch by
# carefully selected a compatible version: 
# https://pytorch.org/get-started/locally/
```

Install Hugging Face related packages:
```bash
pip install transformers accelerate datasets
```

Login your hugging face account:
```bash
huggingface-cli login
```

> [!NOTE]
> You might need to create a [hugging face](https://huggingface.co/) account and [generate](https://huggingface.co/settings/tokens) a read token:
> ![](figs/hf-token.png)


### Install this repo itself


Our project modifies the code from https://github.com/EleutherAI/lm-evaluation-harness
To install this repo, please use:
```bash
cd cs245-llm4/

pip install -e .
```

## Experiments

> [!NOTE]
> The below scripts launch experiment with `accelerate launch -m lm-eval --model ...` for multi-GPUs acceleration.
> If you want to run in single GPU, use `lm-eval --model ...` instead.


### Exp 1: Baseline experiments

**Features:**
* Zero-shot prompting

> [!NOTE]
> By running the following script, we: 1) download the LLaMA-2 model, 2) download the MMLU datasets and 3) evaluate the LLaMA-2-7b model in the MMLU dataset.


```bash
accelerate launch -m lm-eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu \
--write_out \
--output_path evaluation_results/exp1_baseline
```

**Example Prompt:**

```plain
The following are multiple choice questions (with answers) about high school world history.

This question refers to the following information.
No task is more urgent than that of preserving peace. Without peace our independence means little. The rehabilitation and upbuilding of our countries will have little meaning. Our revolutions will not be allowed to run their course. What can we do? We can do much! We can inject the voice of reason into world affairs. We can mobilize all the spiritual, all the moral, all the political strength of Asia and Africa on the side of peace. Yes, we! We, the peoples of Asia and Africa, 1.4 billion strong.
Indonesian leader Sukarno, keynote address to the Bandung Conference, 1955
The passage above is most associated with which of the following developments?
A. The formation of the non-aligned movement
B. Global disarmanent and nuclear non-proliferation
C. The Green Revolution in agriculture
D. Mobilization of pan-Asian ideology
Answer:
```


### Exp 2: Few-shot (k=5) Prompting


This experiment evaluates the Basic Prompting Strategy 1.

**Features:**
* 5-shot prompting

```bash
lm-eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu \
--write_out \
--num_fewshot 5 \
--output_path evaluation_results/exp2_5shot
```






## FAQ


### How to access the LLama2 model in Hugging Face?

1. Get approval from Meta.
2. Get approval from HF.
3. Create a read token from here : https://huggingface.co/settings/tokens
4. `pip install transformers`.
5. execute `huggingface-cli login` and provide the read token.
6. Execute your code. It should work fine.


### How can I debug the model?

You can set `--limit 1` to only process the first 2 documents. For example, you can run:
```bash
python lm_eval/__main__.py \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--limit s \
--tasks mmlu \
--write_out
```
to fast debug the baseline experiment.