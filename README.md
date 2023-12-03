# LLM4: The Impact of Prompting Strategies

This repo contains the code to reproduce our experiments for the project LLM4 in the 
COM SCI 245 Big Data Analysis 2023 Fall at UCLA.


Members:
* Zhenghao Peng
* Haoting Ni
* Shihao Yang
* Yuanxuan Fang


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
> You might need to create a [hugging face](https://huggingface.co/) account and [generate](https://huggingface.co/settings/tokens) a read token.

### Install this repo itself

To install this repo, please run:
```bash
cd cs245-llm4/

pip install -e .
```

## Experiments

> [!NOTE]
> The below scripts launch experiment with `accelerate launch -m lm-eval --model ...` for multi-GPUs acceleration.
> If you want to run in single GPU, use `lm-eval --model ...` instead.

Note that running the following script, we will (1) download the LLaMA-2 model, (2) download the MMLU datasets and (3) evaluate the LLaMA-2-7b model in the MMLU dataset.



### Exp 1: Baseline experiments

**Features:**
* Zero-shot prompting

```bash
accelerate launch -m lm_eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu \
--output_path evaluation_results/exp1_baseline
```


<details>
<summary><b>Example Prompt:</b></summary>

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
</details>




### Exp 2: Few-shot (k=3, 5) Prompting


This experiment evaluates the Basic Prompting Strategy 1.

**Features:**
* 3-shot and 5-shot prompting

```bash
accelerate launch -m lm_eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu \
--num_fewshot 3 \
--output_path evaluation_results/exp2_3shot

accelerate launch -m lm_eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu \
--num_fewshot 5 \
--output_path evaluation_results/exp2_5shot
```

<details>
<summary><b>Example Prompt:</b></summary>

```plain
The following are multiple choice questions (with answers) about high school statistics.

Which of the following is a correct statement about correlation?
A. If the slope of the regression line is exactly 1, then the correlation is exactly 1.
B. If the correlation is 0, then the slope of the regression line is undefined.
C. Switching which variable is called x and which is called y changes the sign of the correlation.
D. The correlation r is equal to the slope of the regression line when z-scores for the y-variable are plotted against z-scores for the x-variable.
Answer: D

Suppose X and Y are random variables with E(X) = 37, var(X) = 5, E(Y) = 62, and var(Y) = 12. What are the expected value and variance of the random variable X + Y?
A. E(X + Y) = 99, var(X + Y) = 8.5
B. E(X + Y) = 99, var(X + Y) = 13
C. E(X + Y) = 99, var(X + Y) = 17
D. There is insufficient information to answer this question.
Answer: D

After a frost warning was issued, the owner of a large orange grove asked his workers to spray all his trees with water. The water was supposed to freeze and form a protective covering of ice around the orange blossom. Nevertheless, the owner suspected that some trees suffered considerable damage due to the frost. To estimate the proportion of trees that suffered more than 50 percent damage due to the frost, he took a random sample of 100 trees from his grove. What is the response variable in this experiment?
A. The proportion of trees that suffered more than 50 percent damage due to frost.
B. The number of trees affected by the frost.
C. The number of trees sampled from the grove.
D. For each sampled tree, whether it suffered more than 50 percent damage or at most 50 percent damage.
Answer: D

A new smartwatch is manufactured in one part of a factory, then secured for shipping in another, independent part of the factory. The weight of the smartwatch has a mean of 62 grams and a standard deviation of 1.0 grams. The weight of the packaging (box, user's guide, bubble wrap, etc.) has a mean of 456 grams and a standard deviation of 6 grams. Together, the distribution of the weight of the smartwatch and its packaging would have the following mean and standard deviation:
A. Mean 518 grams; standard deviation 7.0 grams
B. Mean 518 grams; standard deviation 3.5 grams
C. Mean 518 grams; standard deviation 6.1 grams
D. Mean 394 grams; standard deviation 6.1 grams
Answer: C

Which of the following sets has the smallest standard deviation? Which has the largest?
I: {1,2,3}
II: {-10,10}
III: {100}
A. I, II
B. II, III
C. III, I
D. III, II
Answer: D

The weight of an aspirin tablet is 300 milligrams according to the bottle label. An FDA investigator weighs a simple random sample of seven tablets, obtains weights of 299, 300, 305, 302, 299, 301, and 303, and runs a hypothesis test of the manufacturer's claim. Which of the following gives the P-value of this test?
A. P(t > 1.54) with df = 6
B. 2P(t > 1.54) with df = 6
C. P(t > 1.54) with df = 7
D. 2P(t > 1.54) with df = 7
Answer:
```
</details>



### Exp 3: Zero-shot Chain-of-Thoughts Prompting

```bash
accelerate launch -m lm_eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu_flan_cot_zeroshot \
--output_path evaluation_results/exp3_0shot_cot
```


<details>
<summary><b>Example Prompt:</b></summary>

```plain
The following are multiple choice questions (with answers) about conceptual physics.

Q: A real image can be cast on a screen by a
(A) converging lens (B) diverging lens (C) Either of these (D) Neither of these
A: Let's think step by step.
```
</details>



### Exp 4: Few-shot Chain-of-Thoughts Prompting

```bash
accelerate launch -m lm_eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu_flan_cot_fewshot \
--output_path evaluation_results/exp4_fewshot_cot
```


<details>
<summary><b>Example Prompt:</b></summary>

```plain
The following are multiple choice questions (with answers) about high school statistics.

Q: A new smartwatch is manufactured in one part of a factory, then secured for shipping in another, independent part of the factory. The weight of the smartwatch has a mean of 62 grams and a standard deviation of 1.0 grams. The weight of the packaging (box, user's guide, bubble wrap, etc.) has a mean of 456 grams and a standard deviation of 6 grams. Together, the distribution of the weight of the smartwatch and its packaging would have the following mean and standard deviation:
(A) Mean 518 grams; standard deviation 7.0 grams (B) Mean 518 grams; standard deviation 3.5 grams (C) Mean 518 grams; standard deviation 6.1 grams (D) Mean 394 grams; standard deviation 6.1 grams
A: Let's think step by step. Since the weight of the watch and the weight of the packaging are independent random variables, the mean and variance of their sum is equal to the sum of their individual means and variances. So the mean is 62 + 456 = 518 grams, and the variances is 1.0^2 + 6.0^2 = 37, leading to a standard deviation of 6.1 grams. The answer is (C).

Q: After a frost warning was issued, the owner of a large orange grove asked his workers to spray all his trees with water. The water was supposed to freeze and form a protective covering of ice around the orange blossom. Nevertheless, the owner suspected that some trees suffered considerable damage due to the frost. To estimate the proportion of trees that suffered more than 50 percent damage due to the frost, he took a random sample of 100 trees from his grove. What is the response variable in this experiment?
(A) The proportion of trees that suffered more than 50 percent damage due to frost. (B) The number of trees affected by the frost. (C) The number of trees sampled from the grove. (D) For each sampled tree, whether it suffered more than 50 percent damage or at most 50 percent damage.
A: Let's think step by step. In this experiment, the response variable is what is measured. For each tree, what is measured is whether or not it suffered more than 50 percent damage due to the frost. The answer is (D).

Q: Suppose X and Y are random variables with E(X) = 37, var(X) = 5, E(Y) = 62, and var(Y) = 12. What are the expected value and variance of the random variable X + Y?
(A) E(X + Y) = 99, var(X + Y) = 8.5 (B) E(X + Y) = 99, var(X + Y) = 13 (C) E(X + Y) = 99, var(X + Y) = 17 (D) There is insufficient information to answer this question.
A: Let's think step by step. While means of sums of random variables add (regardless of whether the variables are independent) in order to determine the variance of a sum of random variables, we need to know not just their individual variances but the covariance of the two variables, which is not given in this problem. The answer is (D).

Q: Which of the following sets has the smallest standard deviation? Which has the largest?
I: {1,2,3}
II: {-10,10}
III: {100}
(A) I, II (B) II, III (C) III, I (D) III, II
A: Let's think step by step. The variance of distribution I is the expected squared deviation from its mean (which is 2), so the variance is 2/3 . The variance of distribution II is 10^2 (because both elements are 10 away from the mean of zero). The variance of distribution III is 0, since it has a single entry. So distribution III has the smallest standard deviation and distribution II has the largest. The answer is (D).

Q: Which of the following is a correct statement about correlation?
(A) If the slope of the regression line is exactly 1, then the correlation is exactly 1. (B) If the correlation is 0, then the slope of the regression line is undefined. (C) Switching which variable is called x and which is called y changes the sign of the correlation. (D) The correlation r is equal to the slope of the regression line when z-scores for the y-variable are plotted against z-scores for the x-variable.
A: Let's think step by step. Statement A is false because the slope of the regression line being exactly 1 can occur even when the two variables are not perfectly correlated. Statement B is false because uncorrelated variables regression lines can have slope zero. Statement C is false because correlation is symmetric in the two random variables. The answer is (D).Q: What are the mean and standard deviation of a binomial experiment that occurs with probability of success 0.76 and is repeated 150 times?
(A) 114, 27.35 (B) 100.5, 5.23 (C) 114, 5.23 (D) The mean is 114, but there is not enough information given to determine the standard deviation.
A: Let's think step by step.
```
</details>



## FAQ


### How to access the LLama2 model in Hugging Face?

1. Get approval from Meta.
2. Get approval from HF.
3. Create a read token from here : https://huggingface.co/settings/tokens
4. `pip install transformers`.
5. execute `huggingface-cli login` and provide the read token.
6. Execute your code. It should work fine.


### How can I debug the model and print example prompt?

You can set `--limit 2` to only process the first 2 documents. Set `--write_out` will also print the example prompt.
For example, to debug the baseline experiment, run:
```bash
python lm_eval/__main__.py \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--limit 2 \
--tasks mmlu \
--write_out

# Add other flags here to debug different experiments
```

## Reference


* Our project modifies the code from: https://github.com/EleutherAI/lm-evaluation-harness
* The LLM we use is from: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
