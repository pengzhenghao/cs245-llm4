# LLM4: The Impact of Prompting Strategies

This repo contains the code to reproduce our experiments for the project LLM4 in the 
COM SCI 245 Big Data Analysis 2023 Fall at UCLA.


Members:
* Zhenghao Peng
* Haoting Ni
* Shihao Yang


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


### Exp 2B: Task-agnostic Few-shot Prompting

**Features:**
* 3-shot and 5-shot prompting

```bash
accelerate launch -m lm_eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu_flan_cot_fewshot_global \
--output_path evaluation_results/exp2_taskagnostic_fewshot
```


```bash
accelerate launch -m lm_eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu_flan_cot_fewshot_global_10shot \
--output_path evaluation_results/exp2_taskagnostic_fewshot_10shot
```



<details>
<summary><b>Example Prompt:</b></summary>

```plain
The following are multiple choice questions (with answers) about logical fallacies.

Q: When an arguer causes confusion during refutation because of real or feigned lack of an ability to engage in refutation, that arguer may have committed the fallacy of
(A) poor sportsmanship (B) appeal to compassion (C) argument against the person (D) ignorance of refutation
A: Let's think step by step. We refer to Wikipedia articles on logical fallacies for help. Ignorance of refutation, one of Aristotle's original list of logical fallacies in his Organon, is when someone causes confusion in an argument through real or feigned inability to engage in refutation, in order to win the argument. The answer is (D).

The following are multiple choice questions (with answers) about nutrition.

Q: What is the first-line drug for patients with type 2 diabetes and obesity, as of 2020?
(A) Acarbose (B) Metformin (C) Sulphonylureas (D) Insulin
A: Let's think step by step. We refer to Wikipedia articles on nutrition for help. Metformin (Fortamet, Glumetza, or others) is usually the first medication prescribed for type 2 diabetes, as well as obesity. It works by lowering glucose production in the liver and improving the body's sensitivity to insulin. The answer is (B).

The following are multiple choice questions (with answers) about virology.

Q: The median survival time to AIDS and death was established by following:
(A) Seroprevalent HIV-infected individuals (B) Seronegatives (C) Seroconverters (D) High-risk seronegatives
A: Let's think step by step. We refer to Wikipedia articles on virology for help. The median survival time to AIDS and death was established as a result of the development of seroconverters. The answer is (C).

The following are multiple choice questions (with answers) about computer security.

Q: SHA-1 has a message digest of
(A) 160 bits (B) 512 bits (C) 628 bits (D) 820 bits
A: Let's think step by step. Since SHA-1 is a hash function which takes an input and produces a 160-bit (20-byte) hash value, its message digest is 160 bits. The answer is (A).

The following are multiple choice questions (with answers) about professional accounting.

Q: An auditor traces the serial numbers on equipment to a nonissuer’s subledger. Which of the following management assertions is supported by this test?
(A) Valuation and allocation (B) Completeness (C) Rights and obligations (D) Presentation and disclosure
A: Let's think step by step. We refer to Wikipedia articles on accounting for help. The completeness assertion is tested by tracing supporting documents to the record entries. The answer is (B).

The following are multiple choice questions (with answers) about college computer science.

Q: Sometimes the object module produced by a compiler includes information (from the symbol table) mapping all source program names to their addresses. The most likely purpose of this information is
(A) for use as input to a debugging aid (B) to increase the run-time efficiency of the program (C) for the reduction of the symbol-table space needed by the compiler (D) to tell the loader where each variable belongs
A: Let's think step by step.
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





### Exp 3B: Zero-shot Chain-of-Thoughts Prompting with Additional Instruction

```bash
accelerate launch -m lm_eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu_flan_cot_zeroshot_addoutputformat \
--output_path evaluation_results/exp3_0shot_cot_addoutputformat
```


<details>
<summary><b>Example Prompt:</b></summary>

```plain
The following are multiple choice questions (with answers) about anatomy. You will analyze the problem and each choice. You should end your answer by the sentence 'The answer is (X).' where you should replace 'X' in the sentence by 'A', 'B', 'C', or 'D' indicating your choice.

Q: Which of the following terms describes the body's ability to maintain its normal state?
(A) Anabolism (B) Catabolism (C) Tolerance (D) Homeostasis
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



## Few-shot Chain-of-Thoughts Prompting with Reflection


```bash
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 accelerate launch -m lm_eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu_flan_cot_fewshot_global_reflection \
--output_path evaluation_results/exp5_fewshot_cot_global_reflection
```


<details>
<summary><b>Example Prompt:</b></summary>

```plain
The following are multiple choice questions (with answers) about logical fallacies, along with a reflective pause to consider our reasoning process.

Q: When an arguer causes confusion during refutation because of real or feigned lack of an ability to engage in refutation, that arguer may have committed the fallacy of
(A) poor sportsmanship (B) appeal to compassion (C) argument against the person (D) ignorance of refutation
A: Let's think step by step. We refer to Wikipedia articles on logical fallacies for help. Ignorance of refutation, one of Aristotle's original list of logical fallacies in his Organon, is when someone causes confusion in an argument through real or feigned inability to engage in refutation, in order to win the argument. Reflecting on this, it seems crucial to distinguish between genuine misunderstanding and strategic confusion in arguments. The answer is (D).

The following are multiple choice questions (with answers) about nutrition, with a moment for reflection.

Q: What is the first-line drug for patients with type 2 diabetes and obesity, as of 2020?
(A) Acarbose (B) Metformin (C) Sulphonylureas (D) Insulin
A: Let's think step by step. We refer to Wikipedia articles on nutrition for help. Metformin (Fortamet, Glumetza, or others) is usually the first medication prescribed for type 2 diabetes, as well as obesity. It works by lowering glucose production in the liver and improving the body's sensitivity to insulin. Reflecting on this, it's clear how crucial medication can be in managing chronic conditions, and the importance of staying updated with medical guidelines. The answer is (B).

The following are multiple choice questions (with answers) about virology, with an opportunity to reflect on our learning process.

Q: The median survival time to AIDS and death was established by following:
(A) Seroprevalent HIV-infected individuals (B) Seronegatives (C) Seroconverters (D) High-risk seronegatives
A: Let's think step by step. We refer to Wikipedia articles on virology for help. The median survival time to AIDS and death was established as a result of the development of seroconverters. Reflecting on this, we see how epidemiological studies provide vital insights into disease progression and treatment outcomes. The answer is (C).

The following are multiple choice questions (with answers) about computer security, with a pause for reflection.

Q: SHA-1 has a message digest of
(A) 160 bits (B) 512 bits (C) 628 bits (D) 820 bits
A: Let's think step by step. Since SHA-1 is a hash function which takes an input and produces a 160-bit (20-byte) hash value, its message digest is 160 bits. Pausing to reflect, we understand the relevance of hash functions in ensuring data integrity and security. The answer is (A).

The following are multiple choice questions (with answers) about professional accounting, including a moment for self-reflection.

Q: An auditor traces the serial numbers on equipment to a nonissuer’s subledger. Which of the following management assertions is supported by this test?
(A) Valuation and allocation (B) Completeness (C) Rights and obligations (D) Presentation and disclosure
A: Let's think step by step. We refer to Wikipedia articles on accounting for help. The completeness assertion is tested by tracing supporting documents to the record entries. Reflecting on this, it's evident how meticulous and detailed accounting practices help in ensuring financial transparency and accountability. The answer is (B).

The following are multiple choice questions (with answers) about anatomy, with a moment for reflection.

Q: Sometimes the object module produced by a compiler includes information (from the symbol table) mapping all source program names to their addresses. The most likely purpose of this information is
(A) for use as input to a debugging aid (B) to increase the run-time efficiency of the program (C) for the reduction of the symbol-table space needed by the compiler (D) to tell the loader where each variable belongs
A: Let's think step by step.
```
</details>


## Few-shot Chain-of-Thoughts Prompting with Structured Reasoning




```bash
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 accelerate launch -m lm_eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu_flan_cot_fewshot_global_structured \
--output_path evaluation_results/exp6_fewshot_cot_global_structured
```


<details>
<summary><b>Example Prompt:</b></summary>

```plain
The following are multiple choice questions (with answers) about logical fallacies.

Q: When an arguer causes confusion during refutation because of real or feigned lack of an ability to engage in refutation, that arguer may have committed the fallacy of
(A) poor sportsmanship (B) appeal to compassion (C) argument against the person (D) ignorance of refutation
A: Let's think step by step and analyze each option:
- (A) Poor sportsmanship: This is more about behavior in competition than argumentation.
- (B) Appeal to compassion: This involves eliciting sympathy to win an argument, not confusion.
- (C) Argument against the person: This is attacking the person, not their inability to argue.
- (D) Ignorance of refutation: This is when someone can't engage in refutation, either genuinely or pretending. It matches the scenario described.
The answer is (D).

The following are multiple choice questions (with answers) about nutrition.

Q: What is the first-line drug for patients with type 2 diabetes and obesity, as of 2020?
(A) Acarbose (B) Metformin (C) Sulphonylureas (D) Insulin
A: Let's think step by step and analyze each option:
- (A) Acarbose: This is used for diabetes but isn't usually the first-line treatment.
- (B) Metformin: Known to be the first-line treatment for type 2 diabetes and obesity.
- (C) Sulphonylureas: Used in diabetes treatment but not typically first-line.
- (D) Insulin: Often used but not usually the first choice for type 2 diabetes.
The answer is (B).

The following are multiple choice questions (with answers) about virology.

Q: The median survival time to AIDS and death was established by following:
(A) Seroprevalent HIV-infected individuals (B) Seronegatives (C) Seroconverters (D) High-risk seronegatives
A: Let's think step by step and analyze each option:
- (A) Seroprevalent HIV-infected individuals: These individuals already have HIV but the question is about establishing median survival time.
- (B) Seronegatives: These are individuals who do not have HIV, so they would not be relevant for establishing survival time to AIDS.
- (C) Seroconverters: These are individuals who have recently converted from being HIV-negative to HIV-positive, making them relevant for tracking the progression to AIDS.
- (D) High-risk seronegatives: Like seronegatives, they do not have HIV and are not relevant for determining AIDS survival time.
The answer is (C).

The following are multiple choice questions (with answers) about computer security.

Q: SHA-1 has a message digest of
(A) 160 bits (B) 512 bits (C) 628 bits (D) 820 bits
A: Let's think step by step and analyze each option:
- (A) 160 bits: SHA-1 is designed to produce a 160-bit hash value, making this option correct.
- (B) 512 bits: This is incorrect as SHA-1 produces a smaller digest size.
- (C) 628 bits: This size is not standard for any well-known hash function.
- (D) 820 bits: This is also not a standard size for hash functions like SHA-1.
The answer is (A).

The following are multiple choice questions (with answers) about professional accounting.

Q: An auditor traces the serial numbers on equipment to a nonissuer’s subledger. Which of the following management assertions is supported by this test?
(A) Valuation and allocation (B) Completeness (C) Rights and obligations (D) Presentation and disclosure
A: Let's think step by step and analyze each option:
- (A) Valuation and allocation: This involves assessing the value and allocation of assets, not necessarily their existence or recording.
- (B) Completeness: Tracing serial numbers to a subledger supports the assertion of completeness, ensuring all assets are recorded.
- (C) Rights and obligations: This relates to whether the entity has the rights to the asset and the obligations related to it, not about recording completeness.
- (D) Presentation and disclosure: This is about the appropriate presentation and disclosure in financial statements, not about the recording process.
The answer is (B).

The following are multiple choice questions (with answers) about high school european history.

Q: This question refers to the following information.
"The Italian nation has been at last united in our own days, and we all rejoiced in its union. Yet we may be allowed to doubt whether the union was not a little too speedy and a little too thorough. It is surely carrying unity too far to wipe out all traces of the independent being, for most purposes to wipe out the very name, of such a land as Sicily. It jars on our feelings to find that, while Ireland at least forms part of the royal style of its sovereign, Sicily is no longer even a geographical expression. The island realm of Roger has sunk to be seven provinces of the kingdom on the mainland. And there is another result of Italian unity, a result in which we may rejoice without drawbacks, but which still has somewhat of sadness about it as finally ending that great phase of the history of Europe with which we have throughout been dealing. Never were ties with the past so fully snapped as when the army of Italy entered liberated Rome. Of all novelties in European history the greatest was when Rome became the centre of a dominion with acknowledged metes and bounds, the head in short of a local Italian kingdom. "Rome the capital of Italy" was a formula which might well gladden our hearts; but it was a formula which formally swept away the œcumenical position, the œcumenical traditions, of Rome&….
But the kingdom of Italy is not an appendage to Rome; Rome is the head of the kingdom. The whole is greater than its part; Rome, by her own free will and by the free will of Italy, has become less than Italy. By becoming the willing head of an Italian kingdom she has formally cast aside her Imperial traditions as they were not cast aside when brute force made her the head of a French department."
Edward A. Freeman, British historian and politician, The Chief Periods of European History, 1885
The author would likely take exception to the methods of which leader during this time period in Italy?
(A) Napoleon III (B) Giuseppe Garibaldi (C) Victor Emmanuel (D) Count Cavour
A: Let's think step by step and analyze each option:
```
</details>



## Few-shot Chain-of-Thoughts Prompting with Structured Reasoning + Reflection




```bash
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 accelerate launch -m lm_eval \
--model hf \
--model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=float16 \
--tasks mmlu_flan_cot_fewshot_global_structured_reflection \
--output_path evaluation_results/exp7_fewshot_cot_global_structured_reflection
```


<details>
<summary><b>Example Prompt:</b></summary>

```plain
The following are multiple choice questions (with answers) about logical fallacies.

Q: When an arguer causes confusion during refutation because of real or feigned lack of an ability to engage in refutation, that arguer may have committed the fallacy of
(A) poor sportsmanship (B) appeal to compassion (C) argument against the person (D) ignorance of refutation
A: Let's think step by step and analyze each option:
- (A) Poor sportsmanship: More about competitive behavior, not relevant to argumentation.
- (B) Appeal to compassion: Involves eliciting sympathy, not related to the ability to argue.
- (C) Argument against the person: Focuses on attacking the person, not their argument skills.
- (D) Ignorance of refutation: Directly relates to the inability to engage in refutation, as described.
Reflecting on these options, it's clear that option (D) is the only one that directly addresses the issue of confusion during refutation due to inability. The answer is (D).

The following are multiple choice questions (with answers) about nutrition.

Q: What is the first-line drug for patients with type 2 diabetes and obesity, as of 2020?
(A) Acarbose (B) Metformin (C) Sulphonylureas (D) Insulin
A: Let's think step by step and analyze each option:
- (A) Acarbose: Used for diabetes, but not primarily for first-line treatment.
- (B) Metformin: Widely recognized as the first-line treatment for type 2 diabetes and obesity.
- (C) Sulphonylureas: Secondary treatment option, not typically first-line.
- (D) Insulin: Usually a subsequent choice, not the first line of treatment.
Reflecting on these options, Metformin stands out as the most appropriate first-line treatment, aligning with current medical guidelines. The answer is (B).

The following are multiple choice questions (with answers) about virology.

Q: The median survival time to AIDS and death was established by following:
(A) Seroprevalent HIV-infected individuals (B) Seronegatives (C) Seroconverters (D) High-risk seronegatives
A: Let's think step by step and analyze each option:
- (A) Seroprevalent HIV-infected individuals: Already have HIV but not specifically relevant to median survival time.
- (B) Seronegatives: These individuals do not have HIV, not applicable for AIDS survival time.
- (C) Seroconverters: Relevant for tracking the progression to AIDS from the point of HIV conversion.
- (D) High-risk seronegatives: While at risk, they do not provide direct information about AIDS survival.
Reflecting on these options, seroconverters are most relevant for establishing the median survival time to AIDS, as they represent the group where the progression of the disease can be tracked from the point of HIV infection. The answer is (C).

The following are multiple choice questions (with answers) about computer security.

Q: SHA-1 has a message digest of
(A) 160 bits (B) 512 bits (C) 628 bits (D) 820 bits
A: Let's think step by step and analyze each option:
- (A) 160 bits: SHA-1 is designed to produce a 160-bit hash value, making this option correct.
- (B) 512 bits: Incorrect as SHA-1 produces a smaller digest size.
- (C) 628 bits: Not a standard size for hash functions.
- (D) 820 bits: Also not a standard size for hash functions like SHA-1.
Reflecting on these options, it's evident that SHA-1 is known for its 160-bit hash value, aligning with option (A). The answer is (A).

The following are multiple choice questions (with answers) about professional accounting.

Q: An auditor traces the serial numbers on equipment to a nonissuer’s subledger. Which of the following management assertions is supported by this test?
(A) Valuation and allocation (B) Completeness (C) Rights and obligations (D) Presentation and disclosure
A: Let's think step by step and analyze each option:
- (A) Valuation and allocation: Concerns the value and allocation of assets, not their recording.
- (B) Completeness: Ensuring all assets are recorded, directly supported by tracing serial numbers.
- (C) Rights and obligations: About the entity's rights to the asset and related obligations, not recording completeness.
- (D) Presentation and disclosure: Involves financial statement presentation, not the recording process.
Reflecting on these options, the purpose of tracing serial numbers to a subledger supports the completeness assertion, as it ensures all assets are accounted for. The answer is (B).

The following are multiple choice questions (with answers) about anatomy.

Q: Which of the following terms describes the body's ability to maintain its normal state?
(A) Anabolism (B) Catabolism (C) Tolerance (D) Homeostasis
A: Let's think step by step and analyze each option:
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
--write_out \
--verbosity DEBUG 
# Add other flags here to debug different experiments
```

## Reference


* Our project modifies the code from: https://github.com/EleutherAI/lm-evaluation-harness
* The LLM we use is from: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf