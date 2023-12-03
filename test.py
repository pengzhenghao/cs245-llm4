import torch
import transformers
from transformers import AutoTokenizer

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

text = """The following are multiple choice questions (with answers) about high school statistics.

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
A: Let's think step by step. Statement A is false because the slope of the regression line being exactly 1 can occur even when the two variables are not perfectly correlated. Statement B is false because uncorrelated variables regression lines can have slope zero. Statement C is false because correlation is symmetric in the two random variables. The answer is (D).

Q: What are the mean and standard deviation of a binomial experiment that occurs with probability of success 0.76 and is repeated 150 times?
(A) 114, 27.35 (B) 100.5, 5.23 (C) 114, 5.23 (D) The mean is 114, but there is not enough information given to determine the standard deviation.
A: Let's think step by step."""


text = """The following are multiple choice questions (with answers) about high school statistics. You will analyze the problem and each choice. You should end your answer by the sentence 'The answer is (X).' where you should replace 'X' in the sentence by 'A', 'B', 'C', or 'D' indicating your choice.

Q: What are the mean and standard deviation of a binomial experiment that occurs with probability of success 0.76 and is repeated 150 times?
(A) 114, 27.35 (B) 100.5, 5.23 (C) 114, 5.23 (D) The mean is 114, but there is not enough information given to determine the standard deviation.
A: Let's think step by step."""

text = """Before answering each question, first reason the problem and each choice. Then reflect on your approach and knowledge relevant to the topic. After reflection, provide your answer directly. You should end your answer by the sentence 'The answer is (X).' where you should replace 'X' in the sentence by 'A', 'B', 'C', or 'D' indicating your choice. For example, \nQ: A new smartwatch is manufactured in one part of a factory, then secured for shipping in another, independent part of the factory. The weight of the smartwatch has a mean of 62 grams and a standard deviation of 1.0 grams. The weight of the packaging (box, user's guide, bubble wrap, etc.) has a mean of 456 grams and a standard deviation of 6 grams. What is the mean and standard deviation of the total weight?\n(A) Mean 518 grams; standard deviation 7.0 grams (B) Mean 518 grams; standard deviation 3.5 grams (C) Mean 518 grams; standard deviation 6.1 grams (D) Mean 394 grams; standard deviation 6.1 grams\nReasoning: Let's think step by step. Since the weight of the watch and the packaging are independent random variables, we add their means to find the total mean, which is 62 + 456 = 518 grams. To find the standard deviation, we sum the squares of the individual standard deviations (since they are independent), which gives 1.0^2 + 6.0^2 = 37, and then take the square root, resulting in approximately 6.1 grams.\nReflection: In reasoning through this question, I applied statistical principles for independent variables and was careful to correctly apply the formulas for mean and standard deviation. An alternative method might have involved using a statistical software for computation, but the manual calculation was straightforward in this case.\nA: The answer is (C).\n\nQ: What are the mean and standard deviation of a binomial experiment that occurs with probability of success 0.76 and is repeated 150 times?
(A) 114, 27.35 (B) 100.5, 5.23 (C) 114, 5.23 (D) The mean is 114, but there is not enough information given to determine the standard deviation.
Reasoning: Let's think step by step."""

# Zero-shot CoT with Instruction
text = """The following are multiple choice questions (with answers) about anatomy. You will analyze the problem and each choice. You should end your answer by the sentence 'The answer is (X).' where you should replace 'X' in the sentence by 'A', 'B', 'C', or 'D' indicating your choice.

Q: Which of the following terms describes the body's ability to maintain its normal state?
(A) Anabolism (B) Catabolism (C) Tolerance (D) Homeostasis
A: Let's think step by step."""

# Task-agnostic few-shot prompting
text = """The following are multiple choice questions (with answers) about logical fallacies.

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
A: Let's think step by step."""

# with Reflection
text = """The following are multiple choice questions (with answers) about logical fallacies, along with a reflective pause to consider our reasoning process.

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
A: Let's think step by step."""

sequences = pipeline(
    text,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=4096,
)
print("======")
for seq in sequences:
    print(f"{seq['generated_text']}")
    print('======')
