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
# text = """The following are multiple choice questions (with answers) about computer science. You should end your answer by the sentence 'The answer is (X).' where you should replace 'X' in the sentence by 'A', 'B', 'C', or 'D' indicating your choice.
#
# Q: The Singleton design pattern is used to guarantee that only a single instance of a class may be instantiated. Which of the following is (are) true of this design pattern?
# I. The Singleton class has a static factory method to provide its instance.
# II. The Singleton class can be a subclass of another class.
# III. The Singleton class has a private constructor.
# (A) I only
# (B) II only
# (C) III only
# (D) I, II, and III
# A: Let's think step by step."""

# Few-shot CoT
text = """The following are multiple choice questions (with answers) about college computer science.\n\nQ: Which of the following regular expressions is equivalent to (describes the same set of strings as) (a* + b)*(c + d)?\n(A) a*(c + d)+ b(c + d)\n(B) a*(c + d)* + b(c + d)*\n(C) a*(c + d)+ b*(c + d)\n(D) (a + b)*c +(a + b)*d\nA: Let's think step by step. We know that:\n1. (X* + Y)* = (X + Y)*\n2. X(Y + Z)? = XY + XZ\nUsing equation 1 we can rewrite (a* + b)*(c + d)? as:\n3. (a + b)*(c + d)?\nUsing equation 2 we can rewrite equation 3 as:\n(a + b)*c + (a + b)*d The answer is (D).\n\nQ: A certain pipelined RISC machine has 8 general-purpose registers R0, R1, . . . , R7 and supports the following operations:\nADD Rs1, Rs2, Rd (Add Rs1 to Rs2 and put the sum in Rd)\nMUL Rs1, Rs2, Rd (Multiply Rs1 by Rs2 and put the product in Rd)\nAn operation normally takes one cycle; however, an operation takes two cycles if it produces a result required by the immediately following operation in an operation sequence.\nConsider the expression AB + ABC + BC, where variables A, B, C are located in registers R0, R1, R2. If the contents of these three registers must not be modified, what is the minimum number of clock cycles required for an operation sequence that computes the value of AB + ABC + BC?\n(A) 5 (B) 6 (C) 7 (D) 8\nA: Let's think step by step. First, we are given that A is in R0, B is in R1, and C is in R2.\nNext, we can see that we must compute three multiplies (AB, BC, and ABC) and two adds (AB + ABC, (AB + ABC) + BC) to compute our final answer, resulting in a minimum of five clock cycles.\nNext, we can see that there is no way to avoid at least one pipeline stall when computing our final answer, because to compute our final sum we must wait at least one cycle for the results from the previous stage to be ready. Thus, our minimum number of cycles must be 6.\nWe can verify that we can create a solution that requires only six cycles as follows:\ncompute AB: MUL R0, R1, R3\ncompute BC: MUL R1, R2, R4\ncompute ABC: MUL R3, R4, R5\ncompute AB + BC: ADD R3, R4, R6\nSTALL\ncompute AB + ABC + BC: ADD R5, R6, R7\nSo there are 6 cycles. The answer is (B).\n\nQ: A compiler generates code for the following assignment statement.\nG := (A + B) * C - (D + E) * F\nThe target machine has a single accumulator and a single-address instruction set consisting of instructions load, store, add, subtract, and multiply. For the arithmetic operations, the left operand is taken from the accumulator and the result appears in the accumulator. The smallest possible number of instructions in the resulting code is\n(A) 5 (B) 6 (C) 7 (D) 9\nA: Let's think step by step. We can compute the final answer with the following sequence of operations:\n1. LOAD D  (accumulator = D)\n2. ADD E  (accumulator = D+E)\n3. MUL F  (accumulator = (D+E)*F)\n4. STORE X (X = (D+E)*F)\n5. LOAD A  (accumulator = A)\n6. ADD B  (accumulator = A+B)\n7. MUL C  (accumulator = (A+B)*C)\n8. SUB X  (accumulator = (A+B)*C - (D+E)*F)\n9. STORE G (G = (A+B)*C - (D+E)*F)\nThis sequence takes 9 instructions. The answer is (D).\n\nQ: Consider a computer design in which multiple processors, each with a private cache memory, share global memory using a single bus. This bus is the critical system resource. Each processor can execute one instruction every 500 nanoseconds as long as memory references are satisfied by its local cache. When a cache miss occurs, the processor is delayed for an additional 2,000 nanoseconds. During half of this additional delay, the bus is dedicated to serving the cache miss. During the other half, the processor cannot continue, but the bus is free to service requests from other processors. On average, each instruction requires 2 memory references. On average, cache misses occur on 1 percent of references. What proportion of the capacity of the bus would a single processor consume, ignoring delays due to competition from other processors?\n(A) 1/50 (B) 1/27 (C) 1/25 (D) 2/27\nA: Let's think step by step. We know that each instruction requires two memory references per instruction, and that there is an average cache miss rate of one percent.\nThus a given processor has:\n(1 cache miss / 100 references) * (2 references / instruction) =\n(2 cache misses / 100 instructions), so:\nmisses_per_instruction = 1 cache miss / 50 instructions.\nNext, we know that each instruction requires 500 nanoseconds when there is no cache miss, and 500 + 2000 = 2500 nanoseconds when there is a cache miss. Thus:\n50 instructions / (49 * 500) + (1 * 2500) nanoseconds, so:\ninstructions_per_ns = 50 instructions / 27000 nanoseconds.\nNow, we know that each cache miss locks the bus for half of the 2000 nanosecond cache miss delay, or 1000 nanoseconds, so:\nlock_ns_per_miss = 1000 nanoseconds / cache miss.\nThus we can see that on average a single processor will lock the bus for:\nlock_ns_per_miss * misses_per_instruction * instructions_per_ns =\n(1000 nanoseconds / cache miss) * (1 cache miss / 50 instructions) * (50 instructions / 27000 nanoseconds) = 1000 * (1/50) * (50/27000) = 1000/27000 = 1/27. The answer is (B).

The following are multiple choice questions (with answers) about computer science.

Q: The Singleton design pattern is used to guarantee that only a single instance of a class may be instantiated. Which of the following is (are) true of this design pattern?
I. The Singleton class has a static factory method to provide its instance.
II. The Singleton class can be a subclass of another class.
III. The Singleton class has a private constructor.
(A) I only
(B) II only
(C) III only
(D) I, II, and III
A: Let's think step by step.
"""


# # Task-agnostic few-shot prompting
# text = """The following are multiple choice questions (with answers) about logical fallacies.
#
# Q: When an arguer causes confusion during refutation because of real or feigned lack of an ability to engage in refutation, that arguer may have committed the fallacy of
# (A) poor sportsmanship (B) appeal to compassion (C) argument against the person (D) ignorance of refutation
# A: Let's think step by step. We refer to Wikipedia articles on logical fallacies for help. Ignorance of refutation, one of Aristotle's original list of logical fallacies in his Organon, is when someone causes confusion in an argument through real or feigned inability to engage in refutation, in order to win the argument. The answer is (D).
#
# The following are multiple choice questions (with answers) about nutrition.
#
# Q: What is the first-line drug for patients with type 2 diabetes and obesity, as of 2020?
# (A) Acarbose (B) Metformin (C) Sulphonylureas (D) Insulin
# A: Let's think step by step. We refer to Wikipedia articles on nutrition for help. Metformin (Fortamet, Glumetza, or others) is usually the first medication prescribed for type 2 diabetes, as well as obesity. It works by lowering glucose production in the liver and improving the body's sensitivity to insulin. The answer is (B).
#
# The following are multiple choice questions (with answers) about virology.
#
# Q: The median survival time to AIDS and death was established by following:
# (A) Seroprevalent HIV-infected individuals (B) Seronegatives (C) Seroconverters (D) High-risk seronegatives
# A: Let's think step by step. We refer to Wikipedia articles on virology for help. The median survival time to AIDS and death was established as a result of the development of seroconverters. The answer is (C).
#
# The following are multiple choice questions (with answers) about computer security.
#
# Q: SHA-1 has a message digest of
# (A) 160 bits (B) 512 bits (C) 628 bits (D) 820 bits
# A: Let's think step by step. Since SHA-1 is a hash function which takes an input and produces a 160-bit (20-byte) hash value, its message digest is 160 bits. The answer is (A).
#
# The following are multiple choice questions (with answers) about professional accounting.
#
# Q: An auditor traces the serial numbers on equipment to a nonissuer’s subledger. Which of the following management assertions is supported by this test?
# (A) Valuation and allocation (B) Completeness (C) Rights and obligations (D) Presentation and disclosure
# A: Let's think step by step. We refer to Wikipedia articles on accounting for help. The completeness assertion is tested by tracing supporting documents to the record entries. The answer is (B).
#
# The following are multiple choice questions (with answers) about college computer science.
#
# Q: The Singleton design pattern is used to guarantee that only a single instance of a class may be instantiated. Which of the following is (are) true of this design pattern?
# I. The Singleton class has a static factory method to provide its instance.
# II. The Singleton class can be a subclass of another class.
# III. The Singleton class has a private constructor.
# (A) I only
# (B) II only
# (C) III only
# (D) I, II, and III
# A: Let's think step by step."""


# With structured
# text = """The following are multiple choice questions (with answers) about logical fallacies.
#
# Q: When an arguer causes confusion during refutation because of real or feigned lack of an ability to engage in refutation, that arguer may have committed the fallacy of
# (A) poor sportsmanship (B) appeal to compassion (C) argument against the person (D) ignorance of refutation
# A: Let's think step by step and analyze each option:
# - (A) Poor sportsmanship: This is more about behavior in competition than argumentation.
# - (B) Appeal to compassion: This involves eliciting sympathy to win an argument, not confusion.
# - (C) Argument against the person: This is attacking the person, not their inability to argue.
# - (D) Ignorance of refutation: This is when someone can't engage in refutation, either genuinely or pretending. It matches the scenario described.
# The answer is (D).
#
# The following are multiple choice questions (with answers) about nutrition.
#
# Q: What is the first-line drug for patients with type 2 diabetes and obesity, as of 2020?
# (A) Acarbose (B) Metformin (C) Sulphonylureas (D) Insulin
# A: Let's think step by step and analyze each option:
# - (A) Acarbose: This is used for diabetes but isn't usually the first-line treatment.
# - (B) Metformin: Known to be the first-line treatment for type 2 diabetes and obesity.
# - (C) Sulphonylureas: Used in diabetes treatment but not typically first-line.
# - (D) Insulin: Often used but not usually the first choice for type 2 diabetes.
# The answer is (B).
#
# The following are multiple choice questions (with answers) about virology.
#
# Q: The median survival time to AIDS and death was established by following:
# (A) Seroprevalent HIV-infected individuals (B) Seronegatives (C) Seroconverters (D) High-risk seronegatives
# A: Let's think step by step and analyze each option:
# - (A) Seroprevalent HIV-infected individuals: These individuals already have HIV but the question is about establishing median survival time.
# - (B) Seronegatives: These are individuals who do not have HIV, so they would not be relevant for establishing survival time to AIDS.
# - (C) Seroconverters: These are individuals who have recently converted from being HIV-negative to HIV-positive, making them relevant for tracking the progression to AIDS.
# - (D) High-risk seronegatives: Like seronegatives, they do not have HIV and are not relevant for determining AIDS survival time.
# The answer is (C).
#
# The following are multiple choice questions (with answers) about computer security.
#
# Q: SHA-1 has a message digest of
# (A) 160 bits (B) 512 bits (C) 628 bits (D) 820 bits
# A: Let's think step by step and analyze each option:
# - (A) 160 bits: SHA-1 is designed to produce a 160-bit hash value, making this option correct.
# - (B) 512 bits: This is incorrect as SHA-1 produces a smaller digest size.
# - (C) 628 bits: This size is not standard for any well-known hash function.
# - (D) 820 bits: This is also not a standard size for hash functions like SHA-1.
# The answer is (A).
#
# The following are multiple choice questions (with answers) about professional accounting.
#
# Q: An auditor traces the serial numbers on equipment to a nonissuer’s subledger. Which of the following management assertions is supported by this test?
# (A) Valuation and allocation (B) Completeness (C) Rights and obligations (D) Presentation and disclosure
# A: Let's think step by step and analyze each option:
# - (A) Valuation and allocation: This involves assessing the value and allocation of assets, not necessarily their existence or recording.
# - (B) Completeness: Tracing serial numbers to a subledger supports the assertion of completeness, ensuring all assets are recorded.
# - (C) Rights and obligations: This relates to whether the entity has the rights to the asset and the obligations related to it, not about recording completeness.
# - (D) Presentation and disclosure: This is about the appropriate presentation and disclosure in financial statements, not about the recording process.
# The answer is (B).
#
# The following are multiple choice questions (with answers) about computer science.
#
# Q: The Singleton design pattern is used to guarantee that only a single instance of a class may be instantiated. Which of the following is (are) true of this design pattern?
# I. The Singleton class has a static factory method to provide its instance.
# II. The Singleton class can be a subclass of another class.
# III. The Singleton class has a private constructor.
# (A) I only
# (B) II only
# (C) III only
# (D) I, II, and III
# A: Let's think step by step and analyze each option:"""


# Per-choice reasoning + reflection

text = """The following are multiple choice questions (with answers) about logical fallacies.

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

 The following are multiple choice questions (with answers) about computer science.

Q: The Singleton design pattern is used to guarantee that only a single instance of a class may be instantiated. Which of the following is (are) true of this design pattern?
I. The Singleton class has a static factory method to provide its instance.
II. The Singleton class can be a subclass of another class.
III. The Singleton class has a private constructor.
(A) I only
(B) II only
(C) III only
(D) I, II, and III
A: Let's think step by step and analyze each option:"""


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
