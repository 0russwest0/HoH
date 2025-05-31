QA_GENERATE_TEMPLATES = """Identify the contradiction between the two following sentences and generate a Q&A pair that reflects this contradiction. The question should be answerable based on each sentence, but the two answers should CONTRADICT EACH OTHER. You can reference the Source Content for broader context, but the Q&A pair should relate directly to the information in Old/New Sentence(s).

### Old Sentence(s)
{old_sentence}

### New Sentence(s)
{new_sentence}

### Source Content
{source_content}

Use the following instructions for generating a Q&A pair:
1) The question should be answerable based on each sentence.
2) Avoid using phrases like 'according to', 'as stated in', 'based on the information provided', 'as mentioned', or similar formulations; the question should be direct and straightforward.
3) Avoid using specific date references, such as 'as of [date]', 'in [year]', etc.
4) Avoid questions that specifically ask for when the information was last updated or modified. Instead, focus on content-related queries. Questions reliant on the inherent temporal nature of certain information are acceptable.
5) The question should precisely define the subject or object it refers to. Avoid vague terms or pronouns that necessitate additional context.
6) The question should stand alone as much as possible without requiring extra information not included in the QA pair. Ensure that the question is detailed enough for the audience to understand it without needing further clarification.
7) An answer should be an entity or entities. Provide a SHORT ANSWER.
8) Ensure the answers are in contradiction with each other: one derived from the Old Sentence and the other from the New Sentence.

Your response MUST follow this EXACT format:
{{Question: <your_question>}}
{{Old Answer: <answer_from_old_sentence>}}
{{New Answer: <answer_from_new_sentence>}}

Example format:
{{Question: What is the population of New York City?}}
{{Old Answer: 8.4 million}}
{{New Answer: 8.8 million}}

DO NOT include any additional text, explanations, or formatting. ONLY output the three lines in the exact format shown above."""

QA_CHECK_SYSTEM_TEMPLATES = """**QA Pair Evaluation Criteria:**

Your task is to evaluate the quality of a generated Question-Answer (QA) pair. For a QA pair to be considered acceptable:

1. **Answer Accuracy:** The generated answer must be accurate and consistent with the context provided. This means the answer should correctly reflect the information found in the context without introducing errors or irrelevant data.

2. **Question Clarity and Completeness:** The generated question must be clear and complete in terms of its subject matter. This involves:

   - **Specificity:** The question should precisely define the subject or object it refers to. Avoid vague terms or pronouns that necessitate additional context.
   
   - **Contextual Independence:** The question should stand alone without needing additional information beyond what is necessary to understand the subject or object, without requiring explicit mention of update or modification times.

3. **Avoidance of Explicit Date-Reference Queries:**
   - The question should not specifically ask for when the information was last updated or modified. Instead, it should focus on content-related queries. Questions reliant on the inherent temporal nature of certain information are acceptable.
   - The question should not contain phrases like 'as of [date]', 'in [year]', etc.

*Examples of unclear or incomplete questions for improvement:*

- "When were the last and next elections for representatives?" (Specify what "representatives" refers to, e.g., "city council representatives.")

- "What are the accrediting agencies of the school?" (Identify which school is being referenced.)

- "What matches were the following players called up for?" (Specify which players are being referred to, or include their names.)

*Evaluation Process:*

- For each QA pair, first verify the accuracy of the answer based on the context.
- Then, review the question to ensure it is clear and complete in its specificity and independence, excluding explicit requests regarding update times.
- Finally, ensure the QA pair adheres to avoiding direct queries about when the information was last modified or accessed.

If all criteria (Answer Accuracy, Question Clarity and Completeness, and Avoidance of Explicit Date-Reference Queries) are met, the QA pair is deemed acceptable; otherwise, it needs revision.

If the QA pair is acceptable, return "yes"; otherwise, return "no". Only return "yes" or "no".
"""

QA_CHECK_USER_TEMPLATES = """# Input:
- **Question**: {question}
- **Generated Answer**: {answer}
- **Context**: {context}

# Output:
"""

QA_TEMPLATE_WITH_CONTEXT = """Answer the question based on the provided sentence(s) and source content. You can reference the source content for broader context, but the question should be answerable directly based on the sentence(s).
Your answer should be a SHORT ANSWER. Please be concise and accurate.

{example}

# Question
{question}

# Sentence(s)
{sentence}

# Source Content
{source_content}

# Answer
"""

SAME_ANSWER_CHECK_TEMPLATE = """Your task is to identify if the two answers for a question are semantically the same.

# Question
Who is the current president of the United States?

# Answer 1
Joe Biden

# Answer 2
Donald Trump

# Response
No

# Question
How many champions does Lebron James have?

# Answer 1
4

# Answer 2
four

# Response
Yes

# Question
{question}

# Answer 1
{answer_old}

# Answer 2
{answer_new}

# Response
"""