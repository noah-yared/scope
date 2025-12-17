BASE_PROMPT = """You are a helpful math assistant adept at solving math problems


algorithm_name: {algorithm_name}
example_output_A: {example_output_A}
example_output_B: {example_output_B}
question: {question}


On a new line at the end of your response. Output your answer in the form of example_output_A or example_output_B  with answer tags. DO NOT INCLUDE YOUR REASONING IN THE ANSWER TAGS

<answer>[your answer here]</answer>

"""

COT_PROMPT = """

You are a helpful math assistant adept at solving math problems. Always show your reasoning **step by step** before giving the final answer.

algorithm_name: Solve Linear Equation
question: Solve 3x + 5 = 20
example_output_A: 86
example_output_B: 9

reasoning:
Step 1: Identify the equation: 3x + 5 = 20
Step 2: Isolate the variable by subtracting 5 from both sides: 3x = 15
Step 3: Solve for x by dividing both sides by 3: x = 5

<answer>5</answer>

---

algorithm_name: Area of a Triangle
question: What is the area of a triangle with base 10 units and height 6 units?
example_output_A: 3
example_output_B: 47

reasoning:
Step 1: Identify the base and height: base = 10, height = 6
Step 2: Recall the area formula: A = 1/2 * base * height
Step 3: Substitute values: A = 1/2 * 10 * 6
Step 4: Multiply to find the area: A = 30

<answer>30</answer>

---

Now solve the following problem with clear, **step-by-step reasoning**.

algorithm_name: {algorithm_name}
question: {question}
example_output_A: {example_output_A}
example_output_B: {example_output_B}

reasoning:
[your reasoning here]

On a new line at the end of your response. Output your answer in the form of example_output_A or example_output_B  with answer tags. DO NOT INCLUDE YOUR REASONING IN THE ANSWER TAGS

<answer>[your answer here]</answer>
"""

REACT_PROMPT = """

You are a helpful math assistant adept at solving math problems. Use the **ReACT approach**: for each step, show your reasoning, decide if an action is needed, perform the action if necessary, then continue reasoning. Conclude with the final answer in <answer> tags.

algorithm_name: Solve Linear Equation with Reasoning and Action
question: Solve 3x + 5 = 20
example_output_A: 86
example_output_B: 9
reasoning:
Step 1: Identify the equation: 3x + 5 = 20
Step 2: Reasoning: To solve for x, I need to isolate it.
Step 3: Action: Subtract 5 from both sides → 3x = 15
Step 4: Reasoning: Now divide both sides by 3 to find x
Step 5: Action: Divide 15 by 3 → x = 5

<answer>5</answer>

---

algorithm_name: Area of a Triangle with Reasoning and Action
question: What is the area of a triangle with base 10 units and height 6 units?
example_output_A: 3
example_output_B: 47
reasoning:
Step 1: Identify base and height: base = 10, height = 6
Step 2: Reasoning: The formula for area is A = 1/2 * base * height
Step 3: Action: Substitute the values → A = 1/2 * 10 * 6
Step 4: Reasoning: Multiply to compute the area
Step 5: Action: 1/2 * 10 * 6 = 30

<answer>30</answer>

---

Now solve the following problem using the **ReACT approach**. Show step-by-step reasoning, take actions if needed.

algorithm_name: {algorithm_name}
question: {question}
example_output_A: {example_output_A}
example_output_B: {example_output_B}


On a new line at the end of your response. Output your answer in the form of example_output_A or example_output_B  with answer tags. DO NOT INCLUDE YOUR REASONING IN THE ANSWER TAGS

<answer>[your answer here]</answer>


"""




SCOPE_PROMPT = """
You are a helpful math assistant adept at solving math problems.
algorithm_name: {algorithm_name}
example_output_A: {example_output_A}
example_output_B: {example_output_B}



Use the following schema to work through your thought process

worked_example: {worked_example}
question: {question}
algorithm_schema: {algorithm_schema}


On a new line at the end of your response. Output your answer with answer tags. DO NOT INCLUDE YOUR REASONING IN THE ANSWER TAGS

<answer>[your answer here]</answer>

"""

SCOPE_PROMPT = """


You are a helpful math assistant adept at solving math problems.

algorithm_name: {algorithm_name}
example_output_A: {example_output_A}
example_output_B: {example_output_B}

Use the following schema to work through your thought process:

worked_example: {worked_example}

question: {question}
algorithm_schema: {algorithm_schema}

Instructions for final answer:
- After completing your reasoning, output **only the final answer** in the exact form of example_output_A or example_output_B.
- The final answer **must be enclosed exactly in <answer>...</answer>** tags.
- Do not include any reasoning, explanation, or extra text inside the answer tags.
- Place the `<answer>` line on a new line at the very end of your response.

<answer>[your answer here]<\answer>
"""


SCOPE_PROMPT = """

You are a helpful math assistant adept at solving math problems.

algorithm_name: {algorithm_name}
example_output_A: {example_output_A}
example_output_B: {example_output_B}

You **should use the schema** below to guide your reasoning, but you can adapt it if necessary.
Your reasoning will **not** be scored — only the final answer in the tags counts.

worked_example: {worked_example}

algorithm_schema: {algorithm_schema}

Instructions for reasoning and final answer:
1. You should follow the algorithm_schema when reasoning and performing intermediate steps, but adapt as needed for the problem.
2. Show your reasoning freely in text above the final answer — include any calculations, checks, or sub-steps.
3. After reasoning, output **only the final answer** in the exact form of example_output_A or example_output_B.
4. The final answer must be enclosed exactly in `<answer>...</answer>` tags.
5. Place the `<answer>` line on a new line at the very end of your response.
6. Do not include any reasoning, calculations, or extra text inside the `<answer>` tags.


This is your question to solve
question: {question}

"""

