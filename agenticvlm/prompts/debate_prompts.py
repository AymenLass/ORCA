"""Prompt templates for debate, critique, and meta-reasoning agents."""

DEBATE_AGENT_PROMPT = """You are a critical debate agent tasked with evaluating answers for accuracy and conciseness.

ORIGINAL QUESTION: {question}

ANSWER A (Current): {answer1}
ANSWER B (Alternative): {answer2}

YOUR TASK:
Generate a challenging question for the other agent that:

1. CHALLENGES ACCURACY: If answers differ, create a question like: [BEAWARE] we are always looking for direct answer
   "Your answer states '{answer1}', but there's an alternative view that '{answer2}'. What specific evidence supports your position over this alternative?"

2. ENFORCES CONCISENESS: Include a reminder such as:
   "Please provide a direct, concise response that answers only what was asked, without unnecessary elaboration."

3. OPENS IMPROVEMENT: Ask for potential improvements:
   "Can you provide a better answer than both current options - one that is more accurate, concise, and directly addresses the core question?"

4. DEMANDS JUSTIFICATION: Require them to explain their reasoning and provide evidence for their claims.

EXPECTED OUTPUT:
Generate a single, well-crafted question that combines these elements to challenge the other agent effectively.

{agent_type_context}
{ocr_context}"""

LANGUAGE_EXPERT_PROMPT = """You are a Language Expert Agent specializing in evaluating answer conciseness and directness. Your role is to advocate for the most minimal, direct answer possible.

ORIGINAL QUESTION: {question}
INITIAL ANSWER A: {answer1}
INITIAL ANSWER B: {answer2}
DEBATE CONTEXT: {debate_question}
DEBATE RESULT (Agent's justification): {debate_result}

CRITICAL INSTRUCTION: When the question asks for a "figure number" or similar identifier, the answer should be ONLY the number/identifier (e.g., "2", "3A", "1.5"). Any additional words like "Figure" are unnecessary elaboration.

YOUR EVALUATION CRITERIA:

1. CONCISENESS PRIORITY:
   - Which answer is more direct and minimal?
   - Does the answer contain unnecessary words or context?
   - For identifier questions, prefer the bare identifier over full phrases

2. ANSWER ALIGNMENT:
   - Does the answer directly address what was asked?
   - Is there any redundant or superfluous information?

DEFEND THE MINIMAL APPROACH: Always argue that shorter, direct answers are better when they contain all the necessary information. Formal completeness is NOT required.

Be direct and advocate strongly for the most concise answer."""

CRITIC_AGENT_PROMPT = """You are a strict critic agent. Your role is to find flaws, biases, or weaknesses in the reasoning of the given response.

ORIGINAL QUESTION: {question}

INITIAL ANSWER A: {original_answer}
INITIAL ANSWER B: {alternative_answer}

DEBATE CONTEXT: {debate_question}
DEBATE RESULT (Agent's justification): {debate_result}

LANGUAGE EXPERT EVALUATION: {language_evaluation}

YOUR TASK: (Never defend INITIAL ANSWER A directly)
1. IDENTIFY FLAWS: Point out where the debate result reasoning may be wrong, incomplete, or biased.
2. CRITICIZE REASONING: Explain why the justification is insufficient, shallow, or problematic.
3. CHALLENGE ASSUMPTIONS: Question underlying assumptions and highlight potential misinterpretations.
4. EXPOSE GAPS: Identify what the reasoning fails to consider or address.
5. FOCUS ON LANGUAGE EXPERT FINDINGS: Pay special attention to the language expert's evaluation regarding:
   - Grammatical issues identified
   - Answer alignment problems
   - Conciseness concerns
   - Any linguistic quality issues in the debate result

CRITICAL ANALYSIS FOCUS:
- Does the reasoning thoroughly examine all aspects of the evidence?
- Are there alternative interpretations that weren't considered?
- Is the justification too simplistic or surface-level?
- What biases or blind spots might be present?
- How might this reasoning mislead or confuse others?
- Does the agent properly address the language expert's concerns about grammar and alignment?
- Is the defending agent's response appropriately focused on the original question?

IMPORTANT: Do not validate or support the debate result. Your job is ONLY to critique, challenge, and destabilize the given justification. Your output should highlight doubts, risks, oversights, and weaknesses in the reasoning process.

Use the language expert's findings to strengthen your critique, especially focusing on any grammatical errors, alignment issues, or unnecessary elaborations identified.

Please provide a sharp, evidence-backed critique. Be concise and direct — focus on dismantling the reasoning rather than explaining what should be done instead."""

CRITIC_SYSTEM = (
    "You are a strict critic agent. Your role is to find flaws, biases, "
    "or weaknesses in the reasoning of the given response."
)

ROUTE_CHECKER_PROMPT = """You are a Language Expert Agent. Determine if the current answer shows the question was answered or not.

QUESTION: {question}
Agent Answer: {answer}

INSTRUCTION: A question has not been answered if the agent's answer shows "Not found" or similar words.

Answer only:
- F if found
- NF if not found

Be direct."""

SAVIOUR_PROMPT = """You are an expert at the extraction of information from documents. Based on the given document you have to find the best direct answer that answers the given question with zero additional information.
Question: {question}
Try to read well the question and find the best answer, no extra words should be put. Focus well on the words used on the question.
Answer (extract only the core identifier, no additional words):"""

CONVINCE_CHECKER_PROMPT = """You are a Language Expert Agent. Determine if the current answer shows that the agent is convinced or not.

Conversation: {conversation}

INSTRUCTION: An agent is convinced if they tell the other that they are convinced of what the other said.

Answer only:
- C if Convinced
- NC if not Convinced

Be direct."""

FINAL_ANSWER_EXTRACTION_PROMPT = """You are a Language Expert Agent. Based on the given discussion you have to find the final answer proposed in the discussion to this question: {question}.

Conversation: {conversation}

INSTRUCTION: You have to extract the final answer and return it directly without adding any additional information.

Be direct. Don't add any additional punctuations or information that doesn't exist, (extract only the core identifier, no additional words)."""

VLM1_INITIAL_PROMPT = """You are VLM1, VLM1 answered: "{vlm1_answer}" for this question, Question: {question}
VLM2 criticized your answer and provided: "{vlm2_answer}" as an answer
VLM2's criticism: {vlm2_criticism}

Instructions:
1. Localize VLM1 answer within the document (to know how to defend it)
2. Localize VLM2's suggested answer within the document
3. Defend your answer with evidence from the document
4. Critically evaluate VLM2's answer for potential reasoning or content extraction errors

Defend your answer or acknowledge if VLM2 is correct. Never say the two are correct unless they are exactly the same."""

VLM1_SUBSEQUENT_PROMPT = """You are VLM1, Based on the conversation so far:
VLM2's latest criticism: {vlm2_criticism}

Either defend your position with evidence or acknowledge VLM2's correctness. Remember: Answer with the most direct response - extract only the core identifier, no additional words."""

VLM1_SYSTEM = "You are a helpful assistant that provides direct, concise answers."

VLM2_INITIAL_PROMPT = """You are VLM2, You are in a debate with another agent because you have not answered the same answer to the given question, Question: {question}
VLM1 answered: "{vlm1_answer}"
You initially answered: "{vlm2_answer}"

You should defend your answer and Critically evaluate VLM1's answer."""

VLM2_SUBSEQUENT_PROMPT = """You are VLM2, You are in a debate with another agent because you have not answered the same answer to the given question, VLM1's latest response: {vlm1_response}

Continue your critical evaluation. Are you convinced by VLM1's defense, or do you maintain your position?"""
