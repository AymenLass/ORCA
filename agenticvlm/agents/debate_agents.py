"""Text-only debate, evaluation, and reasoning agents.

Agents backed by Qwen3 (text-only LLM):
- DebateAgent — generates challenging debate questions for stress testing
- LanguageExpertAgent — evaluates grammatical quality and conciseness
- RouteChecker — verifies if an answer was actually found
- ConvinceChecker — determines if a debating agent has been convinced
- EvaluationAgent — assesses pass/fail for stress testing turns (Stage 3)
- SanityChecker — corrects formatting consistency (Stage 5)
- JudgeAgent — generates summaries between multi-turn debate turns (Stage 4)
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

from agenticvlm.agents.base import BaseTextAgent
from agenticvlm.models.qwen3 import GenerationConfig, Qwen3Model

logger = logging.getLogger(__name__)


class DebateAgent(BaseTextAgent):
    """Generates challenging debate questions between two candidate answers."""

    def __init__(self, model: Qwen3Model) -> None:
        self.model = model

    @staticmethod
    def create_prompt(
        question: str,
        answer1: str,
        answer2: str,
        agent_type: str = "",
        ocr_extraction: str = "",
    ) -> str:
        prompt = f"""You are a critical debate agent tasked with evaluating answers for accuracy and conciseness.

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

{f"AGENT TYPE: {agent_type}" if agent_type else ""}
{f"OCR CONTEXT: {ocr_extraction}" if ocr_extraction else ""}"""
        return prompt

    def generate_debate_question(
        self,
        question: str,
        answer1: str,
        answer2: str,
        agent_type: str = "",
        ocr_extraction: str = "",
    ) -> Tuple[str, str]:
        """Generate a challenging debate question.

        Returns:
            Tuple of ``(thinking_content, debate_question)``.
        """
        prompt = self.create_prompt(question, answer1, answer2, agent_type, ocr_extraction)
        config = GenerationConfig(max_new_tokens=32768, temperature=1e-20)
        full_output = self.model.generate(
            image_path=None, prompt=prompt, gen_config=config
        )
        return "", full_output  # thinking is stripped by the model wrapper

    def generate(self, prompt: str, **kwargs: Any) -> str:
        config = GenerationConfig(max_new_tokens=32768, temperature=1e-20)
        return self.model.generate(image_path=None, prompt=prompt, gen_config=config)


class LanguageExpertAgent(BaseTextAgent):
    """Evaluates answer conciseness, grammatical correctness, and directness.
    Advocates for minimal, direct answers."""

    def __init__(self, model: Qwen3Model) -> None:
        self.model = model

    @staticmethod
    def create_prompt(
        question: str,
        answer1: str,
        answer2: str,
        debate_question: str,
        debate_result: str,
    ) -> str:
        prompt = f"""You are a Language Expert Agent specializing in evaluating answer conciseness and directness. Your role is to advocate for the most minimal, direct answer possible.

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
        return prompt

    def evaluate_language_quality(
        self,
        question: str,
        answer1: str,
        answer2: str,
        debate_question: str,
        debate_result: str,
    ) -> Tuple[str, str]:
        """Evaluate language quality and conciseness.

        Returns:
            Tuple of ``(thinking_content, language_evaluation)``.
        """
        prompt = self.create_prompt(question, answer1, answer2, debate_question, debate_result)
        config = GenerationConfig(max_new_tokens=32768, temperature=1e-20)
        full_output = self.model.generate(
            image_path=None, prompt=prompt, gen_config=config
        )
        return "", full_output

    def generate(self, prompt: str, **kwargs: Any) -> str:
        config = GenerationConfig(max_new_tokens=32768, temperature=1e-20)
        return self.model.generate(image_path=None, prompt=prompt, gen_config=config)


class RouteChecker(BaseTextAgent):
    """Checks whether a specialist agent actually found an answer or returned
    a "Not found" style response."""

    def __init__(self, model: Qwen3Model) -> None:
        self.model = model

    @staticmethod
    def create_prompt(question: str, answer: str) -> str:
        prompt = f"""You are a Language Expert Agent. Determine if the current answer shows the question was answered or not.

QUESTION: {question}
Agent Answer: {answer}

INSTRUCTION: A question has not been answered if the agent's answer shows "Not found" or similar words.

Answer only:
- F if found
- NF if not found

Be direct."""
        return prompt

    def check_answer(self, question: str, answer: str) -> str:
        """Check if the answer was found.

        Returns:
            ``'F'`` if found, ``'NF'`` if not found.
        """
        prompt = self.create_prompt(question, answer)
        config = GenerationConfig(max_new_tokens=32768, temperature=1e-20)
        return self.model.generate(image_path=None, prompt=prompt, gen_config=config)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        config = GenerationConfig(max_new_tokens=32768, temperature=1e-20)
        return self.model.generate(image_path=None, prompt=prompt, gen_config=config)


class ConvinceChecker(BaseTextAgent):
    """Determines if a debating agent has been convinced and extracts the
    final answer from a multi-turn conversation."""

    def __init__(self, model: Qwen3Model) -> None:
        self.model = model

    @staticmethod
    def create_convince_prompt(conversation: str, answer: str) -> str:
        prompt = f"""You are a Language Expert Agent. Determine if the current answer shows that the agent is convinced or not.

Conversation: {conversation}
Previous answer: {answer}

INSTRUCTION: An agent is convinced if they tell the other that they are convinced of what the other said.

Answer only:
- C if Convinced
- NC if not Convinced

Be direct."""
        return prompt

    @staticmethod
    def create_final_answer_prompt(conversation: str, question: str) -> str:
        prompt = f"""You are a Language Expert Agent. Based on the given discussion you have to find the final answer proposed in the discussion to this question : {question}.

Conversation: {conversation}

INSTRUCTION: You have to extract the final answer and return it directly without adding any additional information.

Be direct. Don't add any additional punctuations or information that doesn't exist , (extract only the core identifier, no additional words). """
        return prompt

    @staticmethod
    def create_linguistic_confidence_prompt(conversation: str) -> str:
        """Prompt for linguistic confidence analysis after 3 turns."""
        return f"""You are a Linguistic Confidence Analyzer. Analyze the following debate conversation for signs of hedging, reduced confidence, or implicit concession.

Conversation: {conversation}

INDICATORS OF REDUCED CONFIDENCE:
- Hedging language: "perhaps", "maybe", "I think", "it's possible"
- Qualification: "but I could be wrong", "in some cases"
- Weakened stance: Previously strong claims becoming tentative
- Partial agreement: "You make a good point, but..."
- Evasion: Not directly addressing challenges

For EACH agent (VLM1, VLM2), assess their confidence level:
- HIGH: Maintains strong, evidence-based positions
- MEDIUM: Shows some hedging but still defends position
- LOW: Significant hedging or implicit concession

Output format:
VLM1_CONFIDENCE: [HIGH/MEDIUM/LOW]
VLM2_CONFIDENCE: [HIGH/MEDIUM/LOW]
LIKELY_WINNER: [VLM1/VLM2/UNDECIDED]"""

    def check_conversation(self, conversation: str, answer: str) -> str:
        """Check if an agent has been convinced.

        Returns:
            ``'C'`` if convinced, ``'NC'`` if not convinced.
        """
        prompt = self.create_convince_prompt(conversation, answer)
        config = GenerationConfig(max_new_tokens=32768, temperature=1e-30)
        return self.model.generate(image_path=None, prompt=prompt, gen_config=config)

    def get_final_answer(self, conversation: str, question: str) -> str:
        """Extract the final answer from a debate conversation."""
        prompt = self.create_final_answer_prompt(conversation, question)
        config = GenerationConfig(max_new_tokens=32768, temperature=1e-20)
        return self.model.generate(image_path=None, prompt=prompt, gen_config=config)

    def analyze_linguistic_confidence(self, conversation: str) -> str:
        """Perform linguistic confidence analysis after 3+ turns.

        Returns:
            Structured analysis with confidence levels and likely winner.
        """
        prompt = self.create_linguistic_confidence_prompt(conversation)
        config = GenerationConfig(max_new_tokens=512, temperature=1e-30)
        return self.model.generate(image_path=None, prompt=prompt, gen_config=config)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        config = GenerationConfig(max_new_tokens=32768, temperature=1e-20)
        return self.model.generate(image_path=None, prompt=prompt, gen_config=config)


class EvaluationAgent(BaseTextAgent):
    """Assesses pass/fail for each stress-testing turn."""

    def __init__(self, model: Qwen3Model) -> None:
        self.model = model

    @staticmethod
    def create_prompt(
        question: str,
        debate_challenge: str,
        specialist_response: str,
        specialist_answer: str,
        turn: int = 1,
    ) -> str:
        return f"""You are a strict Evaluation Agent. You must assess whether the specialist agent adequately defended its answer against the debate challenge.

ORIGINAL QUESTION: {question}
SPECIALIST'S CURRENT ANSWER: {specialist_answer}

DEBATE CHALLENGE (Turn {turn}): {debate_challenge}
SPECIALIST'S RESPONSE: {specialist_response}

EVALUATION CRITERIA:
1. Did the specialist provide evidence-based reasoning?
2. Did the specialist address the specific challenge raised?
3. Is the specialist's answer consistent with the evidence provided?
4. Did the specialist avoid contradictions or logical fallacies?

INSTRUCTIONS:
- If the specialist adequately defended its answer with evidence, output: PASS
- If the specialist's defense was inadequate, contradictory, or unsupported, output: FAIL

Output ONLY one word: PASS or FAIL"""

    def evaluate_turn(
        self,
        question: str,
        debate_challenge: str,
        specialist_response: str,
        specialist_answer: str,
        turn: int = 1,
    ) -> str:
        """Evaluate a single stress-testing turn.

        Returns:
            ``'PASS'`` or ``'FAIL'``.
        """
        prompt = self.create_prompt(
            question, debate_challenge, specialist_response, specialist_answer, turn
        )
        config = GenerationConfig(max_new_tokens=32768, temperature=1e-30)
        output = self.model.generate(image_path=None, prompt=prompt, gen_config=config)
        result = output.strip().upper()
        if "PASS" in result:
            return "PASS"
        return "FAIL"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        config = GenerationConfig(max_new_tokens=32768, temperature=1e-30)
        return self.model.generate(image_path=None, prompt=prompt, gen_config=config)


class SanityChecker(BaseTextAgent):
    """Corrects formatting inconsistencies in the final answer."""

    def __init__(self, model: Qwen3Model) -> None:
        self.model = model

    @staticmethod
    def create_prompt(question: str, answer: str, context: str = "") -> str:
        return f"""You are a Sanity Checker Agent. Your ONLY job is to correct the formatting of the final answer to ensure it matches the original document's style.

QUESTION: {question}
PROPOSED ANSWER: {answer}
{f"DOCUMENT CONTEXT: {context}" if context else ""}

INSTRUCTIONS:
1. Check if the answer has correct spacing (no extra or missing spaces)
2. Check if punctuation matches what appears in the original document
3. Check if capitalization is consistent with the document
4. Do NOT change the semantic content of the answer
5. Do NOT add explanations or extra information
6. If the answer is already correctly formatted, return it unchanged

Return ONLY the corrected answer, nothing else:"""

    def refine_answer(
        self,
        question: str,
        answer: str,
        context: str = "",
    ) -> str:
        """Refine the formatting of a final answer.

        Args:
            question: The original question.
            answer: The proposed final answer.
            context: Optional document context for formatting reference.

        Returns:
            The formatting-corrected answer.
        """
        prompt = self.create_prompt(question, answer, context)
        config = GenerationConfig(max_new_tokens=256, temperature=1e-30)
        output = self.model.generate(image_path=None, prompt=prompt, gen_config=config)
        refined = output.strip()
        if len(refined) > len(answer) * 3 + 20:
            return answer
        return refined if refined else answer

    def generate(self, prompt: str, **kwargs: Any) -> str:
        config = GenerationConfig(max_new_tokens=256, temperature=1e-30)
        return self.model.generate(image_path=None, prompt=prompt, gen_config=config)


class JudgeAgent(BaseTextAgent):
    """Generates structured summaries between multi-turn debate rounds."""

    def __init__(self, model: Qwen3Model) -> None:
        self.model = model

    @staticmethod
    def create_summary_prompt(
        question: str,
        thesis_answer: str,
        antithesis_answer: str,
        current_turn_text: str,
        turn_number: int,
    ) -> str:
        return f"""You are a Judge Agent moderating a structured debate. Generate a summary of Turn {turn_number}.

QUESTION: {question}
THESIS (Specialist) ANSWER: {thesis_answer}
ANTITHESIS (Vision Expert) ANSWER: {antithesis_answer}

TURN {turn_number} EXCHANGE:
{current_turn_text}

Generate a structured summary with these EXACT sections:

[REFERENCE]
Summarize the key evidence and arguments presented in this turn.

[CRITICISM]
Identify the main points of contention and unresolved disagreements.

[CONCLUSION]
State which answer is currently better supported and why.

Be concise and neutral. Focus on evidence quality, not argument sophistication."""

    def generate_turn_summary(
        self,
        question: str,
        thesis_answer: str,
        antithesis_answer: str,
        turn_text: str,
        turn_number: int,
    ) -> str:
        """Generate a structured summary for a debate turn.

        Returns:
            Structured text with [REFERENCE], [CRITICISM], [CONCLUSION].
        """
        prompt = self.create_summary_prompt(
            question, thesis_answer, antithesis_answer, turn_text, turn_number
        )
        config = GenerationConfig(max_new_tokens=1024, temperature=1e-30)
        return self.model.generate(image_path=None, prompt=prompt, gen_config=config)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        config = GenerationConfig(max_new_tokens=1024, temperature=1e-30)
        return self.model.generate(image_path=None, prompt=prompt, gen_config=config)