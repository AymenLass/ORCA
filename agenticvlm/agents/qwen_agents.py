"""Qwen2.5-VL-backed specialist agents.

These agents use the Qwen2.5-VL (7B) or Qwen2-VL-OCR (2B) models with
``process_vision_info`` and path-based image format for document analysis.

Agents:
- FormAgent — forms, applications, structured input documents
- FreeTextAgent — unstructured text, paragraphs, continuous prose
- HandwrittenAgent — handwritten text and low-quality OCR
- YesNoAgent — binary yes/no decisions
- LayoutAgent — document structure and spatial organization
- TableListAgent — tabular data and structured lists
- CriticAgent — critique and challenge reasoning
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agenticvlm.agents.base import BaseSpecialistAgent
from agenticvlm.models.base import GenerationConfig
from agenticvlm.models.qwen25vl import Qwen25VLModel
from agenticvlm.models.qwen2_ocr import Qwen2OCRModel

logger = logging.getLogger(__name__)

_SPECIALIST_GEN = GenerationConfig(max_new_tokens=128, temperature=1e-20)
# Critic gets slightly more tokens for detailed critique
_CRITIC_GEN = GenerationConfig(max_new_tokens=256, temperature=1e-20)


class _QwenSpecialistMixin:
    """Shared Qwen2.5-VL generation logic for specialist agents."""

    model: Qwen25VLModel | Qwen2OCRModel
    SYSTEM_MESSAGE: str

    def _generate_qwen(
        self,
        image_path: str,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate a response using the Qwen model with system message."""
        try:
            config = gen_config or _SPECIALIST_GEN
            return self.model.generate(
                image_path=image_path,
                prompt=prompt,
                gen_config=config,
                system_message=self.SYSTEM_MESSAGE,
            )
        except Exception as e:
            logger.error("Qwen specialist generation error: %s", e)
            return ""


def _make_initial_prompt(role_description: str, question: str, debate_question: str) -> str:
    return f"""{role_description}

PRIMARY QUESTION: {question}
DEBATE CONTEXT: {debate_question}

INSTRUCTIONS:
1. First, carefully examine the document to answer the primary question
2. Consider the debate context - there may be competing interpretations or answers
3. Provide evidence-based reasoning for your answer
4. If the debate presents alternative viewpoints, address why your answer is more accurate
5. Be concise but thorough in your analysis
6. Focus only on what you can directly observe in the document

Your response should be structured as:
ANALYSIS: [Your detailed observation of the image]
ANSWER: [Your direct answer to the primary question]
JUSTIFICATION: [Why this answer is correct, addressing any debate points if relevant]

Begin your analysis:"""


def _make_final_prompt(
    role_description: str,
    question: str,
    debate_question: str,
    original_answer: str,
    alternative_answer: str,
    debate_result: str,
    critique: str,
    suffix: str = "",
) -> str:
    extra = ""
    if suffix:
        extra = f" {suffix}"
    return f"""{role_description}

ORIGINAL QUESTION: {question}
Your Answer: {original_answer}
Alternative Answer: {alternative_answer}

DEBATE CONTEXT: {debate_question}
Your Justification: {debate_result}
Critique: {critique}

INSTRUCTIONS:
Based on the entire conversation, if you are very confident, defend your result with stronger, high-level arguments. If the critiques are valid, update your answer accordingly and provide your final, confident answer.{extra}

Your Final Answer (extract only the core identifier, no additional words):"""


class FormAgent(_QwenSpecialistMixin, BaseSpecialistAgent):
    """Specialist for forms, applications, and structured input documents."""

    SYSTEM_MESSAGE = (
        "You are a specialized FORM PROCESSING agent. Your role is to extract and "
        "analyze information from forms, applications, and structured input documents."
    )

    def __init__(self, model: Qwen25VLModel) -> None:
        self.model = model

    def analyze_image(self, image_path: str, question: str, debate_question: str) -> str:
        prompt = _make_initial_prompt(self.SYSTEM_MESSAGE, question, debate_question)
        return self._generate_qwen(image_path, prompt)

    def final_analysis(self, image_path, question, debate_question, original_answer,
                       alternative_answer, debate_result, critique) -> str:
        prompt = _make_final_prompt(
            self.SYSTEM_MESSAGE, question, debate_question,
            original_answer, alternative_answer, debate_result, critique,
        )
        return self._generate_qwen(image_path, prompt)


class FreeTextAgent(_QwenSpecialistMixin, BaseSpecialistAgent):
    """Specialist for unstructured running text, paragraphs, and continuous prose."""

    SYSTEM_MESSAGE = (
        "You are a specialized free text reading agent. Your task is to extract "
        "precise information from unstructured running text, paragraphs, and "
        "continuous prose in the document image."
    )

    def __init__(self, model: Qwen25VLModel) -> None:
        self.model = model

    def analyze_image(self, image_path: str, question: str, debate_question: str) -> str:
        prompt = _make_initial_prompt(self.SYSTEM_MESSAGE, question, debate_question)
        return self._generate_qwen(image_path, prompt)

    def final_analysis(self, image_path, question, debate_question, original_answer,
                       alternative_answer, debate_result, critique) -> str:
        prompt = _make_final_prompt(
            self.SYSTEM_MESSAGE, question, debate_question,
            original_answer, alternative_answer, debate_result, critique,
        )
        return self._generate_qwen(image_path, prompt)


class HandwrittenAgent(_QwenSpecialistMixin, BaseSpecialistAgent):
    """Specialist for handwritten text and low-quality document OCR.

    Uses the lighter Qwen2-VL-OCR-2B model for better handwriting recognition.
    """

    SYSTEM_MESSAGE = (
        "You are a specialized OCR and TEXT EXTRACTION agent. Your ONLY role is "
        "to extract ALL text content from the document image due to its low "
        "quality or being handwritten."
    )

    def __init__(self, model: Qwen2OCRModel) -> None:
        self.model = model  # type: ignore[assignment]

    def analyze_image(self, image_path: str, question: str, debate_question: str) -> str:
        prompt = _make_initial_prompt(self.SYSTEM_MESSAGE, question, debate_question)
        return self._generate_qwen(image_path, prompt)

    def final_analysis(self, image_path, question, debate_question, original_answer,
                       alternative_answer, debate_result, critique) -> str:
        prompt = _make_final_prompt(
            self.SYSTEM_MESSAGE, question, debate_question,
            original_answer, alternative_answer, debate_result, critique,
        )
        return self._generate_qwen(image_path, prompt)


class YesNoAgent(_QwenSpecialistMixin, BaseSpecialistAgent):
    """Specialist for binary yes/no decision questions."""

    SYSTEM_MESSAGE = (
        "You are a specialized binary decision agent. Your role is to determine if "
        "questions can be answered with a simple YES or NO, and provide that binary "
        "response when appropriate."
    )

    def __init__(self, model: Qwen25VLModel) -> None:
        self.model = model

    def analyze_image(self, image_path: str, question: str, debate_question: str) -> str:
        prompt = _make_initial_prompt(self.SYSTEM_MESSAGE, question, debate_question)
        return self._generate_qwen(image_path, prompt)

    def final_analysis(self, image_path, question, debate_question, original_answer,
                       alternative_answer, debate_result, critique) -> str:
        prompt = _make_final_prompt(
            self.SYSTEM_MESSAGE, question, debate_question,
            original_answer, alternative_answer, debate_result, critique,
        )
        return self._generate_qwen(image_path, prompt)


class LayoutAgent(_QwenSpecialistMixin, BaseSpecialistAgent):
    """Specialist for document structure, positioning, and spatial organization."""

    SYSTEM_MESSAGE = (
        "You are a specialized DOCUMENT LAYOUT analysis agent. Your role is to "
        "analyze document structure, positioning, and spatial organization."
    )

    def __init__(self, model: Qwen25VLModel) -> None:
        self.model = model

    def analyze_image(self, image_path: str, question: str, debate_question: str) -> str:
        prompt = _make_initial_prompt(self.SYSTEM_MESSAGE, question, debate_question)
        return self._generate_qwen(image_path, prompt)

    def final_analysis(self, image_path, question, debate_question, original_answer,
                       alternative_answer, debate_result, critique) -> str:
        prompt = _make_final_prompt(
            self.SYSTEM_MESSAGE, question, debate_question,
            original_answer, alternative_answer, debate_result, critique,
        )
        return self._generate_qwen(image_path, prompt)


class TableListAgent(_QwenSpecialistMixin, BaseSpecialistAgent):
    """Specialist for tabular data and structured lists.

    The final analysis prompt includes an extra instruction asking
    the agent to explicitly choose between the two candidate answers.
    """

    SYSTEM_MESSAGE = (
        "You are a specialized TABLE AND LIST extraction agent. Your ONLY role "
        "is to extract and answer questions from tabular data and structured "
        "lists in document images."
    )

    def __init__(self, model: Qwen25VLModel) -> None:
        self.model = model

    def analyze_image(self, image_path: str, question: str, debate_question: str) -> str:
        prompt = _make_initial_prompt(self.SYSTEM_MESSAGE, question, debate_question)
        return self._generate_qwen(image_path, prompt)

    def final_analysis(self, image_path, question, debate_question, original_answer,
                       alternative_answer, debate_result, critique) -> str:
        suffix = (
            f'You have at the end to choose "{original_answer}" or '
            f'"{alternative_answer}" or shorten or add important element to one of them.'
        )
        prompt = _make_final_prompt(
            self.SYSTEM_MESSAGE, question, debate_question,
            original_answer, alternative_answer, debate_result, critique,
            suffix=suffix,
        )
        return self._generate_qwen(image_path, prompt)


class CriticAgent:
    """Finds flaws, biases, and weaknesses in specialist agent responses."""

    SYSTEM_MESSAGE = (
        "You are a strict critic agent. Your role is to find flaws, biases, "
        "or weaknesses in the reasoning of the given response."
    )

    def __init__(self, model: Qwen25VLModel) -> None:
        self.model = model

    @staticmethod
    def create_prompt(
        question: str,
        debate_question: str,
        debate_result: str,
        original_answer: str,
        alternative_answer: str,
        language_evaluation: str,
        agent_type: str = "",
        ocr_extraction: str = "",
    ) -> str:
        return f"""You are a strict critic agent. Your role is to find flaws, biases, or weaknesses in the reasoning of the given response.

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

    def critique_answer(
        self,
        image_path: str,
        question: str,
        debate_question: str,
        debate_result: str,
        original_answer: str,
        alternative_answer: str,
        language_evaluation: str,
        agent_type: str = "",
        ocr_extraction: str = "",
    ) -> str:
        """Produce a critique of the specialist agent's reasoning.

        Args:
            image_path: Path to the document image.
            question: Original user question.
            debate_question: The debate agent's challenge.
            debate_result: The specialist's initial analysis.
            original_answer: Answer A from stage 1.
            alternative_answer: Answer B from stage 1.
            language_evaluation: The language expert's evaluation.
            agent_type: Optional agent type hint.
            ocr_extraction: Optional OCR context.

        Returns:
            A text critique.
        """
        prompt = self.create_prompt(
            question, debate_question, debate_result,
            original_answer, alternative_answer, language_evaluation,
            agent_type, ocr_extraction,
        )
        try:
            return self.model.generate(
                image_path=image_path,
                prompt=prompt,
                gen_config=_CRITIC_GEN,
                system_message=self.SYSTEM_MESSAGE,
            )
        except Exception as e:
            logger.error("Critic agent error: %s", e)
            return ""
