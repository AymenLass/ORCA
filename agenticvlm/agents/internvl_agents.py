"""InternVL3-backed specialist agents.

These agents use the InternVL3 vision-language model with URL-based
image format for document analysis.

- FigureDiagramAgent — technical figures, charts, plots, scientific illustrations
- ImagePhotoAgent — photographic content, real-world scenes, visual elements
- Saviour — fallback agent when the primary pipeline returns "Not Found"
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from agenticvlm.agents.base import BaseSpecialistAgent
from agenticvlm.models.internvl3 import InternVL3Model

logger = logging.getLogger(__name__)


class _InternVLSpecialistMixin:
    """Shared generation logic for InternVL3-based specialists."""

    model: InternVL3Model

    def _generate_internvl(self, image_path: str, prompt: str) -> str:
        """Generate a response using the InternVL3 model."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self.model.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.model.device, dtype=torch.bfloat16)

            generate_ids = self.model.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=1e-20,
            )
            decoded = self.model.processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            return decoded.strip() if decoded else ""
        except Exception as e:
            logger.error("InternVL generation error: %s", e)
            return ""


class FigureDiagramAgent(_InternVLSpecialistMixin, BaseSpecialistAgent):
    """Specialist for technical figures, diagrams, charts, plots, and
    scientific illustrations."""

    SYSTEM_MESSAGE = (
        "You are a specialized FIGURE AND DIAGRAM analysis agent. "
        "Your role is to analyze technical figures, diagrams, charts, plots, "
        "and scientific illustrations with precision and accuracy."
    )

    def __init__(self, model: InternVL3Model) -> None:
        self.model = model

    @staticmethod
    def _initial_prompt(question: str, debate_question: str) -> str:
        return f"""You are a specialized FIGURE AND DIAGRAM analysis agent. Your role is to analyze technical figures, diagrams, charts, plots, and scientific illustrations with precision and accuracy.

PRIMARY QUESTION: {question}
DEBATE CONTEXT: {debate_question}

INSTRUCTIONS:
1. First, carefully examine the image to answer the primary question
2. Consider the debate context - there may be competing interpretations or answers
3. Provide evidence-based reasoning for your answer
4. If the debate presents alternative viewpoints, address why your answer is more accurate
5. Be concise but thorough in your analysis
6. Focus only on what you can directly observe in the image

Your response should be structured as:
ANALYSIS: [Your detailed observation of the image]
ANSWER: [Your direct answer to the primary question]
JUSTIFICATION: [Why this answer is correct, addressing any debate points if relevant]

Begin your analysis:"""

    @staticmethod
    def _final_prompt(
        question: str,
        debate_question: str,
        original_answer: str,
        alternative_answer: str,
        debate_result: str,
        critique: str,
    ) -> str:
        return f"""You are a specialized FIGURE AND DIAGRAM analysis agent. Your role is to analyze technical figures, diagrams, charts, plots, and scientific illustrations with precision and accuracy.

ORIGINAL QUESTION: {question}
Your Answer: {original_answer}
Alternative Answer: {alternative_answer}

DEBATE CONTEXT: {debate_question}
Your Justification: {debate_result}
Critique: {critique}

INSTRUCTIONS:
Based on the entire conversation, if you are very confident, defend your result with stronger, high-level arguments. If the critiques are valid, update your answer accordingly and provide your final, confident answer.

Your Final Answer (extract only the core identifier, no additional words):"""

    def analyze_image(self, image_path: str, question: str, debate_question: str) -> str:
        prompt = self._initial_prompt(question, debate_question)
        return self._generate_internvl(image_path, prompt)

    def final_analysis(
        self,
        image_path: str,
        question: str,
        debate_question: str,
        original_answer: str,
        alternative_answer: str,
        debate_result: str,
        critique: str,
    ) -> str:
        prompt = self._final_prompt(
            question, debate_question, original_answer,
            alternative_answer, debate_result, critique,
        )
        return self._generate_internvl(image_path, prompt)


class ImagePhotoAgent(_InternVLSpecialistMixin, BaseSpecialistAgent):
    """Specialist for photographic content, real-world scenes, objects,
    and visual elements."""

    SYSTEM_MESSAGE = (
        "You are a specialized IMAGE AND PHOTO analysis agent. "
        "Your role is to analyze photographic content, real-world scenes, "
        "objects, and visual elements."
    )

    def __init__(self, model: InternVL3Model) -> None:
        self.model = model

    @staticmethod
    def _initial_prompt(question: str, debate_question: str) -> str:
        return f"""You are a specialized IMAGE AND PHOTO analysis agent. Your role is to analyze photographic content, real-world scenes, objects, and visual elements.

PRIMARY QUESTION: {question}
DEBATE CONTEXT: {debate_question}

INSTRUCTIONS:
1. First, carefully examine the image to answer the primary question
2. Consider the debate context - there may be competing interpretations or answers
3. Provide evidence-based reasoning for your answer
4. If the debate presents alternative viewpoints, address why your answer is more accurate
5. Be concise but thorough in your analysis
6. Focus only on what you can directly observe in the image

Your response should be structured as:
ANALYSIS: [Your detailed observation of the image]
ANSWER: [Your direct answer to the primary question]
JUSTIFICATION: [Why this answer is correct, addressing any debate points if relevant]

Begin your analysis:"""

    @staticmethod
    def _final_prompt(
        question: str,
        debate_question: str,
        original_answer: str,
        alternative_answer: str,
        debate_result: str,
        critique: str,
    ) -> str:
        return f"""You are a specialized IMAGE AND PHOTO analysis agent. Your role is to analyze photographic content, real-world scenes, objects, and visual elements.

ORIGINAL QUESTION: {question}
Your Answer: {original_answer}
Alternative Answer: {alternative_answer}

DEBATE CONTEXT: {debate_question}
Your Justification: {debate_result}
Critique: {critique}

INSTRUCTIONS:
Based on the entire conversation, if you are very confident, defend your result with stronger, high-level arguments. If the critiques are valid, update your answer accordingly and provide your final, confident answer.

Your Final Answer (extract only the core identifier, no additional words):"""

    def analyze_image(self, image_path: str, question: str, debate_question: str) -> str:
        prompt = self._initial_prompt(question, debate_question)
        return self._generate_internvl(image_path, prompt)

    def final_analysis(
        self,
        image_path: str,
        question: str,
        debate_question: str,
        original_answer: str,
        alternative_answer: str,
        debate_result: str,
        critique: str,
    ) -> str:
        prompt = self._final_prompt(
            question, debate_question, original_answer,
            alternative_answer, debate_result, critique,
        )
        return self._generate_internvl(image_path, prompt)


class Saviour(_InternVLSpecialistMixin):
    """Fallback agent used when the primary pipeline returns "Not Found".

    Generates a direct answer using InternVL3 without debate context.
    Also used as the second opinion (VLM2 answer) for the multi-turn
    conversation when answers disagree.
    """

    def __init__(self, model: InternVL3Model) -> None:
        self.model = model

    @staticmethod
    def _build_prompt(question: str) -> str:
        return f"""You are an expert at the extraction of information from documents. Based on the given document you have to find the best direct answer that answers the given question with zero additional information.
         Question : {question}
         try to read well the question and find the best answer, no extra words should be put. Focus well on the words used on the question.
         Answer(extract only the core identifier, no additional words) : """

    def propose_answer(self, question: str, image_path: str) -> str:
        """Generate a direct answer as a fallback.

        Args:
            question: The user question.
            image_path: Path to the document image.

        Returns:
            A concise, direct answer.
        """
        prompt = self._build_prompt(question)
        return self._generate_internvl(image_path, prompt)
