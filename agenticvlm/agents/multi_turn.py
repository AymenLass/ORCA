"""Stage 4 — Multi-turn structured VLM debate for consensus resolution."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from agenticvlm.agents.debate_agents import ConvinceChecker, JudgeAgent
from agenticvlm.models.base import GenerationConfig
from agenticvlm.models.internvl3 import InternVL3Model
from agenticvlm.models.qwen25vl import Qwen25VLModel

logger = logging.getLogger(__name__)


class MultiTurnConversation:
    """Stage 4 multi-turn structured debate between two VLMs.


    Args:
        vlm1_model: Qwen2.5-VL / Qwen3VL model (thesis / defender).
        vlm2_model: InternVL3 model (antithesis / challenger).
        convince_checker: ConvinceChecker instance (Qwen3-1.7B).
        judge_agent: JudgeAgent instance (Qwen3-1.7B).
        vlm1_answer: VLM1's answer from Stage 2.
        vlm2_answer: VLM2's alternative answer.
        question: The user question.
        image_path: Path to the document image.
        max_turns: Maximum number of conversation turns.
    """

    VLM1_MAX_TOKENS = 256
    VLM2_MAX_TOKENS = 500

    def __init__(
        self,
        vlm1_model: Qwen25VLModel,
        vlm2_model: InternVL3Model,
        convince_checker: ConvinceChecker,
        judge_agent: Optional[JudgeAgent] = None,
        vlm1_answer: str = "",
        vlm2_answer: str = "",
        question: str = "",
        image_path: Optional[str] = None,
        max_turns: int = 3,
    ) -> None:
        self.vlm1_model = vlm1_model
        self.vlm2_model = vlm2_model
        self.convince_checker = convince_checker
        self.judge_agent = judge_agent
        self.vlm1_answer = vlm1_answer
        self.vlm2_answer = vlm2_answer
        self.question = question
        self.image_path = image_path
        self.max_turns = max_turns

        self.conversation_history: list[Dict[str, str]] = []
        self.turn_summaries: list[str] = []
        self.convinced = False
        self.final_answer: Optional[str] = None
        self.resolution_method: str = ""

    def _vlm1_prompt(self, turn: int, vlm2_criticism: str = "", judge_summary: str = "") -> str:
        """Generate thesis (VLM1) prompt with structured format."""
        base = (
            "Remember: Answer with the most direct response — "
            "extract only the core identifier, no additional words."
        )
        summary_ctx = f"\n\nJudge Summary:\n{judge_summary}" if judge_summary else ""

        if turn == 0:
            return f"""You are the THESIS agent defending your answer in a structured debate.

Your Answer: "{self.vlm1_answer}"
Question: {self.question}

The ANTITHESIS agent criticized your answer and proposed: "{self.vlm2_answer}"
Their criticism: {vlm2_criticism}
{summary_ctx}

Respond using this structure:

[REFERENCE]
Cite specific evidence from the document that supports your answer.

[CRITICISM]
Address the antithesis agent's objections point by point.

[CONCLUSION]
State your final position — defend your answer or acknowledge if the antithesis is correct.
Never say both answers are correct unless they are exactly the same. {base}"""

        return f"""You are the THESIS agent. Based on the conversation so far:

Antithesis latest criticism: {vlm2_criticism}
{summary_ctx}

Respond using this structure:

[REFERENCE]
Cite evidence from the document supporting your position.

[CRITICISM]
Address the latest objections.

[CONCLUSION]
State your current position with confidence. {base}"""

    def _vlm2_prompt(self, turn: int, vlm1_response: str = "", judge_summary: str = "") -> str:
        """Generate antithesis (VLM2) prompt with structured format."""
        summary_ctx = f"\n\nJudge Summary:\n{judge_summary}" if judge_summary else ""

        if turn == 0:
            return f"""You are the ANTITHESIS agent challenging the thesis in a structured debate.

Question: {self.question}
THESIS answer: "{self.vlm1_answer}"
Your initial answer: "{self.vlm2_answer}"
{summary_ctx}

Respond using this structure:

[REFERENCE]
Cite specific evidence from the document that supports YOUR answer.

[CRITICISM]
Critique the thesis agent's answer — identify errors, misreadings, or unsupported claims.

[CONCLUSION]
State why your answer is more accurate than the thesis."""

        return f"""You are the ANTITHESIS agent continuing the structured debate.

THESIS latest response: {vlm1_response}
{summary_ctx}

Respond using this structure:

[REFERENCE]
New or reinforced evidence supporting your position.

[CRITICISM]
Address the thesis agent's latest arguments. Are you convinced?

[CONCLUSION]
State your current position — maintain or update your answer."""

    def _vlm1_turn(self, vlm2_criticism: str = "", judge_summary: str = "") -> str:
        """VLM1 (Thesis) turn — Qwen2.5-VL with system+user message."""
        try:
            turn_num = len(self.conversation_history) // 2
            prompt = self._vlm1_prompt(turn_num, vlm2_criticism, judge_summary)
            system_message = "You are a helpful assistant that provides direct, concise answers using structured debate format."

            from qwen_vl_utils import process_vision_info

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.image_path},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text = self.vlm1_model.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vlm1_model.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            generated_ids = self.vlm1_model.model.generate(
                **inputs, max_new_tokens=self.VLM1_MAX_TOKENS
            )
            trimmed = [
                out[len(inp) :]
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.vlm1_model.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text[0].strip() if output_text else ""
        except Exception as e:
            logger.error("VLM1 turn error: %s", e)
            response = "Error processing response"

        self.conversation_history.append({"speaker": "THESIS", "message": response})
        return response

    def _vlm2_turn(self, vlm1_response: str = "", judge_summary: str = "") -> str:
        """VLM2 (Antithesis) turn — InternVL3 with URL-based image format."""
        try:
            turn_num = len(self.conversation_history) // 2
            prompt = self._vlm2_prompt(turn_num, vlm1_response, judge_summary)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": self.image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self.vlm2_model.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.vlm2_model.model.device, dtype=torch.bfloat16)

            generate_ids = self.vlm2_model.model.generate(
                **inputs, max_new_tokens=self.VLM2_MAX_TOKENS
            )
            decoded = self.vlm2_model.processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            response = decoded.strip() if decoded else ""
        except Exception as e:
            logger.error("VLM2 turn error: %s", e)
            response = "Error processing response"

        self.conversation_history.append({"speaker": "ANTITHESIS", "message": response})
        return response

    def _generate_judge_summary(self, turn_number: int) -> str:
        """Have the judge generate a structured summary for the current turn."""
        if self.judge_agent is None:
            return ""

        turn_messages = self.conversation_history[-2:]
        turn_text = "\n".join(
            f"{m['speaker']}: {m['message']}" for m in turn_messages
        )

        summary = self.judge_agent.generate_turn_summary(
            question=self.question,
            thesis_answer=self.vlm1_answer,
            antithesis_answer=self.vlm2_answer,
            turn_text=turn_text,
            turn_number=turn_number,
        )
        self.turn_summaries.append(summary)
        return summary

    def _format_conversation(self) -> str:
        lines = [
            f"Question: {self.question}",
            f"THESIS initial answer: {self.vlm1_answer}",
            f"ANTITHESIS initial answer: {self.vlm2_answer}",
            "",
            "Conversation:",
        ]
        for i, turn in enumerate(self.conversation_history):
            lines.append(f"{turn['speaker']}: {turn['message']}")
            summary_idx = i // 2
            if i % 2 == 1 and summary_idx < len(self.turn_summaries):
                lines.append(f"\n--- Judge Summary (Turn {summary_idx + 1}) ---")
                lines.append(self.turn_summaries[summary_idx])
        return "\n".join(lines)

    def _check_convinced(self) -> bool:
        if not self.conversation_history:
            return False
        conversation = self._format_conversation()
        last_msg = self.conversation_history[-1]["message"]
        try:
            result = self.convince_checker.check_conversation(conversation, last_msg)
            return result.strip().upper() == "C"
        except Exception as e:
            logger.error("ConvinceChecker error: %s", e)
            return False

    def _resolve_via_linguistic_analysis(self) -> Optional[str]:
        """Resolve the debate via linguistic confidence analysis after 3 turns."""
        conversation = self._format_conversation()
        try:
            analysis = self.convince_checker.analyze_linguistic_confidence(conversation)
            logger.info("Linguistic confidence analysis: %s", analysis[:200])

            analysis_upper = analysis.upper()
            if "LIKELY_WINNER: VLM1" in analysis_upper or "LIKELY_WINNER: THESIS" in analysis_upper:
                self.resolution_method = "linguistic_confidence_thesis"
                return self.vlm1_answer
            elif "LIKELY_WINNER: VLM2" in analysis_upper or "LIKELY_WINNER: ANTITHESIS" in analysis_upper:
                self.resolution_method = "linguistic_confidence_antithesis"
                return self.vlm2_answer
        except Exception as e:
            logger.error("Linguistic analysis error: %s", e)
        return None

    def _resolve_final_answer(self, conversation: str, where: int = 1) -> str:
        """Determine the final answer when an agent is convinced."""
        if not self.conversation_history:
            return self.vlm2_answer

        if self.convinced:
            f_answer = self.convince_checker.get_final_answer(conversation, self.question)
            if (
                str(self.vlm2_answer) not in f_answer
                and str(self.vlm1_answer) not in f_answer
            ):
                return self.vlm2_answer if where == 1 else self.vlm1_answer

            len_f = len(f_answer)
            len_v1 = len(str(self.vlm1_answer))
            len_v2 = len(str(self.vlm2_answer))
            if (
                len_f == len_v1
                or len_f == len_v2
                or len_f == len_v1 + 1
                or len_f == len_v2 + 1
            ):
                return f_answer
            return self.vlm2_answer if where == 1 else self.vlm1_answer

        return self.vlm2_answer

    def generate_conversation(self) -> str:
        """Run the full multi-turn structured debate.

        Protocol:
        1. Antithesis opens by criticizing the thesis.
        2. For each turn: thesis responds → judge summarises → check convinced.
        3. If not convinced, antithesis responds → judge summarises → check.
        4. After 3 turns, apply linguistic confidence analysis.
        5. If still unresolved, default to antithesis answer.

        Returns:
            The final answer string.
        """
        logger.info("Starting Stage 4 multi-turn debate for: %s", self.question[:80])

        self._vlm2_turn()
        judge_summary = ""

        for turn_idx in range(self.max_turns):
            vlm1_response = self._vlm1_turn(
                self.conversation_history[-1]["message"],
                judge_summary,
            )

            judge_summary = self._generate_judge_summary(turn_idx + 1)

            self.convinced = self._check_convinced()
            if self.convinced:
                self.resolution_method = "convinced_after_thesis"
                self.final_answer = self._resolve_final_answer(
                    self._format_conversation(), where=1
                )
                return str(self.final_answer)

            # After 3 turns, try linguistic confidence analysis
            if turn_idx + 1 >= 3:
                resolved = self._resolve_via_linguistic_analysis()
                if resolved is not None:
                    self.final_answer = resolved
                    return str(self.final_answer)

            if turn_idx < self.max_turns - 1:
                vlm2_response = self._vlm2_turn(vlm1_response, judge_summary)

                judge_summary = self._generate_judge_summary(turn_idx + 1)

                self.convinced = self._check_convinced()
                if self.convinced:
                    self.resolution_method = "convinced_after_antithesis"
                    self.final_answer = self._resolve_final_answer(
                        self._format_conversation(), where=2
                    )
                    return str(self.final_answer)

        # No agreement — default to antithesis answer
        self.resolution_method = "default_antithesis"
        self.final_answer = self.vlm2_answer
        return str(self.final_answer)

    def get_summary(self) -> Dict[str, Any]:
        """Return a structured summary of the conversation."""
        return {
            "initial_answers": {
                "THESIS": self.vlm1_answer,
                "ANTITHESIS": self.vlm2_answer,
            },
            "conversation_history": self.conversation_history,
            "turn_summaries": self.turn_summaries,
            "convinced": self.convinced,
            "resolution_method": self.resolution_method,
            "final_answer": self.final_answer,
            "total_turns": len(self.conversation_history),
        }
