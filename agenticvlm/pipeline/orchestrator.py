"""ReAct Orchestrator — dynamically sequences specialist agents."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from agenticvlm.agents.base import BaseSpecialistAgent
from agenticvlm.data.label_definitions import LABEL_DEFINITIONS
from agenticvlm.models.base import GenerationConfig
from agenticvlm.pipeline.question_router import AGENT_PRIORITY

logger = logging.getLogger(__name__)

_REACT_GEN_CONFIG = GenerationConfig(
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
)

_REACT_SYSTEM = (
    "You are an orchestrator inside ORCA, a multi-agent document VQA system. "
    "Your role is to decide which document-analysis specialist to invoke next "
    "so that the ensemble produces the best possible answer.\n\n"
    "Rules:\n"
    "1. Select ONE specialist per step, or say FINISH if enough information "
    "has been gathered.\n"
    "2. You MUST invoke at least one specialist before finishing.\n"
    "3. Prioritise the specialist whose capability is most critical for the "
    "question; use previous observations to decide what is still missing.\n"
    "4. Respond in EXACTLY this format (no extra keys):\n"
    "   Thought: <your reasoning>\n"
    "   Action: <specialist_label or FINISH>"
)


class ReactOrchestrator:
    """ReAct-style orchestrator that sequences specialists dynamically.

    Args:
        reasoning_model: Any model that exposes
            ``generate_text(prompt, gen_config) -> str``.
    """

    def __init__(self, reasoning_model: Any) -> None:
        self.reasoning_model = reasoning_model

    def _build_prompt(
        self,
        question: str,
        available: List[str],
        observations: List[Dict[str, str]],
        reasoning_path: str = "",
    ) -> str:
        """Assemble the ReAct prompt sent to the reasoning model.

        The prompt includes the thinker's reasoning path ``R`` so the
        orchestrator can align agent invocation order with the
        decomposed sub-tasks (Eq. 3 in the paper).
        """
        label_lookup = {k.lower(): v for k, v in LABEL_DEFINITIONS.items()}

        parts: List[str] = [f"QUESTION: {question}\n"]

        if reasoning_path:
            preview = reasoning_path[:800]
            parts.append(f"THINKER REASONING PATH:\n{preview}\n")

        parts.append("AVAILABLE SPECIALISTS:")
        for label in available:
            desc = label_lookup.get(label.lower(), label)
            parts.append(f"  - {label}: {desc}")

        if observations:
            parts.append("\nPREVIOUS OBSERVATIONS:")
            for obs in observations:
                parts.append(f"\n[Specialist: {obs['label']}]")
                preview = obs["output"][:500]
                parts.append(f"Output: {preview}")

        parts.append(
            "\nBased on the reasoning path and the question, decide which "
            "specialist to invoke next. If you already have enough "
            "information, respond with FINISH.\n"
            "Thought: "
        )
        return "\n".join(parts)

    @staticmethod
    def _parse_action(
        response: str,
        available: List[str],
    ) -> Tuple[Optional[str], bool]:
        """Extract the selected action from the model's response.

        Returns:
            ``(selected_label, is_finish)`` — *selected_label* is
            ``None`` when the model signals FINISH or output is
            unparseable.
        """
        match = re.search(r"Action:\s*(.+)", response, re.IGNORECASE)
        if not match:
            return None, False

        action = match.group(1).strip().strip("'\"")

        if action.upper() == "FINISH":
            return None, True

        action_lower = action.lower()
        for label in available:
            if label.lower() == action_lower:
                return label, False

        for label in available:
            if action_lower in label.lower() or label.lower() in action_lower:
                return label, False

        logger.warning(
            "ReAct selected unknown specialist '%s'; falling back to first available",
            action,
        )
        return (available[0] if available else None), False

    def execute(
        self,
        labels: List[str],
        agents: Dict[str, BaseSpecialistAgent],
        image_path: str,
        question: str,
        masked_thinking: str = "",
        reasoning_path: str = "",
    ) -> Dict[str, Any]:
        """Run the ReAct loop to execute agents in a reasoned order.

        Implements Eq. 3–5 from the paper::

            σ = Orchestrate(A_active, R, q, D)   # ordering via R
            a_i = σ_i(q, D, a_{i-1})              # sequential chain
            a_E = σ_n(q, D, a_{n-1}, R*)           # final agent gets R*

        Args:
            labels: Activated router labels (``A_active``).
            agents: Pre-created specialist agents keyed by label.
            image_path: Path to the document image (``D``).
            question: The user question (``q``).
            masked_thinking: Thinker's masked reasoning path ``R*``
                (provided only to the final agent, Eq. 5).
            reasoning_path: Full thinker reasoning path ``R`` used
                by the orchestrator to determine execution order
                (Eq. 3).  Distinct from ``masked_thinking``.

        Returns:
            Dictionary with ``trace``, ``final_answer``,
            ``execution_order``, and ``react_thoughts``.
        """
        if not labels:
            logger.warning("No labels provided — using 'others' fallback")
            labels = ["others"]

        if len(labels) == 1:
            label = labels[0]
            agent = agents.get(label)
            if agent is None:
                return {
                    "trace": [],
                    "final_answer": "",
                    "execution_order": [],
                    "react_thoughts": [],
                }
            context = (
                f"Thinker reasoning (masked): {masked_thinking}"
                if masked_thinking
                else ""
            )
            output = agent.analyze_image(image_path, question, context)
            return {
                "trace": [{"label": label, "output": output}],
                "final_answer": output,
                "execution_order": [label],
                "react_thoughts": [],
            }

        available = list(labels)
        observations: List[Dict[str, str]] = []
        trace: List[Dict[str, str]] = []
        execution_order: List[str] = []
        react_thoughts: List[str] = []
        max_iterations = len(labels)

        for iteration in range(max_iterations):
            prompt = self._build_prompt(
                question, available, observations, reasoning_path,
            )
            response = self.reasoning_model.generate_text(
                prompt, gen_config=_REACT_GEN_CONFIG,
            )
            logger.debug(
                "ReAct iteration %d response: %s",
                iteration + 1,
                response[:300],
            )
            react_thoughts.append(response)

            selected, is_finish = self._parse_action(response, available)

            if is_finish and trace:
                break
            if selected is None:
                if available:
                    selected = available[0]
                else:
                    break

            agent = agents.get(selected)
            if agent is None:
                logger.warning("No agent for '%s', skipping", selected)
                available.remove(selected)
                continue

            available.remove(selected)
            is_last = is_finish or not available

            previous_output = observations[-1]["output"] if observations else ""
            if is_last and masked_thinking:
                context = f"Thinker reasoning (masked): {masked_thinking}"
                if previous_output:
                    context += f"\nPrevious agent analysis: {previous_output}"
            elif previous_output:
                context = f"Previous agent analysis: {previous_output}"
            else:
                context = ""

            logger.info(
                "ReAct step %d: invoking '%s' (is_last=%s)",
                iteration + 1,
                selected,
                is_last,
            )
            output = agent.analyze_image(image_path, question, context)
            trace.append({"label": selected, "output": output})
            observations.append({"label": selected, "output": output})
            execution_order.append(selected)

            if is_last:
                break

        final_answer = trace[-1]["output"] if trace else ""
        logger.info("ReAct execution order: %s", execution_order)

        return {
            "trace": trace,
            "final_answer": final_answer,
            "execution_order": execution_order,
            "react_thoughts": react_thoughts,
        }

    @staticmethod
    def get_primary_label(labels: List[str]) -> str:
        """Return the highest-priority label from a list.

        Falls back to the static priority table for determinism — this
        method is used outside the ReAct loop (e.g. to pick a single
        representative specialist for the debate stage).
        """
        if not labels:
            return "others"
        return min(labels, key=lambda l: AGENT_PRIORITY.get(l, 99))
