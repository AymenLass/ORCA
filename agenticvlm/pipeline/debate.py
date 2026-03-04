"""Stage 3 — Stress Testing via 2-turn debate loop.

The Stress Testing stage validates the specialist's answer (a_E) against
the thinker's answer (a_T).  It runs **only when a_E ≠ a_T** (≈23.4 %
of cases, per the paper).

**Protocol (2-turn loop):**

    For t = 1, 2:
        1. Debate agent generates a challenging question targeting the
           specialist's answer.
        2. Specialist responds with (response, revised answer).
        3. Evaluation agent assesses **PASS** or **FAIL**.

    If **both** turns pass  → a_D = a_E, skip to Stage 5.
    If **any** turn fails   → proceed to Stage 4 (multi-turn conversation).

This replaces the old five-step debate orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agenticvlm.agents.base import BaseSpecialistAgent
from agenticvlm.agents.debate_agents import DebateAgent, EvaluationAgent

logger = logging.getLogger(__name__)


class StressTestResult:
    """Structured result from the stress-testing stage."""

    __slots__ = (
        "passed",
        "turns",
        "final_answer",
        "debate_answer",
    )

    def __init__(self) -> None:
        self.passed: bool = False
        self.turns: list[Dict[str, Any]] = []
        self.final_answer: str = ""
        self.debate_answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "turns": self.turns,
            "final_answer": self.final_answer,
            "debate_answer": self.debate_answer,
        }


class StressTestOrchestrator:
    """Orchestrates the 2-turn stress testing loop (Stage 3).

    Args:
        debate_agent: Generates challenging debate questions.
        evaluation_agent: Assesses pass/fail per turn.
        specialist_agent: The document-type specialist that defends its
            answer against the challenges.
        num_turns: Number of stress-testing turns (default 2 per paper).
    """

    def __init__(
        self,
        debate_agent: DebateAgent,
        evaluation_agent: EvaluationAgent,
        specialist_agent: BaseSpecialistAgent,
        num_turns: int = 2,
    ) -> None:
        self.debate_agent = debate_agent
        self.evaluation_agent = evaluation_agent
        self.specialist = specialist_agent
        self.num_turns = num_turns

    def run(
        self,
        image_path: str,
        question: str,
        thinker_answer: str,
        specialist_answer: str,
        agent_type: str = "",
    ) -> StressTestResult:
        """Execute the 2-turn stress testing loop.

        Args:
            image_path: Path to the document image.
            question: The user question.
            thinker_answer: a_T — the thinker's answer.
            specialist_answer: a_E — the specialist's answer.
            agent_type: Document-type label for context.

        Returns:
            :class:`StressTestResult` indicating whether the specialist
            passed both turns.
        """
        logger.info("Stage 3 — Stress Testing for: %s", question[:80])

        result = StressTestResult()
        current_answer = specialist_answer
        all_passed = True

        for t in range(1, self.num_turns + 1):
            logger.debug("Stress testing turn %d/%d", t, self.num_turns)

            _, debate_challenge = self.debate_agent.generate_debate_question(
                question,
                current_answer,
                thinker_answer,
                agent_type,
            )

            specialist_response = self.specialist.analyze_image(
                image_path,
                question,
                f"Challenge: {debate_challenge}\nYour current answer: {current_answer}",
            )

            revised_answer = self._extract_answer(
                specialist_response, current_answer
            )

            evaluation = self.evaluation_agent.evaluate_turn(
                question=question,
                debate_challenge=debate_challenge,
                specialist_response=specialist_response,
                specialist_answer=revised_answer,
                turn=t,
            )

            turn_record = {
                "turn": t,
                "debate_challenge": debate_challenge,
                "specialist_response": specialist_response,
                "revised_answer": revised_answer,
                "evaluation": evaluation,
            }
            result.turns.append(turn_record)

            if evaluation == "FAIL":
                all_passed = False
                logger.info("Turn %d FAILED — will proceed to Stage 4", t)
                break

            current_answer = revised_answer
            logger.debug("Turn %d PASSED", t)

        result.passed = all_passed
        result.final_answer = current_answer
        result.debate_answer = current_answer if all_passed else specialist_answer

        logger.info(
            "Stress testing %s (answer: %s)",
            "PASSED" if all_passed else "FAILED",
            result.final_answer[:80] if result.final_answer else "",
        )
        return result

    @staticmethod
    def _extract_answer(response: str, fallback: str) -> str:
        """Extract the answer from a specialist's response text."""
        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("ANSWER:"):
                return stripped[len("ANSWER:"):].strip()
        for line in response.split("\n"):
            stripped = line.strip()
            if "final answer" in stripped.lower():
                colon_pos = stripped.find(":")
                if colon_pos != -1:
                    return stripped[colon_pos + 1:].strip()
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        if lines:
            last = lines[-1]
            if len(last) < 200:
                return last
        return fallback


DebateOrchestrator = StressTestOrchestrator
