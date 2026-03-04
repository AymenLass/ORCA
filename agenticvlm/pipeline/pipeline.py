"""AgenticVLM main pipeline — the full 5-stage inference system (ORCA)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agenticvlm.agents.debate_agents import (
    ConvinceChecker,
    DebateAgent,
    EvaluationAgent,
    JudgeAgent,
    RouteChecker,
    SanityChecker,
)
from agenticvlm.agents.internvl_agents import FigureDiagramAgent, ImagePhotoAgent, Saviour
from agenticvlm.agents.multi_turn import MultiTurnConversation
from agenticvlm.agents.qwen_agents import (
    FormAgent,
    FreeTextAgent,
    HandwrittenAgent,
    LayoutAgent,
    TableListAgent,
    YesNoAgent,
)
from agenticvlm.models.base import GenerationConfig
from agenticvlm.models.glm4v import GLM4VModel
from agenticvlm.models.internvl3 import InternVL3Model
from agenticvlm.models.qwen25vl import Qwen25VLModel
from agenticvlm.models.qwen2_ocr import Qwen2OCRModel
from agenticvlm.models.qwen3 import Qwen3Model
from agenticvlm.models.router import RouterModel
from agenticvlm.pipeline.debate import StressTestOrchestrator
from agenticvlm.pipeline.orchestrator import ReactOrchestrator
from agenticvlm.pipeline.question_router import resolve_agent_type
from agenticvlm.utils.text_processing import extract_boxed_answer, mask_thinking

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Structured output from the AgenticVLM 5-stage pipeline."""

    question: str
    image_path: str
    thinker_answer: str = ""
    thinker_thinking: str = ""
    masked_thinking: str = ""
    router_labels: List[str] = field(default_factory=list)
    primary_label: str = ""
    specialist_answer: str = ""
    orchestrator_trace: List[Dict[str, str]] = field(default_factory=list)
    stress_test_skipped: bool = False
    stress_test_passed: bool = False
    stress_test_details: Dict[str, Any] = field(default_factory=dict)
    multi_turn_skipped: bool = False
    multi_turn_summary: Dict[str, Any] = field(default_factory=dict)
    sanity_checked: bool = False
    final_answer: str = ""
    early_termination_stage: Optional[int] = None


class AgenticVLMPipeline:
    """End-to-end AgenticVLM 5-stage inference pipeline.

    Args:
        thinker: GLM-4.5V-9B thinker model.
        router: Router model with Turbo DFS for question classification.
        specialist_model: Qwen2.5-VL-7B for Qwen-type specialists.
        vision_expert: InternVL3-8B for figure/image specialists and antithesis.
        ocr_model: Qwen2-VL-OCR-2B for handwritten text.
        debate_llm: Qwen3-8B for debate agent (enable_thinking=False).
        checker_llm: Qwen3-1.7B for evaluation/judge/convince/sanity
            (enable_thinking=True).
    """

    def __init__(
        self,
        thinker: GLM4VModel,
        router: RouterModel,
        specialist_model: Qwen25VLModel,
        vision_expert: InternVL3Model,
        ocr_model: Qwen2OCRModel,
        debate_llm: Qwen3Model,
        checker_llm: Qwen3Model,
    ) -> None:
        self.thinker = thinker
        self.router = router
        self.specialist_model = specialist_model
        self.vision_expert = vision_expert
        self.ocr_model = ocr_model

        self.debate_agent = DebateAgent(debate_llm)
        self.route_checker = RouteChecker(debate_llm)

        self.evaluation_agent = EvaluationAgent(checker_llm)
        self.convince_checker = ConvinceChecker(checker_llm)
        self.judge_agent = JudgeAgent(checker_llm)
        self.sanity_checker = SanityChecker(checker_llm)

        self.saviour = Saviour(vision_expert)
        self.orchestrator = ReactOrchestrator(debate_llm)

        self._debate_llm = debate_llm
        self._checker_llm = checker_llm

    def _create_specialist(self, label: str):
        """Instantiate the appropriate specialist agent for *label*."""
        canonical = resolve_agent_type(label)
        mapping = {
            "figure/diagram": lambda: FigureDiagramAgent(self.vision_expert),
            "image/photo": lambda: ImagePhotoAgent(self.vision_expert),
            "form": lambda: FormAgent(self.specialist_model),
            "free_text": lambda: FreeTextAgent(self.specialist_model),
            "handwritten": lambda: HandwrittenAgent(self.ocr_model),
            "Yes/No": lambda: YesNoAgent(self.specialist_model),
            "layout": lambda: LayoutAgent(self.specialist_model),
            "table/list": lambda: TableListAgent(self.specialist_model),
        }
        factory = mapping.get(canonical)
        if factory is None:
            logger.warning("No specialist for '%s', defaulting to FreeTextAgent", canonical)
            return FreeTextAgent(self.specialist_model)
        return factory()

    def _run_stage1(self, image_path: str, question: str) -> Dict[str, str]:
        """Stage 1: Thinker generates reasoning trace R and answer a_T.

        Returns:
            Dict with ``thinker_answer``, ``thinking``, ``masked_thinking``.
        """
        logger.info("=== Stage 1: Context Understanding (Thinker) ===")

        full_output, thinker_answer = self.thinker.get_answer(image_path, question)
        masked = mask_thinking(full_output)

        return {
            "thinker_answer": thinker_answer,
            "thinking": full_output,
            "masked_thinking": masked,
        }

    def _run_stage2(
        self,
        image_path: str,
        question: str,
        masked_thinking: str,
        reasoning_path: str = "",
    ) -> Dict[str, Any]:
        """Stage 2: Router → ReAct Orchestrator → Specialists.

        Per Eq. 3–5, the orchestrator uses the full reasoning path
        ``R`` to determine execution order while the masked path
        ``R*`` is provided only to the final specialist.

        Returns:
            Dict with ``router_labels``, ``primary_label``,
            ``specialist_answer``, ``orchestrator_trace``.
        """
        logger.info("=== Stage 2: Collaborative Agent Execution ===")

        router_labels: List[str] = self.router.classify(image_path, question)
        if isinstance(router_labels, str):
            router_labels = [router_labels]

        canonical_labels = [resolve_agent_type(l) for l in router_labels]
        canonical_labels = [l for l in canonical_labels if l != "others"]
        if not canonical_labels:
            canonical_labels = ["free_text"]

        agents_map = {label: self._create_specialist(label) for label in canonical_labels}

        orch_result = self.orchestrator.execute(
            labels=canonical_labels,
            agents=agents_map,
            image_path=image_path,
            question=question,
            masked_thinking=masked_thinking,
            reasoning_path=reasoning_path,
        )

        specialist_answer = orch_result.get("final_answer", "")
        primary_label = self.orchestrator.get_primary_label(canonical_labels)

        route_status = self.route_checker.check_answer(question, specialist_answer)
        if "NF" in route_status.upper():
            logger.info("RouteChecker → Not Found — triggering Saviour fallback")
            specialist_answer = self.saviour.propose_answer(question, image_path)

        return {
            "router_labels": canonical_labels,
            "primary_label": primary_label,
            "specialist_answer": specialist_answer,
            "orchestrator_trace": orch_result.get("trace", []),
        }

    def _run_stage3(
        self,
        image_path: str,
        question: str,
        thinker_answer: str,
        specialist_answer: str,
        primary_label: str,
    ) -> Dict[str, Any]:
        """Stage 3: 2-turn stress testing.

        Only runs when a_E ≠ a_T. Both turns must PASS for the debate
        answer to be accepted; otherwise Stage 4 is triggered.

        Returns:
            Dict with ``passed``, ``skipped``, ``final_answer``,
            ``stress_test_details``.
        """
        # Conditional execution: skip when a_E == a_T (≈77 %)
        if str(thinker_answer).strip().lower() == str(specialist_answer).strip().lower():
            logger.info(
                "Stage 3 SKIPPED — a_T == a_E ('%s')", thinker_answer[:60]
            )
            return {
                "skipped": True,
                "passed": True,
                "final_answer": specialist_answer,
                "stress_test_details": {},
            }

        logger.info("=== Stage 3: Stress Testing (a_T ≠ a_E) ===")

        specialist = self._create_specialist(primary_label)
        stress_tester = StressTestOrchestrator(
            debate_agent=self.debate_agent,
            evaluation_agent=self.evaluation_agent,
            specialist_agent=specialist,
            num_turns=2,
        )
        result = stress_tester.run(
            image_path, question, thinker_answer, specialist_answer, primary_label
        )

        return {
            "skipped": False,
            "passed": result.passed,
            "final_answer": result.final_answer,
            "stress_test_details": result.to_dict(),
        }

    def _run_stage4(
        self,
        image_path: str,
        question: str,
        specialist_answer: str,
    ) -> Dict[str, Any]:
        """Stage 4: Multi-turn structured debate (thesis vs antithesis).

        Thesis = specialist backbone (Qwen2.5-VL).
        Antithesis = InternVL3 (independent vision expert).
        Judge = Qwen3-1.7B (generates inter-turn summaries).

        Returns:
            Dict with ``final_answer``, ``multi_turn_summary``.
        """
        logger.info("=== Stage 4: Multi-turn Conversation ===")

        antithesis_answer = self.saviour.propose_answer(question, image_path)

        conversation = MultiTurnConversation(
            vlm1_model=self.specialist_model,
            vlm2_model=self.vision_expert,
            convince_checker=self.convince_checker,
            judge_agent=self.judge_agent,
            vlm1_answer=specialist_answer,
            vlm2_answer=antithesis_answer,
            question=question,
            image_path=image_path,
            max_turns=3,
        )
        final_answer = conversation.generate_conversation()
        summary = conversation.get_summary()

        return {
            "final_answer": str(final_answer).strip(),
            "multi_turn_summary": summary,
        }

    def _run_stage5(
        self,
        question: str,
        answer: str,
        context: str = "",
    ) -> str:
        """Stage 5: Sanity checker corrects formatting.

        Args:
            question: The original question.
            answer: The current best answer.
            context: Optional document context for formatting reference.

        Returns:
            The formatting-refined final answer.
        """
        logger.info("=== Stage 5: Answer Refinement (Sanity Checker) ===")
        return self.sanity_checker.refine_answer(question, answer, context)

    def predict(self, image_path: str, question: str) -> PipelineResult:
        """Run the complete 5-stage AgenticVLM pipeline on a single sample.

        The pipeline implements the ORCA inference protocol:

        1. Thinker generates reasoning R with answer a_T.
        2. Router + Orchestrator produce specialist answer a_E.
        3. If a_E == a_T → skip to Stage 5 (77 % of cases).
        4. If a_E ≠ a_T → stress test → if fails → multi-turn debate.
        5. Sanity checker refines formatting.

        Args:
            image_path: Path to the document image.
            question: The user question.

        Returns:
            :class:`PipelineResult` with all intermediate and final outputs.
        """
        result = PipelineResult(question=question, image_path=image_path)

        s1 = self._run_stage1(image_path, question)
        result.thinker_answer = s1["thinker_answer"]
        result.thinker_thinking = s1["thinking"]
        result.masked_thinking = s1["masked_thinking"]

        s2 = self._run_stage2(
            image_path, question, s1["masked_thinking"], s1["thinking"],
        )
        result.router_labels = s2["router_labels"]
        result.primary_label = s2["primary_label"]
        result.specialist_answer = s2["specialist_answer"]
        result.orchestrator_trace = s2["orchestrator_trace"]

        current_answer = s2["specialist_answer"]

        s3 = self._run_stage3(
            image_path,
            question,
            s1["thinker_answer"],
            s2["specialist_answer"],
            s2["primary_label"],
        )
        result.stress_test_skipped = s3["skipped"]
        result.stress_test_passed = s3["passed"]
        result.stress_test_details = s3["stress_test_details"]
        current_answer = s3["final_answer"]

        if s3["skipped"]:
            result.early_termination_stage = 2
            result.multi_turn_skipped = True
        elif s3["passed"]:
            # Both turns passed → a_D = a_E, skip Stage 4
            result.early_termination_stage = 3
            result.multi_turn_skipped = True
        else:
            s4 = self._run_stage4(
                image_path, question, current_answer
            )
            current_answer = s4["final_answer"]
            result.multi_turn_summary = s4["multi_turn_summary"]
            result.multi_turn_skipped = False

        refined = self._run_stage5(
            question, current_answer, s1["masked_thinking"]
        )
        result.sanity_checked = True
        result.final_answer = refined

        logger.info(
            "Pipeline complete (early_term=%s) → %s",
            result.early_termination_stage,
            result.final_answer[:100] if result.final_answer else "",
        )
        return result

    def load_all_models(self) -> None:
        """Load all models into memory. Call before ``predict``."""
        logger.info("Loading all models...")
        self.thinker.load()
        self.router.load()
        self.specialist_model.load()
        self.vision_expert.load()
        self.ocr_model.load()
        self._debate_llm.load()
        self._checker_llm.load()
        logger.info("All models loaded.")
