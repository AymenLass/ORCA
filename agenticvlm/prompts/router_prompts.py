"""Prompt template for the question-type router model."""

ROUTER_CLASSIFICATION_PROMPT = """Analyze this image and the following question to determine the appropriate answer type(s).

Question: {question}

TASK: Select the label(s) that best describe what type of answer this question requires by looking at the image content.

AVAILABLE LABELS:
- figure/diagram
- Yes/No
- table/list
- layout
- Image/Photo
- handwritten
- free_text
- form
- others

INSTRUCTIONS:
1. Examine the image carefully
2. Consider what the question is asking about
3. Select the most appropriate label(s)
4. If multiple labels apply, separate them with commas
5. Only use labels from the list above

Answer with only the label(s):"""

from agenticvlm.data.label_definitions import ROUTER_LABELS  # noqa: E402

__all__ = ["ROUTER_CLASSIFICATION_PROMPT", "ROUTER_LABELS"]
