from __future__ import annotations


def choice_index_from_labels(labels: list[str], answer_key: str) -> int:
    # Resolve a dataset-provided answer label into the zero-based index used by the multiple-choice helper.
    return labels.index(answer_key.strip())


def question_answer_prompt(question: str) -> str:
    # Build the standard question-and-answer prompt header used by several benchmark families.
    return f"Question: {question.strip()}\nAnswer:"
