# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.qa_text import best_qa_scores, canonicalize_no_answer

_STOP_STRINGS = ("\n", "\nQuestion:", "\nStory:")


def _coqa_prompt(
    *,
    story: str,
    history_questions: list[str],
    history_answers: list[str],
    question: str,
) -> str:
    if len(history_questions) != len(history_answers):
        raise ValueError("coqa history question/answer counts must match")

    lines = [f"Story: {story.strip()}"]
    for previous_question, previous_answer in zip(history_questions, history_answers, strict=True):
        lines.append(f"Question: {previous_question.strip()}")
        lines.append(f"Answer: {previous_answer.strip()}")
    lines.append(f"Question: {question.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def _load_coqa_turns(
    dataset_path: str,
    *,
    split: str,
    cache_dir: str | None,
    streaming: bool,
) -> Dataset:
    if streaming:
        raise ValueError("coqa turn flattening requires non-streaming dataset loading")

    conversations = load_dataset(
        dataset_path,
        split=split,
        cache_dir=cache_dir,
        streaming=False,
    )
    flattened: list[dict[str, Any]] = []
    for conversation_index, conversation in enumerate(conversations):
        story = str(conversation["story"])
        source = str(conversation["source"])
        questions = [str(question).strip() for question in conversation["questions"]]
        answers = [str(answer).strip() for answer in conversation["answers"]["input_text"]]
        answer_starts = [int(offset) for offset in conversation["answers"]["answer_start"]]
        answer_ends = [int(offset) for offset in conversation["answers"]["answer_end"]]
        if not (
            len(questions)
            == len(answers)
            == len(answer_starts)
            == len(answer_ends)
        ):
            raise ValueError("coqa questions and answers must have matching turn counts")

        turn_count = len(questions)
        for turn_offset, (question, answer, answer_start, answer_end) in enumerate(
            zip(questions, answers, answer_starts, answer_ends, strict=True)
        ):
            flattened.append(
                {
                    "source": source,
                    "story": story,
                    "question": question,
                    "answer": answer,
                    "answer_start": answer_start,
                    "answer_end": answer_end,
                    "history_questions": questions[:turn_offset],
                    "history_answers": answers[:turn_offset],
                    "conversation_index": conversation_index,
                    "turn_index": turn_offset + 1,
                    "turn_count": turn_count,
                }
            )

    return Dataset.from_list(flattened)


@dataclass(slots=True)
class CoQA(BaseTestSuite):
    dataset_path: str = "coqa"
    dataset_name: str | None = None
    split: str = "validation"
    max_rows: int | None = None
    max_new_tokens: int = 32
    batch_size: int | None = None
    cache_dir: str | None = None
    streaming: bool = False
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        return _load_coqa_turns

    def task_name(self) -> str:
        return "coqa"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_qa_exact_match_f1",
            "primary_metric": "f1",
            "prompt_mode": "gold_history_conversation",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=str(doc["answer"]),
                request=GenerationRequest(
                    prompt=_coqa_prompt(
                        story=str(doc["story"]),
                        history_questions=list(doc["history_questions"]),
                        history_answers=list(doc["history_answers"]),
                        question=str(doc["question"]),
                    ),
                    stop=list(_STOP_STRINGS),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        answer = str(prepared_sample.doc["answer"])
        exact, f1_score, best_index = best_qa_scores(output.text, [answer])
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-normalized": canonicalize_no_answer(output.text),
                "best_answer_index": str(best_index),
                "best_answer": answer,
            },
            scores={
                "em": exact,
                "f1": f1_score,
            },
            metadata={
                "source": str(prepared_sample.doc["source"]),
                "conversation_index": int(prepared_sample.doc["conversation_index"]),
                "turn_index": int(prepared_sample.doc["turn_index"]),
                "turn_count": int(prepared_sample.doc["turn_count"]),
                "history_turns": len(prepared_sample.doc["history_questions"]),
                "question": str(prepared_sample.doc["question"]),
                "answer_start": int(prepared_sample.doc["answer_start"]),
                "answer_end": int(prepared_sample.doc["answer_end"]),
            },
        )


def coqa(**kwargs: Any) -> CoQA:
    return CoQA(**kwargs)
