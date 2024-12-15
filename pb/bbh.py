import json
import re
import string
import os
from collections import Counter
from typing import Callable, List, Tuple, Dict, Any
from venv import logger

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


def read_jsonl_bbh(path: str) -> List[Dict[str, str]]:
    """Read jsonl file and return list of HotpotQA examples.

    Args:
        path: Path to jsonl file

    Returns:
        List of dicts, each containing:
            - question: String with context and question
            - answer: Ground truth answer
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    bbh_examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                example = json.loads(line)

                question = example['question']

                # Create normalized example
                bbh_examples.append({
                    "question": question,
                    "answer": example["answer"]
                })

    logger.info(f"Loaded {len(bbh_examples)} examples from {path}")
    return bbh_examples


def extract_content(xml_string, tag):
    # 构建正则表达式，匹配指定的标签内容
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, xml_string, re.DOTALL)  # 使用 re.DOTALL 以匹配换行符
    return match.group(1).strip() if match else ""


def normalize_answer(s: str) -> str:
    """
    Normalize answer for evaluation by:
    1. Converting to lowercase
    2. Removing parentheses, brackets around options
    3. Removing whitespace
    """
    # Remove various forms of option markers: (A), [A], A), A.
    s = re.sub(r'[\(\[\{]([A-Za-z])[\)\]\}]|([A-Za-z])[\.:\)]', r'\1\2', s)
    return s.lower().strip()


def calculate_score_bbh(ground_truth: str, prediction: str) -> Tuple[float, str]:
    """
    Compute exact match score between prediction and ground truth answers.
    Score is 1.0 if strings match exactly after normalization, 0.0 otherwise.
    """
    prediction = extract_content(xml_string=prediction, tag="answer")
    return (1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0, prediction)


class BBHBenchmark:
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = problem["question"]
        expected_output = problem["answer"]
        answers = expected_output.split("|")

        try:
            output, cost = await self._generate_output(graph, input_text)
            scores = []

            for answer in answers:
                if answer.strip() != "":
                    output_parts = output.split("|")
                    for output_part in output_parts:
                        score, _ = calculate_score_bbh(answer, output_part)
                        scores.append(score)

            # If any output part exactly matches any answer, score is 1.0
            em_score = 1.0 if any(scores) else 0.0

            if em_score < 1.0:
                self.log_mismatch(input_text, expected_output, output, output)

            return input_text, output, expected_output, em_score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]