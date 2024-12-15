import json
import os
import re
import string
from collections import Counter
from typing import Callable, List, Tuple, Dict

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


def normalize_answer(s: str) -> str:
    """Normalize answer string by removing articles, punctuation etc."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_score(ground_truth: str, prediction: str) -> Tuple[float, str]:
    """
    Compute the F1 score between prediction and ground truth answers.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, prediction
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, prediction


def calculate_score_drop(ground_truth: str, prediction: str) -> Tuple[float, str]:
    answers = ground_truth.split("|")
    prediction = extract_content(xml_string=prediction, tag="answer")
    f1_scores = []
    for answer in answers:
        if answer.strip() != "":
            output_parts = prediction.split("|")
            for output_part in output_parts:
                f1_score, _ = calculate_score(answer, output_part)
                f1_scores.append(f1_score)

    uni_score = max(f1_scores)

    return uni_score, prediction


def extract_content(xml_string, tag):
    # 构建正则表达式，匹配指定的标签内容
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, xml_string, re.DOTALL)  # 使用 re.DOTALL 以匹配换行符
    return match.group(1).strip() if match else ""


def read_jsonl(path: str) -> List[Dict[str, str]]:
    """Read jsonl file and return list of DROP examples.

    Args:
        path: Path to jsonl file

    Returns:
        List of dicts, each containing:
            - context: String with passage text
            - ref_text: Ground truth answer(s), multiple answers separated by |
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    drop_examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                # Extract passage text as context
                question = example["context"]
                answers = example["ref_text"]

                # Create normalized example
                drop_examples.append({
                    "question": question,
                    "answer": answers,
                })

    print(f"Loaded {len(drop_examples)} examples from {path}")
    return drop_examples


class DROPBenchmark:
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = problem["context"]
        expected_output = problem["ref_text"]
        answers = expected_output.split("|")

        try:
            output, cost = await self._generate_output(graph, input_text)
            f1_scores = []

            for answer in answers:
                if answer.strip() != "":
                    output_parts = output.split("|")
                    for output_part in output_parts:
                        f1_score, _ = calculate_score(answer, output_part)
                        f1_scores.append(f1_score)

            uni_score = max(f1_scores)

            if uni_score < 0.3:
                self.log_mismatch(input_text, expected_output, output, output)

            return input_text, output, expected_output, uni_score, cost

        except Exception as e:
            print(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]
