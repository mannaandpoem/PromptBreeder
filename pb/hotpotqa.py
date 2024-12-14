import json
import re
import string
import os
from collections import Counter
from typing import Callable, List, Tuple, Dict, Any
from venv import logger

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


def read_jsonl(path: str) -> List[Dict[str, str]]:
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

    hotpotqa_examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                # Extract context paragraphs
                paragraphs = [item[1] for item in example["context"] if isinstance(item[1], list)]
                context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)

                # Construct formatted question with context
                question = f"Context: {context_str}\n\nQuestion: {example['question']}"

                # Create normalized example
                hotpotqa_examples.append({
                    "question": question,
                    "answer": example["answer"]
                })

    logger.info(f"Loaded {len(hotpotqa_examples)} examples from {path}")
    return hotpotqa_examples


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

def extract_content(xml_string, tag):
    # 构建正则表达式，匹配指定的标签内容
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, xml_string, re.DOTALL)  # 使用 re.DOTALL 以匹配换行符
    return match.group(1).strip() if match else ""

def calculate_score(ground_truth: str, prediction: str) -> Tuple[float, str]:
    """Calculate F1 score between prediction and ground truth."""
    prediction = extract_content(xml_string=prediction, tag="answer")
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


class HotpotQABenchmark:
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        """Generate model output with retry logic."""
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[
        str, str, str, str, float, float]:
        """Evaluate a single problem.

        Args:
            problem: Problem dictionary containing question, answer and context
            graph: Callable for generating model output
            prompt_template: Optional template for prompt construction

        Returns:
            Tuple of (question, context, prediction, expected_output, score, cost)
        """
        input_text = problem["question"]
        expected_output = problem["answer"]

        # Construct context from paragraphs
        paragraphs = [item[1] for item in problem["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        inputs = f"Context: {context_str}\n\nQuestion: {input_text}"

        try:
            output, cost = await self._generate_output(graph, inputs)
            score, extracted_output = calculate_score(expected_output, output)

            # Log detailed evaluation information
            logger.debug(f"Question: {input_text}")
            logger.debug(f"Prediction: {output}")
            logger.debug(f"Expected: {expected_output}")
            logger.debug(f"Score: {score}")

            return input_text, context_str, output, expected_output, score, cost

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return input_text, context_str, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        """Get column names for evaluation results."""
        return ["question", "context", "prediction", "expected_output", "score", "cost"]
