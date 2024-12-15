import csv
import json
import sys
from typing import Tuple, List, Dict

import openai
import os
import re


def parse_boxed_answer(response_text):
    # Regex to match something like \(\boxed{XYZ}\) or just \boxed{XYZ}
    # We'll try a couple of patterns:
    patterns = [
        r"\\boxed{([^}]*)}",       # \boxed{...}
        r"\(\s*\\boxed{([^}]*)}\s*\)"  # \(\boxed{...}\)
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1).strip()
    return ""


def calculate_score_aime(ground_truth: str, prediction: str) -> Tuple[float, str]:
    """
    Compute the score for AIME math competition answers.

    Args:
        ground_truth: The correct answer (reference answer)
        prediction: The model's predicted answer

    Returns:
        Tuple containing:
            - score: 1.0 if answers match exactly, 0.0 otherwise
            - prediction: The parsed prediction
    """
    # Parse the predicted answer from model output
    parsed_prediction = parse_boxed_answer(prediction)

    if not parsed_prediction:
        # Could not parse a valid answer from the prediction
        return 0.0, prediction

    # Normalize both answers for comparison
    try:
        # Convert strings to integers for exact comparison
        parsed_prediction = int(parsed_prediction)
        ground_truth = int(ground_truth)

        # AIME scoring is binary - either exactly correct or incorrect
        score = 1.0 if parsed_prediction == ground_truth else 0.0

        return score, str(parsed_prediction)

    except ValueError:
        # Handle case where answers cannot be converted to integers
        return 0.0, prediction


def read_jsonl(path: str) -> List[Dict[str, str]]:
    """Read jsonl file and return list of AIME examples.

    Args:
        path: Path to jsonl file

    Returns:
        List of dicts, each containing:
            - question: AIME problem text
            - answer: Reference answer (string)
            - problem_id: Problem identifier (e.g., "2023-II-5")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    aime_examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                example = json.loads(line)

                # Extract required fields
                try:
                    problem_id = example["ID"]
                    question = example["Question"]
                    answer = str(example["Answer"]).strip()

                    # Create normalized example
                    aime_examples.append({
                        "question": question,
                        "answer": answer
                    })

                except KeyError as e:
                    print(f"Missing required field in example: {e}")
                    continue

    print(f"Loaded {len(aime_examples)} examples from {path}")
    return aime_examples


def run_evaluation(model_name, input_csv, mode, api_key):
    """
    Run the evaluation in a given mode (plain or cot).
    Returns (output_filename, correctness_percentage)
    """
    openai.api_key = api_key

    system_message = "You need to provide the final answer in a LaTeX box, i.e. \(\boxed{...}\)"
    cot_prompt = "Please think step by step in greater detail to solve the problem:\n"

    # Read input CSV
    problems = []
    try:
        with open(input_csv, "r", newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                problems.append(row)
    except Exception as e:
        print(f"Error reading input CSV: {str(e)}")
        sys.exit(1)

    if not problems:
        print("No problems found in the input CSV file")
        sys.exit(1)

    output_suffix = f"_{mode}_eval_results.csv"
    output_filename = input_csv.replace(".csv", output_suffix)
    
    correct_count = 0
    total_count = 0

    with open(output_filename, "w", newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["ID", "Problem", "Reference_Answer", "Model_Answer"])

        for problem_data in problems:
            try:
                problem_id = problem_data["ID"]
                problem_text = problem_data["Problem"]
                ref_answer = problem_data["Answer"].strip()
            except KeyError as e:
                print(f"Missing required column in CSV: {str(e)}")
                sys.exit(1)

            if mode == "plain":
                user_content = problem_text
            else:
                # CoT mode
                user_content = cot_prompt + problem_text

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]

            try:
                print(f"Sending request to OpenAI for problem {problem_id}...")
                completion = openai.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=8192
                )
                model_answer = completion.choices[0].message.content.strip()
                print(f"Received response for problem {problem_id}")
            except Exception as e:
                print(f"OpenAI API error for problem {problem_id}: {str(e)}")
                model_answer = f"Error: {str(e)}"

            writer.writerow([problem_id, problem_text, ref_answer, model_answer])

            # Check correctness
            parsed = parse_boxed_answer(model_answer)
            if parsed:
                total_count += 1
                if parsed == ref_answer:
                    correct_count += 1
                    print(f"Problem {problem_id}: Correct")
                else:
                    print(f"Problem {problem_id}: Incorrect")
                    print(f"Expected: {ref_answer}")
                    print(f"Got: {parsed}")
            else:
                total_count += 1
                print(f"Problem {problem_id}: Could not parse answer from response")
                print(f"Raw response: {model_answer}")

    # Compute correctness percentage
    if total_count > 0:
        correctness_percentage = (correct_count / total_count) * 100
    else:
        correctness_percentage = 0.0

    print(f"\nSummary:")
    print(f"Total problems: {total_count}")
    print(f"Correct answers: {correct_count}")

    return output_filename, correctness_percentage

def main():
    if len(sys.argv) < 4:
        print("Usage: python run_eval.py <model_name> <input_csv> <mode>")
        print("mode: 'plain', 'cot', or 'both'")
        sys.exit(1)
    
    model_name = sys.argv[1]
    input_csv = sys.argv[2]
    mode = sys.argv[3]

    if mode not in ["plain", "cot", "both"]:
        print("Mode must be 'plain', 'cot', or 'both'")
        sys.exit(1)

    if not os.path.exists(input_csv):
        print(f"Input CSV file not found: {input_csv}")
        sys.exit(1)
    
    # Prompt user for API key
    api_key = input("Please enter your OpenAI API key: ").strip()
    if not api_key:
        print("API key cannot be empty")
        sys.exit(1)

    if mode in ["plain", "cot"]:
        output_filename, correctness_percentage = run_evaluation(model_name, input_csv, mode, api_key)
        print(f"Done. Results saved in {output_filename}.")
        print(f"Correctness: {correctness_percentage:.2f}%")
    else:
        # both mode
        output_plain, correctness_plain = run_evaluation(model_name, input_csv, "plain", api_key)
        print(f"Plain mode done. Results in {output_plain}, correctness: {correctness_plain:.2f}%")

        output_cot, correctness_cot = run_evaluation(model_name, input_csv, "cot", api_key)
        print(f"COT mode done. Results in {output_cot}, correctness: {correctness_cot:.2f}%")

if __name__ == "__main__":
    main()
