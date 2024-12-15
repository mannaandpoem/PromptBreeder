import asyncio
import json
import warnings
import random
import re
import logging
import os
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import List

from rich import print
import time
from openai import OpenAI

# from pb import gsm
from pb import drop, aime
from pb.drop import calculate_score_drop
from pb.bbh import calculate_score_bbh
from pb.logger import logger
# from pb.hotpotqa import calculate_score
from pb.mutation_operators import mutate, generate, parallel_generate
from pb.pb_types import EvolutionUnit, Population

# gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')
# hotpotqa_examples = hotpotqa.read_jsonl('pb/data/hotpotqa_validate.jsonl')
# drop_examples = drop.read_jsonl('pb/data/drop_validate.jsonl')
# aime_examples = aime.read_jsonl('pb/data/aime_validate.jsonl')

bbh_causal_examples = bbh.read_jsonl_bbh('pb/data/bbh/causal/bbh_causal_dev.jsonl')
# bbh_logical_examples = bbh.read_jsonl_bbh('pb/data/bbh/logical/bbh_logical_dev.jsonl')
# bbh_movie_examples = bbh.read_jsonl_bbh('pb/data/bbh/movie/bbh_movie_dev.jsonl')
# bbh_salient_examples = bbh.read_jsonl_bbh('pb/data/bbh/salient/bbh_salient_dev.jsonl')
# bbh_snarks_examples = bbh.read_jsonl_bbh('pb/data/bbh/snarks/bbh_snarks_dev.jsonl')


def save_population_units(population: Population, base_dir: str = "population_results", timestamp: str = "") -> str:
    """
    Save the population units to a JSON file with timestamp.

    Args:
        population: Population object containing the units to save
        base_dir: Base directory for saving the results (default: 'results')

    Returns:
        str: Path to the saved file

    Raises:
        IOError: If there's an error creating the directory or saving the file
    """
    try:
        # Create results directory if it doesn't exist
        save_dir = Path(base_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if not timestamp:
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create filename with metadata
        filename = f"population_units_{timestamp}.json"
        file_path = save_dir / filename

        # Prepare data for saving
        save_data = {
            "timestamp": timestamp,
            "population_size": population.size,
            "population_age": population.age,
            "problem_description": population.problem_description,
            "units": [
                {
                    "thinking_style": unit.T,
                    "mutation_prompt": unit.M,
                    "task_prompt": unit.P,
                    "fitness": unit.fitness,
                    "history": unit.history
                }
                for unit in population.units
            ]
        }

        # Save to file with pretty printing
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved population units to {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Error saving population units: {str(e)}", exc_info=True)
        raise


def create_population(tp_set: List, mutator_set: List, problem_description: str) -> Population:
    """samples the mutation_prompts and thinking_styles and returns a 'Population' object.

    Args:
        'size' (int): the size of the population to create.
        'problem_description (D)' (str): the problem description we are optimizing for.
    """
    data = {
        'size': len(tp_set)*len(mutator_set),
        'age': 0,
        'problem_description' : problem_description,
        'elites' : [],
        'units': [EvolutionUnit(**{
            'T' : t, 
            'M' : m,
            'P' : '',
            'fitness' : 0,
            'history' : []
            }) for t in tp_set for m in mutator_set]
    }

    return Population(**data)

async def init_run(population: Population, model: OpenAI, num_evals: int):
    """ The first run of the population that consumes the prompt_description and
    creates the first prompt_tasks.

    Args:
        population (Population): A population created by `create_population`.
        model (OpenAI): OpenAI client instance
        num_evals (int): Number of evaluations to perform
    """

    start_time = time.time()
    messages_list = []

    prompt_list = []
    for unit in population.units:
        prompt = f"{unit.T} {unit.M} INSTRUCTION: {population.problem_description} INSTRUCTION MUTANT = "
        prompt_list.append(prompt)

    # results = []
    # for prompt in prompt_list:
    #     result = await generate(prompt, "gpt-4o", temperature=0.7)
    #     results.append(result)
    results = await parallel_generate(prompt_list, model="gpt-4o", temperature=0.7)
    end_time = time.time()

    logger.info(f"Prompt initialization done. {end_time - start_time}s")

    assert len(results) == population.size, "size of OpenAI response to population is mismatched"

    for i, result in enumerate(results):
        population.units[i].P = result

    await _evaluate_fitness(population, num_evals)

    return population

async def run_for_n(n: int, population: Population, model: OpenAI, num_evals: int):
    """ Runs the genetic algorithm for n generations.
    """
    p = population
    for i in range(n):
        print(f"================== Population {i} ================== ")
        await mutate(p)
        print("done mutation")
        await _evaluate_fitness(p, num_evals)
        print("done evaluation")
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        saved_path = save_population_units(p, timestamp=time_str)
        print(f"Saved population units to {saved_path}")

    return p

async def _evaluate_fitness(population: Population, num_evals: int) -> Population:
    logger.info(f"Starting fitness evaluation...")
    start_time = time.time()

    # batch = aime_examples[:num_evals]
    batch = bbh_causal_examples[:num_evals]
    elite_fitness = -1

    prompt_list = []
    answer_list = []
    unit_map = {}  # 添加映射以跟踪unit

    for i, unit in enumerate(population.units):
        unit.fitness = 0
        unit_map[i] = unit
        for example in batch:
            prompt = f"{unit.P}\n{example['question']}"
            prompt += "Ensure the response concludes with the answer in the format: <answer>answer</answer>."
            prompt_list.append((i, prompt))  # 使用unit索引而不是unit对象
            answer_list.append(example['answer'])

    sem = asyncio.Semaphore(30)

    async def process_prompt(i: int, prompt: str):
        async with sem:
            try:
                result = await generate(prompt, "gpt-4o-mini", temperature=0)
                if result:
                    return (i, result)
                logger.warning(f"Empty result for prompt {i}")
            except Exception as exc:
                logger.error(f"Exception processing prompt {i}: {exc}")
            return None

    tasks = [process_prompt(i, prompt) for i, prompt in prompt_list]

    # 收集结果并过滤掉None
    completed_results = await asyncio.gather(*tasks, return_exceptions=False)
    valid_results = [r for r in completed_results if r is not None]

    if not valid_results:
        logger.error("No valid results obtained from evaluation")
        return population

    # 初始化得分追踪
    unit_scores = {i: 0.0 for i in unit_map.keys()}
    unit_counts = {i: 0 for i in unit_map.keys()}

    # 处理有效结果
    for idx, result in valid_results:
        unit_idx = idx // num_evals  # 获取对应的unit索引
        example_idx = idx % num_evals

        if unit_idx not in unit_map:
            logger.error(f"Invalid unit index: {unit_idx}")
            continue

        answer_text = result
        expected_output = answer_list[example_idx]
        # current_score, extracted_output = calculate_score_drop(expected_output, answer_text)
        current_score, extracted_output = calculate_score_bbh(expected_output, answer_text)

        unit_scores[unit_idx] += current_score
        unit_counts[unit_idx] += 1

        logger.warning(f"Question: {batch[example_idx]['question']}")
        logger.warning(f"Prediction: {answer_text}")
        logger.warning(f"Expected: {expected_output}")
        logger.warning(f"Score: {current_score}")

    # 计算最终得分
    current_elite = None
    for unit_idx, unit in unit_map.items():
        if unit_counts[unit_idx] > 0:
            unit.fitness = unit_scores[unit_idx] / unit_counts[unit_idx]
            if unit.fitness > elite_fitness:
                current_elite = unit.model_copy()
                elite_fitness = unit.fitness

    if current_elite:
        population.elites.append(current_elite)

    end_time = time.time()
    logger.info(f"Done fitness evaluation. {end_time - start_time}s")

    return population
