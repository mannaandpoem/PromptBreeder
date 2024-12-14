import asyncio
import warnings
import random
import re
import logging
import os
import concurrent.futures
from typing import List

from rich import print
import time
from openai import OpenAI

from pb.mutation_operators import mutate, generate
from pb import gsm
from pb.pb_types import EvolutionUnit, Population

logger = logging.getLogger(__name__)

gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')

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

    results = []
    for prompt in prompt_list:
        result = await generate(prompt, "gpt-4o", temperature=0.7)
        results.append(result)

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

    return p

async def _evaluate_fitness(population: Population, num_evals: int) -> Population:
    """Evaluates each prompt P on a batch of Q&A samples, and populates the fitness values."""
    logger.info(f"Starting fitness evaluation...")
    start_time = time.time()

    batch = gsm8k_examples[:num_evals]  # Using fixed samples for reproducibility
    elite_fitness = -1

    # 构建prompt列表
    prompt_list = []
    for unit in population.units:
        unit.fitness = 0  # Reset fitness from past run
        for example in batch:
            prompt = f"{unit.P}\n{example['question']}"
            prompt_list.append((unit, prompt))

    # 创建信号量来限制并发请求数量
    sem = asyncio.Semaphore(8)

    async def process_prompt(i: int, unit, prompt: str):
        async with sem:  # 使用信号量控制并发
            try:
                result = await generate(
                    prompt,
                    "gpt-4o-mini",
                    temperature=0
                )
                return i, unit, result
            except Exception as exc:
                logger.error(f"Exception processing prompt {i}: {exc}")
                return None

    # 创建所有任务
    tasks = [
        process_prompt(i, unit, prompt)
        for i, (unit, prompt) in enumerate(prompt_list)
    ]

    # 并发执行所有任务
    completed_results = await asyncio.gather(*tasks)
    results = [r for r in completed_results if r is not None]

    # 处理结果
    current_elite = None

    # 将结果分配给对应的unit
    for idx, unit, result in results:
        example_idx = idx % num_evals  # 计算这个响应对应哪个例子
        if result is None:
            continue

        answer_text = result
        valid = re.search(gsm.gsm_extract_answer(batch[example_idx]['answer']), answer_text)

        if valid:
            unit.fitness += (1 / num_evals)

        if unit.fitness > elite_fitness:
            current_elite = unit.model_copy()
            elite_fitness = unit.fitness

    # append best unit of generation to the elites list.
    population.elites.append(current_elite)
    end_time = time.time()
    logger.info(f"Done fitness evaluation. {end_time - start_time}s")

    return population
