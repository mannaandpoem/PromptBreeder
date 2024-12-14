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

from pb.hotpotqa import calculate_score
from pb.mutation_operators import mutate, generate
from pb import gsm
from pb.pb_types import EvolutionUnit, Population

logger = logging.getLogger(__name__)

# gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')
hotpotqa_examples = hotpotqa.read_jsonl('pb/data/hotpotqa_validate.jsonl')


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
    """
    评估种群中每个prompt P在Q&A样本批次上的表现，并计算得分。

    Args:
        population: 包含多个单位的种群
        num_evals: 用于评估的样本数量

    Returns:
        更新了得分的种群
    """
    logger.info(f"Starting fitness evaluation...")
    start_time = time.time()

    # 使用固定样本以保证可重现性
    batch = hotpotqa_examples[:num_evals]
    elite_fitness = -1  # 记录最优得分

    # 构建prompt列表和对应的答案列表
    prompt_list = []
    answer_list = []
    for unit in population.units:
        unit.fitness = 0  # 重置上一轮的得分
        for example in batch:
            # 将单位的prompt与问题组合
            prompt = f"{unit.P}\n{example['question']}"
            prompt_list.append((unit, prompt))
            answer_list.append(example['answer'])

    # 创建信号量来限制并发请求数量
    sem = asyncio.Semaphore(8)

    async def process_prompt(i: int, unit, prompt: str):
        """
        处理单个prompt的异步函数

        Args:
            i: prompt的索引
            unit: 对应的种群单位
            prompt: 待处理的prompt文本

        Returns:
            元组 (索引, 单位, 生成结果) 或 None(如果发生错误)
        """
        async with sem:  # 使用信号量控制并发
            try:
                prompt += "Ensure the response concludes with the answer in the format: <answer>answer</answer>."
                result = await generate(
                    prompt,
                    "gpt-4o-mini",
                    temperature=0
                )
                return i, unit, result
            except Exception as exc:
                logger.error(f"Exception processing prompt {i}: {exc}")
                return None

    # 创建所有异步任务
    tasks = [
        process_prompt(i, unit, prompt)
        for i, (unit, prompt) in enumerate(prompt_list)
    ]

    # 并发执行所有任务
    completed_results = await asyncio.gather(*tasks)
    results = [r for r in completed_results if r is not None]

    # 处理生成结果
    current_elite = None

    # 用于存储每个unit的得分和计数的列表
    unit_scores = [0.0] * len(population.units)  # 存储每个unit的得分总和
    unit_counts = [0] * len(population.units)  # 存储每个unit的有效样本数

    # 将结果分配给对应的unit并计算得分
    for idx, unit, result in results:
        if result is None:
            continue

        # 计算unit在population中的索引
        unit_idx = population.units.index(unit)
        example_idx = idx % num_evals  # 计算这个响应对应哪个例子

        answer_text = result
        expected_output = answer_list[example_idx]
        # 计算当前答案的得分
        current_score, extracted_output = calculate_score(expected_output, answer_text)

        # 累计每个unit的得分和样本数
        unit_scores[unit_idx] += current_score
        unit_counts[unit_idx] += 1

        # 记录详细的评估信息
        logger.warning(f"Question: {batch[example_idx]['question']}")
        logger.warning(f"Prediction: {answer_text}")
        logger.warning(f"Expected: {expected_output}")
        logger.warning(f"Score: {current_score}")

    # 计算每个unit的平均得分
    for i, unit in enumerate(population.units):
        if unit_counts[i] > 0:  # 确保有有效样本
            unit.fitness = unit_scores[i] / unit_counts[i]
            # 更新精英个体
            if unit.fitness > elite_fitness:
                current_elite = unit.model_copy()
                elite_fitness = unit.fitness

    # 将当代最优个体添加到精英列表中
    population.elites.append(current_elite)

    end_time = time.time()
    logger.info(f"Done fitness evaluation. {end_time - start_time}s")

    return population
