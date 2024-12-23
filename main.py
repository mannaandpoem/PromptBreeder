import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from pb import create_population, init_run, run_for_n, Population, logger, save_population_units
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles

import os
import logging
import argparse
import asyncio

from dotenv import load_dotenv
from rich import print

from pb.token_manager import get_token_tracker

load_dotenv() # load environment variables

parser = argparse.ArgumentParser(description='Run the PromptBreeder Algorithm. Number of units is mp * ts.')
parser.add_argument('-mp', '--num_mutation_prompts', default=5)
parser.add_argument('-ts', '--num_thinking_styles', default=5)
parser.add_argument('-e', '--num_evals', default=20)
parser.add_argument('-n', '--simulations', default=20)
parser.add_argument('-p', '--problem', default="Solve the math word problem, giving your answer as an arabic numeral.")
# parser.add_argument('-p', '--problem', default="Answer the given question by finding and connecting relevant information across multiple provided paragraphs of text.")
# parser.add_argument('-p', '--problem', default="Answer the given question by performing numerical reasoning and calculations based on information found in the provided paragraph.")
# parser.add_argument('-p', '--problem', default="Solve the given mathematical problem using advanced techniques, providing a three-digit integer answer between 000 and 999.")
# parser.add_argument('-p', '--problem', default="Solve the given reasoning problem, which may involve logical deduction, mathematics, coding, or abstract thinking.")

args = vars(parser.parse_args())

total_evaluations = int(args['num_mutation_prompts']) * int(args['num_thinking_styles']) * int(args['num_evals'])
# 修改后的代码


async def main():
    # Setup initial parameters
    tp_set = mutation_prompts[:int(args['num_mutation_prompts'])]
    mutator_set = thinking_styles[:int(args['num_thinking_styles'])]

    logger.info(f'You are prompt-optimizing for the problem: {args["problem"]}')

    # Create population
    logger.info('Creating the population...')
    p = create_population(
        tp_set=tp_set,
        mutator_set=mutator_set,
        problem_description=args['problem']
    )

    # Generate initial prompts
    logger.info('Generating the initial prompts...')
    await init_run(p, int(args['num_evals']))

    # Run genetic algorithm
    logger.info('Starting the genetic algorithm...')
    await run_for_n(
        n=int(args['simulations']),
        population=p,
        num_evals=int(args['num_evals'])
    )

    # Print results and save statistics
    print("%" * 80)
    print("done processing! final gen:")
    print(p.units)

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    saved_path = save_population_units(p, timestamp=time_str)
    print(f"Results saved to: {saved_path}")

    token_tracker = get_token_tracker()
    print(f"Total cost: ${token_tracker.calculate_estimated_cost():.3f}")
    print("*" * 50)
    token_tracker.print_usage_report()

    # Save token usage data
    file_name = f"cost/token_usage_{time_str}.json"
    token_tracker.save_to_json(file_name)


if __name__ == "__main__":
    asyncio.run(main())
