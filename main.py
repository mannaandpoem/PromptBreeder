import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from pb import create_population, init_run, run_for_n, Population
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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Run the PromptBreeder Algorithm. Number of units is mp * ts.')
parser.add_argument('-mp', '--num_mutation_prompts', default=2)     
parser.add_argument('-ts', '--num_thinking_styles', default=4)     
parser.add_argument('-e', '--num_evals', default=10)     
parser.add_argument('-n', '--simulations', default=10)     
parser.add_argument('-p', '--problem', default="Solve the math word problem, giving your answer as an arabic numeral.")       

args = vars(parser.parse_args())

total_evaluations = int(args['num_mutation_prompts']) * int(args['num_thinking_styles']) * int(args['num_evals'])
# 修改后的代码

# set num_workers to total_evaluations so we always have a thread 
base_url= "https://oneapi.deepwisdom.ai/v1"  # or forward url / other llm url
api_key= "sk-itOqZJVK9kQlVJ8kCbCa026154Bc431fAc0a726616E9B614"

client = OpenAI(api_key=api_key, base_url=base_url)


def save_population_units(population: Population, base_dir: str = "population_results") -> str:
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
    await init_run(p, client, int(args['num_evals']))

    # Run genetic algorithm
    logger.info('Starting the genetic algorithm...')
    await run_for_n(
        n=int(args['simulations']),
        population=p,
        model=client,
        num_evals=int(args['num_evals'])
    )

    # Print results and save statistics
    print("%" * 80)
    print("done processing! final gen:")
    print(p.units)
    saved_path = save_population_units(p)
    print(f"Results saved to: {saved_path}")

    token_tracker = get_token_tracker()
    print(f"Total cost: ${token_tracker.calculate_estimated_cost():.3f}")
    print("*" * 50)
    token_tracker.print_usage_report()

    # Save token usage data
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"cost/token_usage_{time_str}.json"
    token_tracker.save_to_json(file_name)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)