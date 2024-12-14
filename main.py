from datetime import datetime

from openai import OpenAI

from pb import create_population, init_run, run_for_n
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

tp_set = mutation_prompts[:int(args['num_mutation_prompts'])]
mutator_set= thinking_styles[:int(args['num_thinking_styles'])]

logger.info(f'You are prompt-optimizing for the problem: {args["problem"]}')

logger.info(f'Creating the population...')
p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=args['problem'])

logger.info(f'Generating the initial prompts...')
asyncio.run(init_run(p, client, int(args['num_evals'])))

logger.info(f'Starting the genetic algorithm...')
asyncio.run(run_for_n(n=int(args['simulations']), population=p, model=client, num_evals=int(args['num_evals'])))

print("%"*80)
print("done processing! final gen:")
print(p.units)

token_tracker = get_token_tracker()
print(f"Total cost: ${token_tracker.calculate_estimated_cost():.3f}")
print("*"*50)
token_tracker.print_usage_report()
time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
file_name = f"cost/token_usage_{time_str}.json"
token_tracker.save_to_json(file_name)
