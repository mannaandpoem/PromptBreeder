import asyncio
import random
import re
import os

from pb.llm_client import check_tokens
from pb.pb_types import Population, EvolutionUnit
from typing import List, Optional
# from sentence_transformers import SentenceTransformer, util
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles
from pb import hotpotqa, drop, aime
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from rich import print

load_dotenv()

# hotpotqa_examples = hotpotqa.read_jsonl('pb/data/hotpotqa_validate.jsonl')
# drop_examples = drop.read_jsonl('pb/data/drop_validate.jsonl')
aime_examples = aime.read_jsonl('pb/data/aime_validate.jsonl')

# Initialize OpenAI client
base_url= "https://oneapi.deepwisdom.ai/v1"  # or forward url / other llm url
api_key= "sk-itOqZJVK9kQlVJ8kCbCa026154Bc431fAc0a726616E9B614"

# client = OpenAI(api_key=api_key, base_url=base_url)
client = AsyncOpenAI(api_key=api_key, base_url=base_url)
# need below for estimation_distribution_mutation, not currently using.
# model = SentenceTransformer('multi-qa-distilbert-cos-v1')
# print(model)


@check_tokens()
async def generate(prompt: str, model: str, temperature: float) -> str:
    """Helper function to generate text using OpenAI API asynchronously"""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


async def parallel_generate(prompt_list: List[str], model: str = "gpt-4", temperature: float = 0.7) -> List[str]:
    """
    并行执行多个提示的生成任务

    Args:
        prompt_list: 提示词列表
        model: 模型名称
        temperature: 温度参数

    Returns:
        生成结果列表
    """

    async def generate_one(prompt: str) -> str:
        try:
            return await generate(prompt, model, temperature=temperature)
        except Exception as e:
            print(f"生成失败 - 提示词: {prompt[:50]}... 错误: {str(e)}")
            return ""

    # 使用 asyncio.gather 并行执行所有生成任务
    results = await asyncio.gather(
        *(generate_one(prompt) for prompt in prompt_list),
        return_exceptions=False
    )

    return [r for r in results if r is not None]  # 过滤掉失败的结果


# Direct Mutation mutators
async def zero_order_prompt_gen(unit: EvolutionUnit, problem_description: str, **kwargs) -> EvolutionUnit:
    """Generates a new task-prompt P by concatenating the problem description D with the prompt
    'a list of 100 hints:'. New task-prompt P is the first generated hint.

    Returns:
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    # Get the generated text from the response
    prompt = problem_description + " An ordered list of 100 hints: "
    result = await generate(prompt,"gpt-4o", temperature=0.7)

    # search for the pattern "anything after 1. and before 2."
    pattern = r"1\.(.*?)2\."
    match = re.search(pattern, result, re.DOTALL)
    if match:
        # return the first match
        unit.P = match.group(1).strip()
    else: 
        unit.P = ""

    return unit

async def first_order_prompt_gen(unit: EvolutionUnit, **kwargs) -> EvolutionUnit:
    """Concatenate the mutation prompt M to the parent task-prompt P and pass it to the LLM to produce P'
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    unit.P = await generate(unit.M + " " + unit.P, "gpt-4o", temperature=0.7)
    return unit
    
# Estimation of Distribution Mutation - there is a variation of this called EDA rank
# and index mutation. I didn't implement it.
async def estimation_distribution_mutation(unit: EvolutionUnit, population_units: List[EvolutionUnit], **kwargs) -> EvolutionUnit:
    """ Provide a filtered and numbered list of the current population of task-prompts to the LLM and ask it to continue this list with new task-prompts.
    The List is filtered via ensuring that no two task-prompts have a score of >0.95 via BERT embedding cosine similarities.
    The List is randomly ordered.  

    NOTE: I am confused by this one. Does this mutate the entire population? What values of the continued list from the LLM do I use as prompts? randomly sampled?
    Not going to implement this one yet. Maybe should email someone. 
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    pass
async def lineage_based_mutation(unit: EvolutionUnit, elites: List[EvolutionUnit], **kwargs) -> EvolutionUnit:
    """Using the stored history of best units, provide the LLM this list in chronological order to produce a novel prompt as continuation.
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    HEADING = "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY \n "
    # made a choice not to format it with newlines, could change later.
    ITEMS = "\n".join(["{}. {}".format(i+1, x.P) for i, x in enumerate(elites)])
    unit.P = await generate(HEADING + ITEMS, "gpt-4o", temperature=0.7)
    
    return unit

# Hypermutation
async def zero_order_hypermutation(unit: EvolutionUnit, problem_description: str, **kwargs) -> EvolutionUnit:
    """ Concatenate the original problem_description to a randomly sampled thinking-style and feed it to the LLM to generate a new mutation-prompt.
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    RANDOM_THINKING_STYLE = random.sample(thinking_styles, 1)[0]
    unit.M = await generate(problem_description + " " + RANDOM_THINKING_STYLE, "gpt-4o", temperature=0.7)
    return unit

async def first_order_hypermutation(unit: EvolutionUnit, **kwargs) -> EvolutionUnit:
    """ Concatenate the hyper-mutation prompt "Please summarize and improve the following instruction:"
    to a mutation-prompt to that the LLM generates a new mutation-prompt. This new mutation-prompt is then 
    instantly applied to the task-prompt of that unit.

    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    HYPER_MUTATION_PROMPT="Please summarize and improve the following instruction: "
    unit.M = await generate(HYPER_MUTATION_PROMPT + unit.M, "gpt-4o", temperature=0.7)
    unit.P = await generate(unit.M + " " + unit.P, "gpt-4o", temperature=0.7)
    return unit 


# Lamarckian Mutation
async def working_out_task_prompt(unit: EvolutionUnit, **kwargs) -> EvolutionUnit:
    """ A 'lamarckian' mutation operator similar to instruction induction in APE.

    As far as I can understand, give it both the Q and A from the gsm8k dataset, 
    concatenated between 'I gave a friend an instruction and some advice. Here
    are the correct examples of his workings out ' and 'The instruction was: '
    The idea is to let the LLM reverse-engineer the task-prompt.

    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    # RANDOM_WORKING_OUT = random.sample(hotpotqa_examples, 1)[0]
    # RANDOM_WORKING_OUT = random.sample(drop_examples, 1)[0]
    RANDOM_WORKING_OUT = random.sample(aime_examples, 1)[0]
    unit.P = await generate("I gave a friend an instruction and some advice. Here are the correct examples of his workings out " + RANDOM_WORKING_OUT['question'] +" " +  RANDOM_WORKING_OUT['answer'] + " The instruction was: ", "gpt-4o", temperature=0.7)
    return unit

# Prompt crossover and context shuffling. These happen AFTER mutation operators. 
def prompt_crossover(**kwargs):
    """
    After a mutation operator is applied, 

    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
def context_shuffling(**kwargs):
    """
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """

# omitting the estimation_distribution_mutation
MUTATORS = [
    zero_order_prompt_gen,
    first_order_prompt_gen,
    #estimation_distribution_mutation,
    lineage_based_mutation,
    zero_order_hypermutation,
    first_order_hypermutation,
    working_out_task_prompt
]

POST_MUTATORS = [
    prompt_crossover,
    context_shuffling
]

async def mutate(population: Population) -> Population:
    """Select and apply a random mutator"""
    # steps
    # 1. parse through the population, grouping each evo unit by 2
    # 2. for each pair of evo units, using a uniform distribution, select a random mutator (of the 9)
    # 3. mutate and populate population.units

    # make index pairs
    indices = [i for i in range(len(population.units))]
    random.shuffle(indices)
    pairs = [indices[2*x:2*x+2] for x in range(len(indices) // 2)]

    mutation_tasks = []
    # binary tourmanent genetic algorithm
    for i in range(len(pairs)):

        first_unit = population.units[pairs[i][0]]
        second_unit = population.units[pairs[i][1]]

        print("%"*77)
        print("First unit: \n")
        print(first_unit)
        print("%"*77)
        print("Second unit: \n")
        print(second_unit)

        # determine which unit has the higher fitness. Since I am currently testing and want to preserve the # of calls I am making to the LLM, there 
        # is a decent chance that I will hit equal fitness levels. in that case, first unit wins and second unit loses.

        # TODO: clean this up
        FIRST_WON = False
        if first_unit.fitness >=  second_unit.fitness:
            # loser gets mutated.
            FIRST_WON = True
            mutation_input = second_unit
        else:
            mutation_input = first_unit

        data = {
            'unit' : mutation_input,
            'elites' : population.elites,
            'problem_description': population.problem_description,
        }

        # uniformly pick and call a random mutation operator on the losing unit
        random_mutator = random.sample(MUTATORS, 1)[0]
        print(f"MUTATING: {mutation_input} with {random_mutator.__name__}")

        # 添加到任务列表
        mutation_tasks.append(random_mutator(**data))

    # 等待所有突变完成
    await asyncio.gather(*mutation_tasks)

    return population