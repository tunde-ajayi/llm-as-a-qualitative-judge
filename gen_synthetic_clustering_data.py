import json
import argparse
import os
import pandas as pd
from collections import defaultdict
import random
import numpy as np

from llms import get_generator
import prompting_approaches
import instuction_processors
from utils import normalize

parser = argparse.ArgumentParser(description='Generate synthetic data for summaries')
# experiment & data
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--num_type_errors', type=int, default=8)
parser.add_argument('--num_examples', type=int, default=100)
parser.add_argument('--output_dir', type=str, required=True, help="path where to save results")
parser.add_argument('--overwrite', action='store_true', help="overwrite the existing folder")
parser.add_argument('--generator_config', type=str, default=None, help="config for generator, includes generator configurations / prompt / etc")

args = parser.parse_args()

if os.path.exists(args.output_dir) and not args.overwrite:
    raise ValueError(f"Path {args.output_dir} exists! If you want to overwrite it set --overwrite flag, otherwise provide a new unique outdir")
os.makedirs(args.output_dir, exist_ok=True)

all_errors = [
    "Retrieved documents do not contain the correct answer",
    "Retrieved documents do not contain all pieces of the correct answer",
    "Retrieved documents do not have titles and hence it is unclear what they are about",
    "A wrong answer was generated while the retrieved documents contained the correct answer",
    "Generation was stopped due to the max new tokens limit reached",
    "The generated answer is in French instead of English",
    "User question is ambiguous",
    "User question is unclear",
    "Generated response is in caps lock",
    "Generated response contains grammar errors",
    "Generated response contains code switching between several languages",
    "Match metric did not correctly match the numeric answer and the written answer",
    "There is an error in one of the reasoning steps",
    "An LLM generated extra comments in addition to the requested answer",
    "The meaning of the source sentence was not preserved",
    "The generated response does not follow the requested formal style",
    "An LLM did not follow all the conditions in the user's instruction",
    "An LLM loops in repetative generation",
    "Inproper translation of one or several words",
    "The answer is too long while the user requested a brief answer"
]

random.seed(args.seed)
np.random.seed(args.seed)
classes_distribution = np.random.randint(low=1, high=100, size=args.num_type_errors)
classes_distribution = classes_distribution / classes_distribution.sum()
error_idxs = np.random.choice(np.arange(len(all_errors)), args.num_type_errors, replace=False)
class_per_examle = np.random.choice(error_idxs, args.num_examples, p=classes_distribution)

gen_config = prompting_approaches.load_prompter_config(args.generator_config)
assert "generator_type" in gen_config
assert "generator_model_name" in gen_config

metrics = defaultdict(int)

# analyser
gen = get_generator(gen_config["generator_type"], 
                    gen_config["generator_model_name"], 
                    gen_config["generator_extra_gen_params"] if "generator_extra_gen_params" in gen_config else "{}")

generator = prompting_approaches.SimpleJsonPrompter(gen_config, 
                                                     gen, 
                                                     metrics, 
                                                     role="generator")

error_per_example = {"id"+str(i):{} for i in range(args.num_examples)}
for ki, k in enumerate(error_idxs):
    error_type = all_errors[k]
    num_errors_to_gen = np.sum(class_per_examle==k)
    def prompt_filler(eval_prompt):
        prompt = eval_prompt.replace("EEE", error_type).replace("NNN", str(num_errors_to_gen)) 
        return prompt
    generation, generation_trace = generator.prompt_single(prompt_filler)
    generated_errors = list(generation.values())[:num_errors_to_gen]
    assert all([type(s)==str for s in generated_errors])
    assert len(generated_errors) == num_errors_to_gen
    for i, abs_pos in enumerate(np.where(class_per_examle==k)[0]):
        id_ = "id" + str(abs_pos)
        error_per_example[id_]["analysis"] = generated_errors[i]
        error_per_example[id_]["cluster_idx"] = ki
        error_per_example[id_]["cluster_name"] = error_type

    
with open(f"{args.output_dir}/per_example_results.json", "w") as fout:
    json.dump(error_per_example, fout, indent=4)