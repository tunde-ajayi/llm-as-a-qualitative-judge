from abc import ABC, abstractmethod
import pandas as pd
from ruamel.yaml import YAML

def load_prompter_config(config_fn):
    yaml = YAML()
    with open(config_fn, "r") as file:
        config = yaml.load(file)
    if "/" in config_fn:
        config_basedir = config_fn[:config_fn.rfind("/")]
    else:
        config_basedir = ""
    for key, value in config.items():
        if "prompt" in key:
            config[key] = open(f"{config_basedir}/{value}").read()
    return config

class Prompter(ABC):
    config_params = {}
    
    def __init__(self, config, generator, error_counter, role):
        self.config = config # dict with all the parameters and prompts
        self.error_counter = error_counter # defaultdict(int)
        self.role = role # str, to be used in error_counter's keys
        self.generator = generator # could be instantiated here, but we want to be able to pass it between prompters as well. but config for generator is inside "config"
        for key, typ in self.config_params.items():
            assert key in self.config, f"Key {key} not in prompter config"
            expected_type = type(self.config[key])
            assert expected_type == typ, f"Wrong type for key {key} in prompter config: {typ} != {expected_type}"
        
    @abstractmethod
    def prompt_single(self, filler_fun):
        # returns result (str) and trace (dict)
        pass

class SimpleTextPrompter(Prompter):
    """
    Single-step prompting, which outputs text.
    """
    
    config_params = {
                     "prompt": str
                     }
    
    def prompt_single(self, filler_fun):
        filled_prompt = filler_fun(self.config["prompt"])
        result, status = self.generator.generate_single(filled_prompt)
        return result, {"generated": result}

class FinalSeparatorTextPrompter(Prompter):
    """
    Single-step prompting, which generates text, 
    and an output is selected as everything after a separator in the generated text.
    """
    
    config_params = {
                     "prompt": str,
                     "final_separator": str
                     }
    
    def prompt_single(self, filler_fun):
        filled_prompt = filler_fun(self.config["prompt"])
        gen, status = self.generator.generate_single(filled_prompt)
        fin_sep = self.config["final_separator"]
        if fin_sep not in gen:
            self.error_counter[f"{self.role}_no_final_sep"] += 1
            return "", {"generated": gen}
        result = gen[gen.find(fin_sep)+len(fin_sep):]
        return result, {"generated": gen}

class MiddleSeparatorTextPrompter(Prompter):
    """
    Single-step prompting, which outputs a text, 
    and a final output is composed by splitting the generated txt by a separator,
    and composing a json with 2 keys/value pairs.
    """
    
    config_params = {
                     "prompt": str,
                     "separator": str,
                     #"keys": list, # a list with two strings = two keys
                     }
    
    def prompt_single(self, filler_fun):
        filled_prompt = filler_fun(self.config["prompt"])
        gen, status = self.generator.generate_single(filled_prompt)
        sep = self.config["separator"]
        if sep not in gen:
            self.error_counter[f"{self.role}_no_sep"] += 1
            return "", {"generated": gen}
        result1, result2 = gen[:gen.find(sep)], gen[gen.find(sep)+len(sep):]
        key1, key2 = self.config["keys"][0], self.config["keys"][1]
        return {key1: result1, key2: result2}, {"generated": gen}
    
class AbstractSimpleJsonPrompter(Prompter):
    """
    an abstract class implementing a function for single-step prompting
    and retrieving output_key from a generated json

    Asssumes self.generator is generating jsons! 
    
    config fields:
    * prompt: str
    * output_key: str
    """
    def one_prompt_call(self, filled_prompt, output_key, call_name=""):
        response_json, status = self.generator.generate_single(filled_prompt)
        if not status:
            self.error_counter[f"{self.role}{call_name}_fail"] += 1
            return "", response_json
        if output_key not in response_json:
            self.error_counter[f"{self.role}{call_name}_output_key_absent"] += 1
            return "", response_json
        result = response_json[output_key]
        return result, response_json
    
class SimpleJsonPrompter(AbstractSimpleJsonPrompter):
    """
    Single-step prompting, which outputs a json, 
    and an output_key is selected from this json.
    
    Asssumes self.generator is generating jsons! 
    """
    
    config_params = {
                     "prompt": str, 
                     "output_key": str
                     }
    
    def prompt_single(self, filler_fun):
        filled_prompt = filler_fun(self.config["prompt"])
        return self.one_prompt_call(filled_prompt,
                                    self.config["output_key"])
    
class AbstractEnsembleJsonPrompter(AbstractSimpleJsonPrompter):
    """
    an abstract class for ensemble prompters
    
    Executes the same prompt num_tries times, 
    each time extracts output_key from the resulting json
    and outputs the results form all steps
    
    config fields:
    * prompt: str
    * output_key: str
    * num_tries: int
    """
    def ensemble(self, filled_prompt, output_key, num_tries, call_name=""):
        results = []
        trace = {}
        for i in range(num_tries):
            result, response_json = self.one_prompt_call(filled_prompt,
                                                         output_key,
                                                         call_name)
            trace[f"iter_{i}"] = response_json
            if len(result):
                results.append(result)
        return results, trace
    
class VotingEnsembleJsonPrompter(AbstractEnsembleJsonPrompter):
    """
    Executes the same prompt num_tries times, 
    each time extracts output_key from the resulting json, 
    and then selects the most frequent answer.
    
    Asssumes self.generator is generating jsons! 
    """
    
    config_params = {
                     "prompt": str, 
                     "output_key": str, 
                     "num_tries": int
                     }
    
    def prompt_single(self, filler_fun):
        filled_prompt = filler_fun(self.config["prompt"])
        results, trace = self.ensemble(filled_prompt, 
                                       self.config["output_key"], 
                                       self.config["num_tries"])
        if len(results) == 0:
            return "", trace
        result = pd.Series(results).value_counts().index[0]
        return result, trace
    
class DoubleStepEnsembleJsonPrompter(AbstractEnsembleJsonPrompter):
    """
    Executes the same (first) prompt num_tries times, 
    each time extracts output_key from the resulting json. 
    After that, executes the second prompt, 
    which is intended to aggregate previous results into a single final result
    
    Asssumes self.generator is generating jsons! 
    Assumes prompt_aggregate contains substring RESRESRES (where to insert previous results)
    """
    
    config_params = {
                     "prompt": str, 
                     "output_key": str, 
                     "num_tries": int,
                     "prompt_aggregate": str
                    }
    
    def prompt_single(self, filler_fun):
        filled_prompt = filler_fun(self.config["prompt"])
        results, trace = self.ensemble(filled_prompt, 
                                       self.config["output_key"], 
                                       self.config["num_tries"])
        formated_results = "\n".join([f"Output {i}: {res}" for i, res\
                                      in enumerate(results)])
        filled_prompt_aggregate_ = filler_fun(self.config["prompt_aggregate"])
        filled_prompt_aggregate = filled_prompt_aggregate_.\
                                  replace("RESRESRES", formated_results)
        result, response_json = self.one_prompt_call(filled_prompt_aggregate,
                                                     self.config["output_key"],
                                                     "_aggregate")
        trace["aggregation"] = response_json
        return result, trace