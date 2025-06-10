import random
import json
import numpy as np

from utils import normalize
from llms import get_generator
import prompting_approaches

def run_per_instance_analysis(analyser_config, data_per_example, metrics, instruction_processor=None):
    """
    Performs per-instance analysis.
    Inputs:
    * analyser_config: a dictionary specifying analyser configuration (see "configs/analysis")
    * data_per_example: a list of dicts, keys: "user_input", "label", "response", 
    (optional) "metric", (optional) "llm_instruction"
    * metrics: an instance of defaultdict(int) where to save errors during analysis
    * instruction_processor: a function which can change a prompt, using "llm_instruction"
    
    The function adds keys "analysis" and "analysis_trace" into each item
    Analysis is a string containing per-instance analysis, 
    analysis trace is a dict containing the results of intermediate steps (if any)

    Returns updated data_per_example and metrics
    """
    generator = get_generator(analyser_config["generator_type"], 
                              analyser_config["generator_model_name"], 
                              analyser_config["generator_extra_gen_params"] \
                              if "generator_extra_gen_params" in analyser_config else "{}"
                             )

    analyser_class = getattr(prompting_approaches, analyser_config["class"])
    analyser = analyser_class(analyser_config, generator, metrics, role="analyser")
    new_data_per_example = []
    for item_id, item in enumerate(data_per_example):
        def prompt_filler(analysis_prompt):
            prompt = analysis_prompt.replace("QQQ", item["user_input"]).\
                                     replace("LLL", str(item["label"])).\
                                     replace("AAA", item["response"])
            if "additional_task_info" in item: # any extra info, e.g. metric info
                prompt = prompt.replace("MMM", item["additional_task_info"])
            else:
                prompt = prompt.replace("MMM", "")
            if instruction_processor is not None:
                prompt = instruction_processor(prompt, item["llm_instruction"])
            return prompt
    
        # perform analysis
        analysis, analysis_trace = analyser.prompt_single(prompt_filler)

        if "no_issue_keyword" in analyser_config and \
            normalize(analysis) == normalize(analyser_config["no_issue_keyword"]):
            continue
    
        item["analysis_trace"] = analysis_trace
        if len(analysis):
            item["analysis"] = analysis
        new_data_per_example.append(item)

    return new_data_per_example, metrics

def clustering_direct_prompting(clustering_config, data_per_example, metrics):
    """
    Performs issue clustering through direct prompting

    Inputs:
    * clustering_config: a dictionary specifying analyser configuration (see "configs/report")
    * data_per_example: a list of dicts, required key in each item: "analysis"
    * metrics: an instance of defaultdict(int) where to save errors during clustering

    Returns summarization_result (a dict containing "report" and "trace") and updated metrics.
    Report is a dictionary containing the generated report, 
    and trace is a dictionary contaning the results of intermediate states. 
    """
    analyses = []
    random.shuffle(data_per_example)
    if len(data_per_example) == 0: return {"report": {}, "trace": {}}, metrics, data_per_example
    for item in data_per_example:
        if "analysis" not in item:
            metrics["no_analysis"] += 1
            continue
        analyses.append(item["analysis"])

    generator = get_generator(clustering_config["generator_type"], 
                              clustering_config["generator_model_name"], 
                              clustering_config["generator_extra_gen_params"] \
                              if "generator_extra_gen_params" in clustering_config else "{}"
                             )
    if "class" in clustering_config:
        _class = getattr(prompting_approaches, clustering_config["class"])
    else:
        _class = prompting_approaches.SimpleJsonPrompter
    summarizer = _class(clustering_config, 
                         generator, 
                         metrics, 
                         role="summarizer")

    def prompt_filler(summary_prompt):
        prompt = summary_prompt.replace("PPP", "\n".join([f"Analysis of {j}-th response: {a}" \
                                                  for j, a in enumerate(analyses)])) 
        return prompt
            
    summary, summarization_trace = summarizer.prompt_single(prompt_filler)

    if type(summary) == str:
        try:
            # nb: ad-hoc postprocessing
            summary = summary.replace("```", "")
            if summary.startswith("json"): summary = summary[4:]
            summary = summary.strip()
            summary = json.loads(summary)
        except:
            metrics["not_a_json"] += 1
            summary = {}
    result = {
        "report": summary,
        "trace": summarization_trace
    }

    return result, metrics, data_per_example

def clustering_cumulative(clustering_config, data_per_example, metrics):
    """
    Performs issue clustering via a cumulative algorithm

    Inputs:
    * clustering_config: a dictionary specifying analyser configuration (see "configs/report")
    * data_per_example: a list of dicts, required key in each item: "analysis"
    * metrics: an instance of defaultdict(int) where to save errors during clustering

    Returns summarization_result (a dict containing "report" and "trace") and updated metrics.
    Report is a dictionary containing the generated report, 
    and trace is a dictionary contaning the results of intermediate states. 
    """
    analyses = []
    random.shuffle(data_per_example)
    for item in data_per_example:
        if "analysis" not in item:
            metrics["no_analysis"] += 1
            continue
        analyses.append(item["analysis"])

    generator = get_generator(clustering_config["generator_type"], 
                              clustering_config["generator_model_name"], 
                              clustering_config["generator_extra_gen_params"] \
                              if "generator_extra_gen_params" in clustering_config else "{}"
                             )

    if "error_classification_class" in clustering_config:
        error_classification_class = getattr(prompting_approaches, clustering_config["error_classification_class"])
    else:
        error_classification_class = prompting_approaches.SimpleJsonPrompter
    error_classifier_config = {key.replace("error_classification_", ""): clustering_config[key]\
                               for key in clustering_config \
                               if "error_classification_" in key and \
                               key != "error_classification_class"}
    error_classifier = error_classification_class(error_classifier_config, 
                                                 generator, 
                                                 metrics, 
                                                 role="error_classifier")
    if "error_type_generation_class" in clustering_config:
        error_type_generation_class = getattr(prompting_approaches, clustering_config["error_type_generation_class"])
    else:
        error_type_generation_class = prompting_approaches.SimpleJsonPrompter
    error_type_generation_config = {key.replace("error_type_generation_", ""): clustering_config[key]\
                               for key in clustering_config \
                               if "error_type_generation_" in key and \
                               key != "error_type_generation_class"}
    error_type_generator = error_type_generation_class(error_type_generation_config, 
                                                     generator, 
                                                     metrics, 
                                                     role="error_type_generator")
    if "regenerate_cluster_names" in clustering_config and clustering_config["regenerate_cluster_names"]:
        if "error_type_regeneration_class" in clustering_config:
            error_type_regeneration_class = getattr(prompting_approaches, clustering_config["error_type_regeneration_class"])
        else:
            error_type_regeneration_class = prompting_approaches.SimpleJsonPrompter
        error_type_regeneration_config = {key.replace("error_type_regeneration_", ""): clustering_config[key]\
                                   for key in clustering_config \
                                   if "error_type_regeneration_" in key and \
                                   key != "error_type_regeneration_class"}
        error_type_regenerator = error_type_regeneration_class(error_type_regeneration_config, 
                                                                 generator, 
                                                                 metrics, 
                                                                 role="error_type_regenerator")
    error_name_prefix = clustering_config["error_name_prefix"]
    error_name_key = clustering_config["error_name_key"]
    error_description_key = clustering_config["error_description_key"]
    error_types = {}
    #np.random.seed(1)
    trace = []
    for j in range(len(analyses)):
        error_types_dict = {}
        if j > 0:
            error_types_ = [ertype["descr_for_future_prompts"] for ertype in error_types.values()]
            np.random.shuffle(error_types_)
            for k, error_type in enumerate(error_types_):
                error_types_dict[error_name_prefix+str(k)] = error_type
                #{"error_name": error_type, "error_description": error_types[error_type][...]
            
        def filler_fn(prompt):
            return prompt.replace("PPP", analyses[j])\
                         .replace("CCC", json.dumps(error_types_dict, indent=4))
        if j > 0:
            error_type_key, _ = error_classifier.prompt_single(filler_fn)
            trace.append({"step": f"error_classification_{j}", "generated": error_type_key})
        if j > 0 and error_type_key.lower().strip() != "none": # assume we chose one of existing errors
            error_type_key = error_type_key.lower().strip()
            if error_type_key not in error_types_dict:
                metrics["error_type_key_not_in_dict"] += 1
                continue
            error_name = error_types_dict[error_type_key]["error_name"]
            error_types[error_name]["count"] += 1
            error_types[error_name]["indexes"].append(j)
        else: # generate a new error type
            chosen_error, _ = error_type_generator.prompt_single(filler_fn)
            trace.append({"step": f"error_type_generation_{j}", "generated": chosen_error})
            if error_name_key in chosen_error:
                error_name = chosen_error[error_name_key]
                error_description = chosen_error[error_description_key]
                error_types[error_name] = chosen_error
                error_types[error_name]["count"] = 1
                error_types[error_name]["indexes"] = [j]
                error_types[error_name]["descr_for_future_prompts"] = {
                                            "error_name": error_name,
                                            "error_description": error_description
                                        }
                                                                
            else:
                metrics["error_name_key_absent_in_generated_error"] += 1
    error_types_ = {}
    # sort issue types by frequency
    for i, (error_name, error_type) in enumerate(sorted(error_types.items(), key=lambda x: x[1]["count"], reverse=True)):
        error_types_[i] = error_type
        if "regenerate_cluster_names" in clustering_config and clustering_config["regenerate_cluster_names"]:
            examples = ["* "+analyses[idx] for idx in error_type["indexes"]]
            def filler_fn(prompt):
                return prompt.replace("PPP", "\n".join(examples))
            regenerated_eror_type, _ = error_type_regenerator.prompt_single(filler_fn)
            error_types_[i][error_name_key] = regenerated_eror_type[error_name_key]
            error_types_[i][error_description_key] = regenerated_eror_type[error_description_key]
            
    clustering_result = {
        "report": error_types_,
        "trace": trace
    }   

    return clustering_result, metrics, data_per_example

def run_filtering(filtering_config, data_per_example, metrics):
    """
    Performs filtering of instances in the dataset, i.e. selects only instances which 
    failed binary LLM-as-a-judge evaluation (judges if the answer is correct or not).
    These instances will then be passed to the per-instance analysis.
    Inputs:
    * filtering_config: a dictionary specifying filterer configuration (see "configs/filtering")
    * data_per_example: a list of dicts, keys: "user_input", "label", "response"
    * metrics: an instance of defaultdict(int) where to save errors during analysis
    
    The function returns a subset of data_per_example and updated metrics
    """
    generator = get_generator(filtering_config["generator_type"], 
                              filtering_config["generator_model_name"], 
                              filtering_config["generator_extra_gen_params"] \
                              if "generator_extra_gen_params" in filtering_config else "{}"
                             )
    accept_keyword = filtering_config["accept_keyword"]
    reject_keyword = filtering_config["reject_keyword"]
    filterer_class = getattr(prompting_approaches, filtering_config["class"])
    filterer = filterer_class(filtering_config, generator, metrics, role="filter")
    new_data_per_example = []
    for item_id, item in enumerate(data_per_example):
        def prompt_filler(analysis_prompt):
            prompt = analysis_prompt.replace("QQQ", item["user_input"]).\
                                     replace("LLL", str(item["label"])).\
                                     replace("AAA", item["response"])
            if "additional_task_info" in item: # any extra info, e.g. metric info
                prompt = prompt.replace("MMM", item["additional_task_info"])
            else:
                prompt = prompt.replace("MMM", "")
            return prompt
    
        # perform analysis
        verdict, _ = filterer.prompt_single(prompt_filler)
        verdict = normalize(verdict)
        if verdict == reject_keyword:
            new_data_per_example.append(item)
        elif verdict == accept_keyword:
            metrics["failed_filtering"] += 1
            # let's not add such examples into the pool

    return new_data_per_example, metrics
