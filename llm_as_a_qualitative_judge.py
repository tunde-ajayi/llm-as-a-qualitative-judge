import json
import argparse
import os
from collections import defaultdict

import prompting_approaches
import instruction_processors
from steps import run_per_instance_analysis, clustering_direct_prompting, clustering_cumulative, run_filtering

PATH = os.path.dirname(os.path.abspath(__file__))

def generate_issue_report_for_json(data,
                                   analyser_config="analysis_json_gen_gpt4o.yaml", 
                                   clustering_config="report_cumulative_gpt4o.yaml",
                                   instruction_processor=None,
                                   top_n_examples=None,
                                   debug=False):
    """
    Inputs:
    * data: a list of dictionaries, each dictionary having keys "user_input", "label", "response", 
    (optional) "additional_task_info", (optional) "llm_instruction"
    * analyser_config (if not specified, default is used): one of the configs from "configs/analysis"
    * clustering_config (if not specified, default is used): one of the configs from "configs/report"
    * instruction_processor (optional): None or one of the function names from instruction_processors.py
      (additional postprocessing of prompt for per-instance analysis, e.g. inserting retrieved docs for RAG)
    * top_n_examples (optional, int): if not None, evaluate only on first top_n examples
    * debug: binary. If True, functions returns report and debug info (otherwise only report)

    Returns an error report (a dictionary)
    """
    if top_n_examples is not None: data = data[:top_n_examples]
    metrics = defaultdict(int)
    analyser_config = prompting_approaches.load_prompter_config(f"{PATH}/configs/analysis/{analyser_config}")
    clustering_config = prompting_approaches.load_prompter_config(f"{PATH}/configs/report/{clustering_config}")
    data_per_example, metrics = run_per_instance_analysis(analyser_config, 
                                                          data, 
                                                          metrics, 
                                                          instruction_processor)
    if clustering_config["method"] == "simple":
        clustering_result, metrics, items = clustering_direct_prompting(clustering_config, 
                                                                        data_per_example, 
                                                                        metrics,
                                                                       )
    elif clustering_config["method"] == "cumulative":
        clustering_result, metrics, items = clustering_cumulative(clustering_config, 
                                                                   data_per_example, 
                                                                   metrics,
                                                                  )
    report = clustering_result["report"]
    if type(report) == list:
        for i in range(len(report)):
            if "descr_for_future_prompts" in report[i]:
                report[i].pop("descr_for_future_prompts")
    else: # dict
        for key in report:
            if "descr_for_future_prompts" in report[key]:
                report[key].pop("descr_for_future_prompts")

    if not debug:
        return report
    else:
        return report, {"per_instance_analysis": data_per_example, 
                        "clustering_debug": clustering_result["trace"]}


def generate_issue_report(user_inputs, labels, generated_responses,
                          llm_instructions=None,
                          additional_task_info=None, 
                          analyser_config="analysis_json_gen_gpt4o.yaml", 
                          clustering_config="report_cumulative_gpt4o.yaml",
                          instruction_processor=None,
                          top_n_examples=None,
                          debug=False,
                         ):
    """
    Inputs:
   
    * user_inputs: List[str] (required), user requests to an LLM (should include e.g. task explanation)
    * labels: List[str] (required), ground-truth labels
    * generated_responses: List[str] (required), responses to be evaluated
    * llm_instructions: List[str] (optional), instructions passed to an LLM e.g. inclusing retrieved documents
    * additional_task_info: str (optional), additional info on the task to be inserted into per-analysis prompt
                                            (e.g. what is the task metric)
    * analyser_config (if not specified, default is used): one of the configs from "configs/analysis"
    * clustering_config (if not specified, default is used): one of the configs from "configs/report"
    * instruction_processor (optional): None or one of the function names from instruction_processors.py
      (additional postprocessing of prompt for per-instance analysis, e.g. inserting retrieved docs for RAG)
    * top_n (optional, int): if not None, evaluate only on first top_n examples
    * debug: binary. If True, functions returns report and debug info (otherwise only report)

    Returns an error report (a dictionary)
    """
    data = [
            {
            "user_input": u, 
            "label": gt, 
            "response": r
            } 
            for u, gt, r in zip(user_inputs, labels, generated_responses)
           ]
    if llm_instructions is not None:
        for i, inst in enumerate(llm_instructions):
            data[i]["llm_instruction"] = inst
    if additional_task_info is not None:
        for i in range(len(data)):
            data[i]["additional_task_info"] = additional_task_info
    return generate_issue_report_for_json(data,
                                          analyser_config, 
                                          clustering_config,
                                          instruction_processor,
                                          top_n_examples,
                                          debug)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Open-ended issue analysis')
    # experiment & data
    parser.add_argument('--input_file', type=str, required=True, help='path to json file')
    parser.add_argument('--output_dir', type=str, required=True, help="directory where to save results")
    parser.add_argument('--overwrite', action='store_true', help="overwrite the existing folder")
    parser.add_argument('--top_n_examples', type=int, default=None, help="only run evaluation on top_n examples")
    parser.add_argument('--instruction_processor', type=str, default=None, help="name of a processor of a user instruction which inserts additional fields into an eval prompt for per-example analysis")
    # analyser
    parser.add_argument('--analyser_config', type=str, default=None, help="config for analyser, includes generator configurations / prompt / etc")
    parser.add_argument('--clustering_config', type=str, default=None, help="config for clustering, includes method / generator configurations / prompt / etc")
    
    args = parser.parse_args()
    
    if os.path.exists(args.output_dir) and not args.overwrite:
        raise ValueError(f"Path {args.output_dir} exists! If you want to overwrite it set --overwrite flag, otherwise provide a new unique outdir")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # save config
    with open(f"{args.output_dir}/config.json", "w") as fout:
        json.dump(vars(args), fout, indent=4)
    
    # load and prepare data
    data_per_example = json.load(open(args.input_file))
    if args.top_n_examples is not None:
        data_per_example = data_per_example[:args.top_n_examples]
    
    # instruction processor (parses additional arguments for the analysis prompt, from the llm instruction)
    if args.instruction_processor is not None:
        instruction_processor = getattr(instruction_processors, args.instruction_processor)
    else:
        instruction_processor = None
    
    metrics = defaultdict(int)
    
    report, debug = generate_issue_report_for_json(data_per_example, 
                                                   analyser_config=args.analyser_config, 
                                                   clustering_config=args.clustering_config,
                                                   instruction_processor=args.instruction_processor,
                                                   debug=True,
                                                   )
    def save_data(data, filename_prefix):
        with open(f"{args.output_dir}/{filename_prefix}.json", "w") as fout:
            json.dump(data, fout, indent=4)
    save_data(report, "report")
    save_data(debug, "intermediate_outputs")