# LLM-as-a-qualitative judge

In [this work](), we present _LLM-as-a-qualitative-judge_, an LLM-based evaluation approach which outputs a _structured report of the common error types_ for a given natural language generation dataset. _LLM-as-a-qualitative-judge_ first analyses each instance individually, and then clusters the discovered error types -- both steps are performed using an LLM. The main goal of _LLM-as-a-qualitative-judge_ is to automate error analysis in natural language generation and provide developers with insights on the weak sides of NLG systems. 

This codebase provides functionality to easily run _LLM-as-a-qualitative-judge_ on your data, as well as code&data for reproducing the experiments in our [paper]().

Citation:
TODO

## Setup
The library uses `ruamel.yaml` to parse evaluation configs:
```
pip3 install ruamel.yaml
```

Depending on which LLM you want to use, you will need the corresponding library, e.g. `pip3 install openai` to use OpenAI models, `pip3 install google-genai` to use Gemini, or `pip3 install torch transformers` to run open-source LLMs using HuggingFace. To use commercial models, you will also need an API key, e.g. `export OPENAI_API_KEY=...` or `export GOOGLE_API_KEY=...`

For running scripts reproducing the paper results, you will also need the following libraries:
```
pip3 install pandas seaborn anthropic
```

## Quick start: running LLM-as-a-qualitative-judge on your data

In python scripts:
```
from llm_as_a_qualitative_judge import generate_issue_report

report = generate_issue_report(user_inputs, labels, generated_responses)
```

Each argument is a list of strings (all of the same length). The function outputs a dictionary with common error types, their descriptions and frequencies. Be default, the script uses OpenAI `gpt4o` model (see below how to change configs to use other models).

In command line: 

_Step 1:_ save you data into a json file with the following schema:
```
[
{
    "user_input": <NLG input 1>,
    "label": <label 1>
    "response": <NLG output 1>
},
{
    "user_input": <NLG input 2>,
    "label": <label 2>
    "response": <NLG output 2>
}
...
]
```

See an example in [data/writing.json](https://github.com/tunde-ajayi/llm-as-a-qualitative-judge/blob/main/data/writing.json)

_Step 2:_
```
python3 llm_as_a_qualitative_judge.py --input_file YOURFILE.json --output_dir OUTPUTDIR
```

__Input data:__ our recommendation is to provide _LLM-as-a-qualitative-judge_ with examples which did not pass quantitative evaluation, e.g. with the main evaluation score lower than a threshold. However, we also included a "No issue" option in the per-analysis prompt in the repo, in case you do not perform such prefiltering. If per-instance analysis outputs the final verdict "No issue", the corresponding example is skipped during the error clustering step.

### More options for LLM-as-a-qualitative-judge

* `analyser_config=` / `--analyser_config=` : a config for per-instance analysis from [configs/analysys](https://github.com/tunde-ajayi/llm-as-a-qualitative-judge/tree/main/configs/analysis) (specifies which LLM / prompt to use). You can add your prompts/LLMs!
* `clustering_config=` / `--clustering_config=` : a config for clustering errors from [configs/report](https://github.com/tunde-ajayi/llm-as-a-qualitative-judge/tree/main/configs/report) (specifies which LLM / prompts to use). For example, you can try both `report_cumulative_gpt4o.yaml` (default) and `report_basic_gpt4o.yaml`.
* `top_n_examples=` / `--top_n_examples=` : if provided, only the first `top_n_examples` will be used in generating the error report
* `additional_task_info=` / field `additional_task_info` in each dictionary in the data json: a string providing more details on the task, e.g. the used evaluation metric, whether ground truth labels represent the only possible answer or just an example of an answer, etc. If provided, this string will be included into the prompt for per-instance analysis.

## Reproducing the experiments in the paper

### Data

As descriped in the [paper](), to evaluate _LLM-as-a-qualitative-judge_, we annotate errors (and their clustering) for ~300 instances from 12 NLG datasets. We provide the [data](https://github.com/tunde-ajayi/llm-as-a-qualitative-judge/blob/main/data/eval_data.zip) in a zip archive protected with a password "judge", to avoid [potential contamination](https://arxiv.org/abs/2310.18018) of LLMs trained on web scrapped data. To unzip the archive, run `cd data; unzip eval_data.zip` and indicate `judge` as password.

### Code

Script `eval_analysis.py` runs the evaluation of per-instance analysis. Example how to run it on all the datasets:
```
DIR=eval
LABEL=gpt4o
CONFIG_GEN=analysis_json_gen_gpt4o_paper.yaml
CONFIG_RAG=analysis_json_rag_gpt4o_paper.yaml
for dataset in flask elitr-bench nlu detox enru gsm8k; do
export OPENAI_API_KEY=; export GOOGLE_API_KEY=; export ANTHROPIC_API_KEY=; python3 run_analysis.py --input_files "data/eval_data/${dataset}.json" --output_dir ${DIR}/${dataset}_${LABEL} --analyser_config=configs/analysis/${CONFIG_GEN} --evaluator_config=configs/evaluation/evaluate_analysis_claude.yaml;
done

for dataset in syllabusqa mkqa_ru bioasq searchqa writing lifestyle; do
python3 run_analysis.py --input_files "data/eval_data/${dataset}.json" --output_dir ${DIR}/${dataset}_${LABEL} --analyser_config=configs/analysis/${CONFIG_RAG} --evaluator_config=configs/evaluation/evaluate_analysis_claude.yaml --instruction_processor bergen_processor;
done
```

The results will be saved in a `${DIR}/${dataset}_${LABEL}` folder. `CONFIG_RAG` differs from `CONFIG_GEN` only by including the retrieved documents into the prompt for per-instance analysis (`--instruction_processor bergen_processor` is also used here to parse the documents from the `llm_instruction` provided in the `eval_data`).

Script `eval_clustering.py` runs the evaluation of the issue clustering. Example how to run it on all the datasets:
```
DIR=eval
LABEL_IN=gpt4o
LABEL_OUT=cum_sum
CONFIG=report_cumulative_gpt4o_paper.yaml
for dataset in lifestyle syllabusqa mkqa_ru bioasq flask elitr-bench nlu detox enru gsm8k searchqa writing; do
python3 run_summary.py --input_dir ${DIR}/${dataset}_${LABEL_IN} --output_subdir ${LABEL_OUT} --clustering_config=configs/report/${CONFIG} --evaluator_config=configs/evaluation/evaluate_sum_labels_claude.yaml; done
```

The results will be saved into the subdirectories named `${LABEL_OUT}` inside each of the directories `${DIR}/${dataset}_${LABEL_IN}`.

You can also experiment with synthetic data, see script `gen_synthetic_clustering_data.py` and configs in `configs/gen_synthetic_data`.

### Archive with our results

We also provide a zip archive [data/experiments_data.zip](https://github.com/tunde-ajayi/llm-as-a-qualitative-judge/blob/main/data/experiments_data.zip) with our results for the 12 datasets. Again, it is protected with password `judge`.

## Repository structure

* Root files:
    * `llm_as_a_qualitative_judge.py`: main script
    * `llm.py`: implements various LLMs (you can add yours!)
    * `prompting_approaches.py`: implements prompting with json / text outputs and they postprocessing, e.g. extracting keys / final answers after a separator.
    * `utils.py`: utils
    * `instruction_processors.py`: these are functions which can be used to extract additional fields from the `llm_instruction` provided in the json data, to be inserted into the prompt for per-instance analysis. In our experiments, we use it to extract retrieved documents.
* `Configs`: configurations for per-instance analysis (`configs/analysis`), error clustering (`configs/report`), evaluation of per-instance analsis or natural language labels in error clustering (`configs/evaluation`) etc. Each configuration specifies an LLM, its settings, and prompt(s).
