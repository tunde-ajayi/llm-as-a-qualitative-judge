import json
import argparse
import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict

from llms import get_generator
import prompting_approaches
import instuction_processors
from utils import normalize
from steps import clustering_direct_prompting, clustering_cumulative

from matplotlib import pyplot as plt
import seaborn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score, rand_score

parser = argparse.ArgumentParser(description='Clustering of issue types')
# experiment & data
parser.add_argument('--input_dir', required=True, help='paths to a directory = result of eval_analysis.py')
parser.add_argument('--output_subdir', type=str, required=True, help="subdirectory where to save results")
parser.add_argument('--overwrite', action='store_true', help="overwrite the existing subfolder")
parser.add_argument('--top_n_examples', type=int, default=None, help="only run summarization on top_n examples")
# analyser
parser.add_argument('--regenerate_cluster_names', action='store_true', help="generate new cluster names after clustering")
parser.add_argument('--clustering_config', type=str, default=None, help="config for clustering, includes method / generator configurations / prompt / etc")
parser.add_argument('--evaluator_config', type=str, default=None, help="config for analysis evaluator, includes generator configurations / prompt / etc")
parser.add_argument('--evaluation_accept_keywords', nargs="+", default=["yes", "true", "1"], help='which words to use as positive verdict in evaluation')
parser.add_argument('--evaluation_reject_keywords', nargs="+", default=["no", "false", "0"], help='which words to use as negative verdict in evaluation')

args = parser.parse_args()

if not os.path.exists(args.input_dir):
    raise ValueError(f"Input dir does not exist! {args.input_dir}")

output_dir = os.path.join(args.input_dir, args.output_subdir)
if os.path.exists(output_dir) and not args.overwrite:
    raise ValueError(f"Path {output_dir} exists! If you want to overwrite it set --overwrite flag, otherwise provide a new unique outdir")
os.makedirs(output_dir, exist_ok=True)

clustering_config = prompting_approaches.load_prompter_config(args.clustering_config)
assert "generator_type" in clustering_config
assert "generator_model_name" in clustering_config
assert "method" in clustering_config
assert "error_name_key" in clustering_config

# save config
with open(f"{output_dir}/clustering_config.json", "w") as fout:
    json.dump(vars(args), fout, indent=4)

# load and prepare data
data_per_example = json.load(open(os.path.join(args.input_dir, "per_instance_results.json")))
if args.top_n_examples is not None:
    data_per_example = data_per_example[:args.top_n_examples]

def save_data(data, filename_prefix):
    with open(f"{output_dir}/{filename_prefix}.json", "w") as fout:
        json.dump(data, fout, indent=4)

metrics = defaultdict(int)

# report generation (clustering issues)

if clustering_config["method"] == "simple":
    clustering_result, metrics, items = clustering_direct_prompting(clustering_config, data_per_example, metrics)
elif clustering_config["method"] == "cumulative":
    clustering_result, metrics, items = clustering_cumulative(clustering_config, 
                                                       data_per_example, 
                                                       metrics, 
                                                       args.regenerate_cluster_names
                                                      )
else:
    raise ValueError(f"Clustering method {clustering_config["method"]} is not implemented")

save_data(clustering_result, "report")

### Evaluate report, if labels are provided
sum_labels_ = []
sum_js = []
label2errorname = {}
j = 0
for item_id, item in items:
    if "analysis" in item:
        if "cluster_idx" in item:
            sum_labels_.append(item["cluster_idx"])
            sum_js.append(j)
            if "cluster_name" in item and item["cluster_name"] is not None:
                label2errorname[item["cluster_idx"]] = item["short_cluster_name"] if "short_cluster_name" in item else item["cluster_name"]
        j += 1 # j indexes only items with analysis, see above
# renumber gt clusters according to frequency
sum_js = np.array(sum_js)
freqs = pd.Series(sum_labels_).value_counts()
old2new = {old:new for new, old in enumerate(freqs.index)}
sum_labels_ = [old2new[k] for k in sum_labels_]
label2errorname = {old2new[k]: label2errorname[k] for k in label2errorname}
error_name_key = clustering_config["error_name_key"]
if len(sum_labels_):
    # Part 1: eval cluster assignments
    sum_labels = []
    sum_classes = []
    class2label = {}
    for error_class, error_type_item in enumerate(clustering_result["report"].values()):
        class2label[error_class] = error_type_item[error_name_key]
        for j in error_type_item["indexes"]:
            if j in sum_js:
                sum_classes.append(error_class)
                sum_labels.append(sum_labels_[np.where(sum_js==j)[0][0]])
    sum_labels = np.array(sum_labels)
    sum_classes = np.array(sum_classes)

    # Part 1.1: metrics
    metrics["num_examples_in_eval"] = len(sum_classes)
    metrics["adjusted_rand_score"] = adjusted_rand_score(sum_labels, sum_classes)
    metrics["rand_score"] = rand_score(sum_labels, sum_classes)
        
    with open(f"{output_dir}/clustering_metrics.json", "w") as fout:
        json.dump(metrics, fout, indent=4)

    # Part 1.2: plot confusion matrix
    conf_matrix = confusion_matrix(sum_labels, sum_classes)[:sum_labels.max()+1, :sum_classes.max()+1]
    free_cols = np.ones(conf_matrix.shape[1], dtype=bool)
    new_conf_matrix = np.zeros(conf_matrix.shape)
    xlabels = []
    for j in range(min(conf_matrix.shape[0], conf_matrix.shape[1])):
        for_argmax = conf_matrix[j] + 0
        for_argmax[~free_cols] = -10000
        chosen_col = np.argmax(for_argmax)
        xlabels.append(class2label[chosen_col] if chosen_col in class2label else "")
        new_conf_matrix[:, j] = conf_matrix[:, chosen_col]
        free_cols[chosen_col] = False
    if conf_matrix.shape[0] < conf_matrix.shape[1]:
        new_conf_matrix[:, conf_matrix.shape[0]:] = conf_matrix[:, free_cols]
        xlabels += [(class2label[k] if k in class2label else "") for k in np.where(free_cols)[0]]
    conf_matrix = new_conf_matrix
    plt.figure(figsize=(4, 4))
    ax = seaborn.heatmap(new_conf_matrix, square=True, cmap="Blues", annot=True, cbar=None)
    plt.ylabel("Ground-truth labels", fontsize=12)
    #plt.xlabel("Clusters")
    plt.title("Clusters", fontsize=12)
    ys = [label2errorname[k] if k in label2errorname else "" for k in range(new_conf_matrix.shape[0])]
    for i, label in enumerate(ys):
        size = 10 #max(6, 12 - 0.3 * max(0, len(label) - 10))
        if len(label) > 30:
            yws = label.split()
            mis = len(yws) // 2
            label = "\n".join([" ".join(yws[:mis]), " ".join(yws[mis:])])
        ax.text(conf_matrix.shape[1] + 0.2, i + 0.5, "-"+label, va='center', ha='left', fontsize=size)

    xs = []
    for x in xlabels:
        if len(x) > 30:
            xws = x.split()
            mis = len(xws) // 2
            xx = "\n".join([" ".join(xws[:mis]), " ".join(xws[mis:])])
            xs.append("--"+xx)
        else:
            xs.append("--"+x)
    ax.set_xticks([i + 0.5 for i in range(len(xs))])
    ax.set_xticklabels(xs, rotation=90, ha='center', fontsize=10)
    
    plt.savefig(f"{output_dir}/confusion_matrix.pdf", bbox_inches="tight")

    # Part 2: evaluate summarization labels
    if args.evaluator_config is not None:
        error_name_key = "error_description"
        class2errorname = {error_class:error_name_key for error_class, error_type_item in \
                           enumerate(clustering_result["report"].values())}
        
        evaluator_config = prompting_approaches.load_prompter_config(args.evaluator_config)
        generator = get_generator(evaluator_config["generator_type"], 
                                  evaluator_config["generator_model_name"], 
                                  evaluator_config["generator_extra_gen_params"] \
                                  if "generator_extra_gen_params" in evaluator_config else {}
                                 )
        # evaluation approach and keywords
        evaluator_class = getattr(prompting_approaches, evaluator_config["class"])
        evaluator = evaluator_class(evaluator_config, generator, metrics, role="evaluator")
        
        accept_keywords = [normalize(keyword) for keyword in args.evaluation_accept_keywords]
        reject_keywords = [normalize(keyword) for keyword in args.evaluation_reject_keywords]
        
        metrics["diagonal_label_yes"] = 0
        metrics["diagonal_label_no"] = 0
        metrics["diagonal_label_unclear"] = 0
        
        for x_error_name, y_error_name in zip(xlabels, ys):
            # todo: change to issue type descriptions (this is what is done in the paper)
            # (currently short cluster names are evaluated)
            def prompt_filler(eval_prompt):
                prompt = eval_prompt.replace("E2", y_error_name)
                prompt = prompt.replace("E1", x_error_name) 
                return prompt
            
            evaluation, evaluation_trace = evaluator.prompt_single(prompt_filler)
            
            if len(evaluation) > 0:
                verdict = normalize(evaluation)
                if verdict in accept_keywords:
                    metrics["diagonal_label_yes"] += 1
                elif verdict in reject_keywords:
                    metrics["diagonal_label_no"] += 1
                else:
                    metrics["diagonal_label_unclear"] += 1
        
        denominator = metrics["diagonal_label_yes"] + metrics["diagonal_label_no"]
        if denominator > 0:
            metrics["diagonal_label_accuracy"] = metrics["diagonal_label_yes"] / denominator
        else:
            metrics["diagonal_label_accuracy"] = 0
        
        with open(f"{output_dir}/clustering_metrics.json", "w") as fout:
            json.dump(metrics, fout, indent=4)