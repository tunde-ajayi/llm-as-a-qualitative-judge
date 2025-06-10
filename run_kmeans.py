import json
import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
from utils import normalize as my_normalize

from matplotlib import pyplot as plt
import seaborn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score, rand_score

parser = argparse.ArgumentParser(description='Open-ended error analysis')
# experiment & data
parser.add_argument('--input_dir', required=True, help='paths to a directory = result of run_analysis.py')
parser.add_argument('--output_subdir', type=str, required=True, help="subdirectory where to save results")
parser.add_argument('--overwrite', action='store_true', help="overwrite the existing folder")
parser.add_argument('--top_n_examples', type=int, default=None, help="only run summarization on top_n examples")
parser.add_argument('--model_name', type=str, default=None, help="encoder model name e.g. BERT")
#parser.add_argument('--num_clusters', type=int, default=None, help="only run summarization on top_n examples")
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

# save config
with open(f"{output_dir}/summary_config.json", "w") as fout:
    json.dump(vars(args), fout, indent=4)

# load and prepare data
data_per_example = json.load(open(os.path.join(args.input_dir, "per_example_results.json")))
if args.top_n_examples is not None:
    data_per_example = data_per_example[:args.top_n_examples]
num_clusters = -1
for item in data_per_example.values():
    if item["cluster_idx"] + 1 > num_clusters:
        num_clusters = item["cluster_idx"] + 1

def save_data(data, filename_prefix):
    with open(f"{output_dir}/{filename_prefix}.json", "w") as fout:
        json.dump(data, fout, indent=4)

metrics = defaultdict(int)

# summarization
analyses = []
for item_id, item in data_per_example.items():
    if "analysis" not in item:
        metrics["no_analysis"] += 1
        continue
    analyses.append(item["analysis"])

tokenizer = BertTokenizer.from_pretrained(args.model_name)
model = BertModel.from_pretrained(args.model_name)
model.eval()

def get_bert_embeddings(sentences):
    embeddings = []
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embedding.squeeze().numpy())
    return np.vstack(embeddings)

embeddings = get_bert_embeddings(analyses)
normalized_embeddings = normalize(embeddings)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(normalized_embeddings)

summarization_result = {"summary": {c:{"indexes":[], "cluster_name": "cl"+str(c)} for c in range(num_clusters)}}
for i, c in enumerate(kmeans.labels_):
    summarization_result["summary"][c]["indexes"].append(i)

save_data(summarization_result, "summary")

### Evaluate summary, if labels are provided
sum_labels_ = []
sum_js = []
label2errorname = {}
j = 0
for item_id, item in data_per_example.items():
    if "analysis" in item:
        if "cluster_idx" in item:
            sum_labels_.append(item["cluster_idx"])
            sum_js.append(j)
            if "cluster_name" in item and item["cluster_name"] is not None:
                label2errorname[item["cluster_idx"]] = item["short_cluster_name"]
        j += 1 # j indexes only items with analysis, see above
# renumber gt clusters according to frequency
sum_js = np.array(sum_js)
freqs = pd.Series(sum_labels_).value_counts()
old2new = {old:new for new, old in enumerate(freqs.index)}
sum_labels_ = [old2new[k] for k in sum_labels_]
label2errorname = {old2new[k]: label2errorname[k] for k in label2errorname}
error_name_key = "cluster_name"
if len(sum_labels_):
    # Part 1: eval cluster assignments
    sum_labels = []
    sum_classes = []
    class2label = {}
    for error_class, error_type_item in enumerate(summarization_result["summary"].values()):
        class2label[error_class] = error_type_item[error_name_key]
        for j in error_type_item["indexes"]:
            if j in sum_js:
                sum_classes.append(error_class)
                sum_labels.append(sum_labels_[np.where(sum_js==j)[0][0]])
    sum_labels = np.array(sum_labels)
    sum_classes = np.array(sum_classes)

    #print(sum_classes, sum_labels)

    # Part 1.1: metrics
    metrics["num_examples_in_eval"] = len(sum_classes)
    metrics["adjusted_rand_score"] = adjusted_rand_score(sum_labels, sum_classes)
    metrics["rand_score"] = rand_score(sum_labels, sum_classes)
        
    with open(f"{output_dir}/summary_metrics.json", "w") as fout:
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
