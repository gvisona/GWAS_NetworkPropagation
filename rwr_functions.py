import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


from tqdm.notebook import tqdm

from sknetwork.ranking import PageRank
from sknetwork.data import from_edge_list

from networkx import pagerank, from_pandas_edgelist


network_files = {
    "BioPlex3": "processed_data/networks/BioPlex3_shared/edges_list_ncbi.csv",
    "HumanNet": "processed_data/networks/HumanNetV3/edges_list_ncbi.csv",
    "PCNet": "processed_data/networks/PCNet/edges_list_ncbi.csv",
    "ProteomeHD": "processed_data/networks/ProteomeHD/edges_list_ncbi.csv",
    "STRING": "processed_data/networks/STRING/edges_list_ncbi.csv",
}


def load_graph(network):
    print("Loading {} graph".format(network))
    df = pd.read_csv(network_files[network])[["node1", "node2"]].astype(str)
    graph = from_edge_list(df.values.astype(str))
    graph["names"] = graph["names"].astype(str)
    return graph

def load_graph_nx(network):
    print("Loading {} graph".format(network))
    df = pd.read_csv(network_files[network], dtype={'node1': str, 'node2': str})[["node1", "node2"]]
    graph = from_pandas_edgelist(df, source="node1", target="node2")
    # graph["names"] = graph["names"].astype(str)
    return graph


def init_rwr_scores_nx(graph, data):
    node2idx = {str(n): i for i, n in enumerate(graph.nodes)}
    # idx2node = {v: k for k, v in node2idx.items()}

    ncbi2gene = dict(zip(data.NCBI_id, data.Gene))
    ncbi_genes = set(data.NCBI_id)
    pegasus_scores = dict(zip(data.NCBI_id, data.Score))
    pagerank_seeds = {}
    for node in graph.nodes: #["names"].astype(str):
        if node in ncbi_genes:
            pagerank_seeds[node] = pegasus_scores[node]
        else:
            pagerank_seeds[node] = 0
    return pagerank_seeds



def load_pegasus_results(disease):
    data = pd.read_csv(
        f"processed_data/gwas_gene_pvals/{disease}/filtered_ncbi_PEGASUS_{disease}_gwas_data.csv")
    data = data[~data["NCBI_id"].isna()]
    data["NCBI_id"] = data["NCBI_id"].astype(str)
    # data[data["NCBI_id"].isin(graph["names"])].sort_values(by="Pvalue")
    # min_pval = data["Pvalue"][data["Pvalue"]>0].min()
    # print(min_pval)
    pegasus_scores = {}
    for i, row in data.iterrows():
        # pv = np.maximum(min_pval, np.minimum(1, row["Pvalue"]))
        pv = row["Pvalue"] if row["Pvalue"]>0.0 else row["Error"]
        pegasus_scores[row["NCBI_id"]] = np.maximum(1e-16, -np.log10(pv))
    # pegasus_ncbi_genes = set(pegasus_scores.keys())
    data["Score"] = data["NCBI_id"].map(pegasus_scores)
    return data

def init_rwr_scores(graph, data):
    node2idx = {str(n): i for i, n in enumerate(graph.names)}
    # idx2node = {v: k for k, v in node2idx.items()}

    ncbi2gene = dict(zip(data.NCBI_id, data.Gene))
    ncbi_genes = set(data.NCBI_id)
    pegasus_scores = dict(zip(data.NCBI_id, data.Score))
    pagerank_seeds = {}
    for node in graph["names"].astype(str):
        if node in ncbi_genes:
            pagerank_seeds[node2idx[node]] = pegasus_scores[node]
        else:
            pagerank_seeds[node2idx[node]] = 0
    return pagerank_seeds

def perform_rwr(alpha, graph, seeds):   
    pagerank = PageRank(damping_factor=alpha, n_iter=20)
    rwr_scores = pagerank.fit_transform(graph.adjacency, seeds)
    return rwr_scores

def perform_rwr_nx(alpha, graph, seeds):   
    rwr_scores = pagerank(graph, alpha=alpha, personalization=seeds)

    return rwr_scores


def process_rwr_results(scores, graph, data, seeds):
    node2idx = {str(n): i for i, n in enumerate(graph.names)}
    idx2node = {v: k for k, v in node2idx.items()}
    ncbi2gene = dict(zip(data.NCBI_id, data.Gene))
    nonzero_genes = [idx2node[g] for g in seeds.keys() if seeds[g]>0]
    
    seeds_vals = np.fromiter(seeds.values(), dtype="float")
    max_val = np.max(seeds_vals[~np.isinf(seeds_vals)])
    rwr_results = []
    for i, node in enumerate(graph["names"]):
        row = {}

        row["Idx"] = node2idx[node]
        row["Gene NCBI ID"] = node
        row["Symbol"] = ncbi2gene[node] if node in ncbi2gene.keys() else "-"

        init_score = seeds[node2idx[node]]
        if np.isinf(init_score):
            init_score = max_val+1

        row["Initial Score"] = init_score
        row["Final Score"] = scores[i]

        rwr_results.append(row)

    rwr_results = pd.DataFrame(rwr_results).sort_values(by="Final Score", ascending=False)
    return rwr_results



def process_rwr_results_nx(scores, graph, data, seeds):
    node2idx = {str(n): i for i, n in enumerate(graph.nodes)}
    idx2node = {v: k for k, v in node2idx.items()}
    ncbi2gene = dict(zip(data.NCBI_id, data.Gene))
    
    seeds_vals = np.fromiter(seeds.values(), dtype="float")
    max_val = np.max(seeds_vals[~np.isinf(seeds_vals)])
    rwr_results = []
    for i, node in enumerate(graph.nodes):
        row = {}

        row["Idx"] = node2idx[node]
        row["Gene NCBI ID"] = node
        row["Symbol"] = ncbi2gene[node] if node in ncbi2gene.keys() else "-"

        init_score = seeds[node]
        if np.isinf(init_score):
            init_score = max_val

        row["Initial Score"] = init_score
        row["Final Score"] = scores[node]

        rwr_results.append(row)

    rwr_results = pd.DataFrame(rwr_results).sort_values(by="Final Score", ascending=False)
    return rwr_results

###############################################################################


def precision_at_k(targets, predictions, K=None):
    if K is not None:
        predictions = predictions[:K]
        denom = K
    else:
        denom = len(predictions)
    num = len(set(targets).intersection(set(predictions)))
    return num/denom
        
def recall_at_k(targets, predictions, K=None):
    if K is not None:
        predictions = predictions[:K]
    num = len(set(targets).intersection(set(predictions)))
    return num/len(targets)
        
def average_precision_at_k(targets, predictions, K=None):
    pak = []
    for pk in range(1, K+1):
        if predictions[pk-1] not in targets:
            continue
        pak.append(precision_at_k(targets, predictions, pk))
    if len(pak)<1:
        return 0.0
    return np.mean(pak)


def calc_apk(y_true, y_pred, k_max=0):

    # Check if all elements in lists are unique
    if len(set(y_true)) != len(y_true):
        raise ValueError("Values in y_true are not unique")

    if len(set(y_pred)) != len(y_pred):
        raise ValueError("Values in y_pred are not unique")

    if k_max != 0:
        y_pred = y_pred[:k_max]


    correct_predictions = 0
    running_sum = 0

    for i, yp_item in enumerate(y_pred):
    
        k = i+1 # our rank starts at 1
    
        if yp_item in y_true:
            correct_predictions += 1
            running_sum += correct_predictions/k

    if correct_predictions==0:
        return 0.0
    return running_sum/correct_predictions



def load_seeds_and_targets(disease):
    # with open("processed_data/gene_seeds/{}_seeds_gene2ncbi.json".format(disease), "r") as f:
    #     disease_seeds_gene2ncbi = json.load(f)
    # gene_seeds_ncbi = [str(n) for n in disease_seeds_gene2ncbi.values()]
    with open("processed_data/gene_seeds/{}_ncbi_seeds.json".format(disease), "r") as f:
        disease_seeds_ncbi = json.load(f)
    gene_seeds_ncbi = [str(n) for n in disease_seeds_ncbi]
    # disease_seeds_gene2ncbi

    # Load targets
    with open("processed_data/gwas_catalog_targets/{}_targets_gene2ncbi.json".format(disease), "r") as f:
        catalog_targets_gene2ncbi = json.load(f)
    ncbi_targets = list(catalog_targets_gene2ncbi.values())
    return gene_seeds_ncbi, ncbi_targets

def calculate_metrics(rwr_res, K, gene_seeds, targets, net_name, alpha, disease, scoring="Score"):
    results = []
    genes = [str(s) for s in rwr_res["Gene NCBI ID"] if str(s) not in gene_seeds] #data[col].astype(str)
    # pak = precision_at_k(targets, genes, K)
    # apk = average_precision_at_k(targets, genes, K)
    apk = calc_apk(targets, genes, k_max=K)
    # results.append({"Network": net_name, "Alpha": alpha, "Metric": "Precision", "K": K, "Value": pak, "Method": scoring, "Disease": disease})
    results.append({"Network": net_name, "Alpha": alpha, "Metric": "Average Precision", "K": K, "Value": apk, "Method": scoring, "Disease": disease})

    results = pd.DataFrame(results)
    return results