import os
import pickle

import networkx as nx
import pandas as pd
import json

from figure_graph import draw_interactive_graph
from groundTruth import data_classes, evaluate_cluster
from utils import mkdir
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

dataset = "GDS"
model ="qwen2.5"
fold = "qwen14"
with open(
        f"results/taxo_ChainofLayers_filter_zero/{dataset}/{model}/{fold}/10shots/model_response.json") as f:
    result = json.load(f)
#print(len(result), )
"""for result_i in result:
    print(result_i)
    for try_i in result_i:
        print(try_i)
    break"""
model = SentenceTransformer('all-MiniLM-L6-v2')


def find_most_similar(type, types):
    embeddings_given = model.encode([type])
    embeddings_list = model.encode(types)
    cosine_similarities = cosine_similarity(embeddings_given, embeddings_list)
    most_similar_index = cosine_similarities.argmax()
    most_similar_element = types[most_similar_index]
    return most_similar_element


def find_descendants(G, node):
    descendants = set()

    # 使用DFS递归遍历所有后代节点
    def dfs(n):
        # 遍历所有子节点
        for child in G.successors(n):
            if child not in descendants:
                descendants.add(child)
                dfs(child)

    # 从给定节点开始遍历
    dfs(node)
    return descendants


def detect_parent_child_relationship(nodes):
    """
    Detect parent-child relationships in a list of nodes.
    Args:
    - nodes (dict): A dictionary where keys are node names and values are their corresponding paths
    Returns:
    - List of tuples representing parent-child relationships.
    """
    max_layer = 0
    relationships = []
    for i, parent_node_dict in enumerate(nodes):
        parent_name = parent_node_dict["name"]
        parent_path = parent_node_dict["order"]
        if len(parent_path) > max_layer:
            max_layer = len(parent_path)
            print(parent_path)
        for child_node_dict in nodes[i:]:
            child_name = child_node_dict["name"]
            child_path = child_node_dict["order"]
            if len(child_path) == len(parent_path) + 1 and child_path[:len(parent_path)] == parent_path:
                # print(parent_path, child_path)
                # print((parent_name, child_name))
                relationships.append((parent_name, child_name))
            if len(child_path) > max_layer:
                max_layer = len(child_path)
                print(child_path)
    return max_layer, relationships


all = []
for i in range(0, 5):  # range(len(result))
    try:
        print(i)
        result_i = result[i]
        # print(result_i)
        tree = nx.DiGraph()
        lines = result_i[0].split('\n')
        cleaned_data = [line.lstrip() for line in lines]
        processed = []
        for data in cleaned_data:
            element = data.split(" ", 1)
            order = element[0].split(".")
            order = [io for io in order if io != '']
            # print(data, element)
            try:
                processed.append({'order': order, 'name': element[1]})
            except:
                continue
        # print(processed)
        max_layer, relationships = detect_parent_child_relationship(processed)

        for relationship in relationships:
            tree.add_edge(relationship[0], relationship[1])
        # print(len(processed))
        # print(tree)
        # print("max_layer", max_layer, len(tree.nodes()))
        data = pd.read_csv(f"E:\SILLM\Result\{dataset}\{fold}\entityTypeS1Try{str(i)}.csv")
        types = list(data["inferred_type"].dropna().unique())

        mapping_data = {}
        excluded_data = []
        for index, row in data.iterrows():
            mapping_data[row["FileName"]] = row["inferred_type"]
            # print(mapping_data["FileName"], row["inferred_type"])

        count = 0
        # for nod in tree.nodes():
        # gt_type = find_most_similar(nod,types)
        # print(nod, gt_type)
        # print(len(set(list(data["inferred_type"]))))
        for nod in set(list(data["inferred_type"])):
            if nod not in tree.nodes:
                node_exclude = [i for i in mapping_data.keys() if mapping_data[i] == nod]
                excluded_data.extend(node_exclude)
                # print(nod)
                count += 1
        print(f"total nodes: {len(tree.nodes)} ", count, "types not in the tree", len(set(excluded_data)))
        children = list(tree.successors("Thing"))
        # print("successors",len(children))
        top_level_children = {}
        for index, child in enumerate(children):
            child_nodes = find_descendants(tree, child)
            child_nodes.add(child)
            # child_nodes = [child]
            tables = []
            for child_node in child_nodes:
                # print(len(mapping_data.values()),child_node )
                node_tables = [i[:-4] for i in mapping_data.keys() if mapping_data[i] == child_node]
                # print(child_node, " node include tables", node_tables)
                tree.nodes[child_node]['tables'] = node_tables
                tables.extend(node_tables)
            if len(tables) == 0:
                continue
            top_level_children[index] = tables
    except:
        continue



    groundTruth = f"datasets/{dataset}/groundTruth.csv"
    dataPath = f"datasets/{dataset}/Test/"
    test_path = f"Result/{dataset}/Detail/COL_Zero/{fold}/{str(i)}/"
    mkdir(test_path)
    with open(os.path.join(test_path, "tree.pkl"),"wb") as f:
        pickle.dump(tree, f)
    #draw_interactive_graph(tree, os.path.join(test_path, "tree.html"))
    gt_clusters, ground_t, gt_cluster_dict = data_classes(dataPath, groundTruth)
    gt_clusters0, ground_t0, gt_cluster_dict0 = data_classes(dataPath, groundTruth, superclass=False)
    del ground_t0, gt_cluster_dict0
    metrics_value = evaluate_cluster(gt_clusters, gt_cluster_dict, top_level_children, test_path,
                                     gt_clusters0)  # ,graph = graph
    all.append(metrics_value)

metric_df = pd.DataFrame(all)
print(metric_df)
mkdir(f"Result/{dataset}/Detail/COL_Zero/{fold}/")
metric_df.to_csv(f"Result/{dataset}/Detail/COL_Zero/{fold}/overall.csv", index=False)
