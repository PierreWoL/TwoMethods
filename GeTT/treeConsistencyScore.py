import os
import pickle
import networkx as nx
import pandas as pd
from collections import Counter
import re



def get_table_properties_and_descendants(G, node):
    descendants = nx.descendants(G, node)
    descendants.add(node)
    table_properties = []
    for n in descendants:
        table_property = G.nodes[n].get('tables', [])
        table_properties.extend(table_property)
    return table_properties


def findnodeGTTypes(tree, dataset, node, isTop=True):
    names = []
    groundTruth = f"E:/Project/CurrentDataset/datasets/{dataset}/groundTruth.csv"
    csvgt = pd.read_csv(groundTruth)
    tables = get_table_properties_and_descendants(tree, node)
    #print(tables)
    def get_class_by_filename(file_name):
        # Find the row that matches the given fileName and return its class
        result = csvgt[csvgt['fileName'] == file_name]
        if not result.empty:
            if isTop is True:
                return result.iloc[0]['superclass']
            else:
                return result.iloc[0]['class']
        else:
            return None

    for table in tables:
        name = get_class_by_filename(table + ".csv")
        #print(table, name)
        if name is not None:
            names.append(name)
    counter = Counter(names)

    if len(counter)!=0:
        max_count = max(counter.values())
        most_frequent_elements = [element for element, count in counter.items() if count == max_count]
    else:
        #print(counter, names)
        most_frequent_elements =[]
    return most_frequent_elements

def TreeConsistencyScore(tree, dataset):
    overall_path_score = 0
    target_path =f"E:/Project/CurrentDataset/datasets/{dataset}/"
    with open(os.path.join(target_path, "graphGroundTruth.pkl"), "rb") as file:
        G = pickle.load(file)

    all_paths = []
    top_layer = list(tree.successors("Thing"))
    lowest_layer = [node for node in tree.nodes if tree.out_degree(node) == 0]

    for bottom_node in lowest_layer:
        for top_node in top_layer:
            paths = list(nx.all_simple_paths(tree, top_node, bottom_node))
            all_paths.extend(paths)
    for path in all_paths:
        matchedElement_GT = 0
        matchedElement = 0
        types_path = [tree.nodes[i]['label'] for i in path]

        if [] in types_path:
            continue
        print(types_path)
        has_path = False
        intersection = set(types_path[0]).intersection(set(types_path[-1]))
        if intersection:
            matchedElement = len(types_path)
            for index, types in enumerate(types_path):
                if index == 0 or index == len(types_path) - 1:
                    matchedElement_GT += 1

                else:
                    intersection_other = set(types_path[0]).intersection(set(types))
                    if intersection_other:
                        matchedElement_GT += 1
            perConsistencyS = matchedElement_GT / matchedElement

        else:
            possible_paths_elements = set()
            for top_node in types_path[0]:
                import ast
                list_data = ast.literal_eval(top_node)
                top_node = list_data[0]
                #print(top_node)
                #print(top_node)
                #match = re.search(r"'(\w+)'", top_node)

                #if match:
                    #top_node = match.group(1)
                for bottom_node in types_path[-1]:
                    if "[" in bottom_node:
                        bottom_nodes = ast.literal_eval(bottom_node)
                        for bottom_node in bottom_nodes:
                            #print(bottom_node)
                            if nx.has_path(G, top_node, bottom_node):
                                paths = list(nx.all_simple_paths(G, top_node, bottom_node))
                                has_path = True
                                matchedElement = len(types_path)
                                for sublist in paths:
                                    possible_paths_elements.update(sublist)
                            elif nx.has_path(G, bottom_node, top_node):
                                paths = list(nx.all_simple_paths(G, bottom_node, top_node))
                                has_path = True
                                matchedElement = len(types_path)
                                for sublist in paths:
                                    possible_paths_elements.update(sublist)
                    else:
                        print(bottom_node)
                        if nx.has_path(G, top_node, bottom_node):
                            paths = list(nx.all_simple_paths(G, top_node, bottom_node))
                            has_path = True
                            matchedElement = len(types_path)
                            for sublist in paths:
                                possible_paths_elements.update(sublist)
                        elif nx.has_path(G, bottom_node, top_node):
                            paths = list(nx.all_simple_paths(G, bottom_node, top_node))
                            has_path = True
                            matchedElement = len(types_path)
                            for sublist in paths:
                                possible_paths_elements.update(sublist)
            if has_path is False:
                perConsistencyS = 0
                # print(types_path, perConsistencyS)
            else:
                for index, types in enumerate(types_path):
                    if index == 0 or index == len(types_path) - 1:
                        matchedElement_GT += 1

                    else:
                        intersection_other = set(possible_paths_elements).intersection(set(types))
                        if intersection_other:
                            matchedElement_GT += 1
                perConsistencyS = matchedElement_GT / matchedElement

        overall_path_score += perConsistencyS
    overall_path_score = overall_path_score / len(all_paths) if len(all_paths) > 0 else 1
    return overall_path_score, len(all_paths)

dataset = "GDS"
folder = "qwen14"
for i in range(5):
    test_path = f"Result/{dataset}/Detail/COL_Zero/{folder}/{str(i)}/"
    file_path = os.path.join(test_path, "tree.pkl")
    if os.path.exists(file_path) is True:
        print(file_path)
        with open(os.path.join(test_path, "tree.pkl"), "rb") as f:
            tree = pickle.load(f)
        children = list(tree.successors("Thing"))
        leaves = [node for node in tree.nodes if tree.out_degree(node) == 0]
        for child in children:
            tlt = findnodeGTTypes(tree, dataset, child, isTop=True)
            tree.nodes[child]['label'] = tlt
        # print(tlt)
        # 遍历所有节点
        for node in tree.nodes:
            if node not in children:
                llt = findnodeGTTypes(tree, dataset, node, isTop=False)
                tree.nodes[node]['label'] = llt
            # print(llt)
        with open(os.path.join(test_path, "tree.pkl"), "wb") as f:
            pickle.dump(tree, f)


all_scores = []
all_path_length = []
for i in range(5):
    test_path = f"Result/{dataset}/Detail/COL_Zero/{folder}/{str(i)}/"
    file_path = os.path.join(test_path, "tree.pkl")
    if os.path.exists(file_path) is True:
        print(file_path)
        with open(os.path.join(test_path, "tree.pkl"), "rb") as f:
            tree = pickle.load(f)
        path_score, all_paths = TreeConsistencyScore(tree, dataset)
        all_scores.append(path_score)
        all_path_length.append(all_paths)
df = pd.DataFrame({
        'path_score': all_scores,
        'all_pathlength': all_path_length
    })
print(df)
df.to_csv(f"Result/{dataset}/Detail/COL_Zero/{folder}/tcs.csv")

"""

"""
