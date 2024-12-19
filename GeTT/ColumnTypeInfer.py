import os
import pickle

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from Clustering.ClusteringParameter import clustering
from groundTruth import column_gts, metric_Spee, rand_Index_custom
from utils import mkdir, most_frequent


def evaluate_col_cluster(gtclusters, gtclusters_dict, clusterDict: dict, folder="", filename=""):
    clusters_label = {}
    column_label_index = []
    false_ones = []
    gt_column_label = []
    columns = []
    columns_ref = []
    for index, column_list in clusterDict.items():
        labels = []
        for column in column_list:
            if column in gtclusters.keys():
                columns.append(column)
                label = gtclusters[column]
                if type(label) is list:
                    for item_label in label:
                        labels.append(item_label)
                    gt_column_label.append(label)
                else:
                    gt_column_label.append(gtclusters_dict[label])
                    labels.append(label)

        if len(labels) == 0:
            continue
        else:
            cluster_label = most_frequent(labels)
        clusters_label[index] = cluster_label
        false_cols = []
        for column in column_list:
            if column in gtclusters.keys():
                column_label_index.append(gtclusters_dict[cluster_label])
                if gtclusters[column] != cluster_label:
                    false_cols.append(column)
                    false_ones.append(column)
        columns_ref.append([column_list, cluster_label, false_cols])
    print(gt_column_label)
    if type(gt_column_label[0]) is not list:
        metric_dict = metric_Spee(gt_column_label, column_label_index)
    else:
        metric_dict = {"random Index": rand_Index_custom(gt_column_label, column_label_index)}
    metric_dict["purity"] = 1 - len(false_ones) / len(column_label_index)
    if folder != "" and filename != "":
        if columns:
            df_cols = pd.DataFrame(columns_ref, columns=['resultCols', 'result label', 'false_cols'])
            df_cols.to_csv(os.path.join(folder, filename + 'cols_results.csv'), encoding='utf-8', index=False)
    return metric_dict


def clusteringColumnResults(input_data, columns, gt_clusters, gt_cluster_dict, folderName=None,
                            filename=None):
    number_estimate = len(gt_cluster_dict) // 2
    min = number_estimate
    max = 2 * number_estimate
    cluster_dict = clustering(input_data, columns, min, max_clusters=max)
    metrics_value = evaluate_col_cluster(gt_clusters, gt_cluster_dict, cluster_dict, folderName, filename)
    return cluster_dict, metrics_value


def colCluster(index, clu, content, Zs, Ts, gt_clusters, gt_cluster_dict, dataset="", trial=""):
    Ts[clu] = []
    Zs[clu] = []
    for vector in content:
        Ts[clu].append(vector[0])
        Zs[clu].append(vector[1])
    Zs[clu] = np.array(Zs[clu]).astype(np.float32)
    fileName = ""
    if dataset != "":
        col_example_path = f"Result/{dataset}/Column/Details/{trial}/"
        mkdir(col_example_path)
        fileName = f"{index}"
    else:
        col_example_path = ""
    cluster_dict, metric_dict = clusteringColumnResults(Zs[clu], Ts[clu], gt_clusters[clu],
                                                        gt_cluster_dict[clu],
                                                        folderName=col_example_path,
                                                        filename=fileName)
    print(metric_dict)
    return metric_dict


def conceptualAttri(content, dataset):
    target_path = f"Result/{dataset}/Column/{dataset}/DictGT/"
    mkdir(target_path)
    gt_clusters, ground_t, gt_cluster_dict = column_gts(dataset)
    with open(os.path.join(target_path, '_gt_cluster.pickle'),
              'wb') as handle:
        pickle.dump(list(gt_cluster_dict.keys()), handle, protocol=pickle.HIGHEST_PROTOCOL)
    Zs = {}
    Ts = {}
    for index, clu in enumerate(list(gt_cluster_dict.keys())):
        print(index, clu)
        colCluster(index, clu, content, Zs, Ts, gt_clusters, gt_cluster_dict)


ds = "WDC"
result = pd.read_csv("E:\SILLM\Result\WDC\Column\ColumnAnnotationAll1.csv")
cols = []
infer1 = []
infer2 = []

for index, row in result.iterrows():
    cols.append(row["table_name"][:-4] + "." + str(row["col_name"]))
    infer1.append(row["inferred type"].split(", "))
    infer2.append(row["named entity re-inferred type"].split(", "))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
"""model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
infer1_encoding = [model.encode(sublist, convert_to_tensor=True, device=device) for sublist in infer1]
infer2_encoding = [model.encode(sublist, convert_to_tensor=True, device=device) for sublist in infer2]
content1 = [(cols[i], torch.mean(infer1_encoding[i], 0)) for i in range(len(infer1))]
content2 = [(cols[i],  torch.mean(infer2_encoding[i], 0)) for i in range(len(infer2))]"""
with open(f"Result/WDC/Column/COL_incodingS1.pkl", 'rb') as file:
    content1 = pickle.load(file)
with open(f"Result/WDC/Column/COL_incodingS2.pkl", 'rb') as file:
    content2 = pickle.load(file)
# print(cols,"\n" ,infer1,"\n" ,infer2)

#print(content1[0], content2[0])
conceptualAttri(content1, ds)
