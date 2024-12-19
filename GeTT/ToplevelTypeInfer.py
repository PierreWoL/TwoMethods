# Read table entity type from the csv file
# Embed them
# Use hierarchical clustering to achieve the top level types
import os
import pickle

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from Clustering.ClusteringParameter import clustering
from groundTruth import data_classes, evaluate_cluster
from utils import mkdir

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

dataset = "WDC"
number = "1"
csv_et = pd.read_csv(f"Result/{dataset}/entityTypeS2Try{number}.csv")
# print(csv_et.columns)
tables = csv_et["FileName"]
entity_types = list(csv_et["inferred_type"])
# print(tables, "\n", entity_types)
#et_incoding = model.encode(entity_types, convert_to_tensor=True, device=device)
#with open(f"Result/{dataset}/et_incodingS2Try{number}.pkl", 'wb') as file:
 #   pickle.dump(et_incoding, file)
from langchain_community.embeddings import OllamaEmbeddings
ollama_emb =OllamaEmbeddings(model="llama3.1")
r1 = ollama_emb.embed_documents(entity_types)

number_estimate = 10
ground_truth = f"E:/Project/CurrentDataset/datasets/{dataset}/groundTruth.csv"
data_path = f"E:/Project/CurrentDataset/datasets/{dataset}/Test/"
test_path = f"Result/{dataset}/Detail/{number}/"
mkdir(test_path)

table_names = [i[:-4] for i in list(tables) ]
def clustering_results(input_data, tables, dataPath, groundTruth=None, folderName=None,
                       numEstimate=0):  # , graph = None
    number_estimate = numEstimate
    min = number_estimate
    max = 3 * number_estimate
    cluster_dict = clustering(input_data, tables, min, max_clusters=max)
    print(cluster_dict)
    gt_clusters, ground_t, gt_cluster_dict = data_classes(dataPath, groundTruth)
    gt_clusters0, ground_t0, gt_cluster_dict0 = data_classes(dataPath, groundTruth, superclass=False)
    del ground_t0, gt_cluster_dict0
    metrics_value = evaluate_cluster(gt_clusters, gt_cluster_dict, cluster_dict, folderName,
                                     gt_clusters0)  # ,graph = graph

    return cluster_dict, metrics_value


cluster_dict, metric_dict = clustering_results(r1, table_names, data_path, groundTruth=ground_truth,
                                               folderName=test_path,numEstimate=number_estimate)
metric_df = pd.DataFrame([metric_dict])
print(metric_dict)
