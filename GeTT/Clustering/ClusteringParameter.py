import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def AgglomerativeClustering_param_search(input_data, cluster_num_min, cluster_num_max):
    input_data = np.array(input_data, dtype=np.float32)
    score = -1
    best_model = AgglomerativeClustering()
    # at_least = math.ceil(cluster_num // 4 * 3) + 2
    for n_clusters in range(cluster_num_min, cluster_num_max, 1):  # math.ceil(2.5* cluster_num), 3* cluster_num+10
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        agg_clustering.fit(input_data)
        labels = agg_clustering.labels_
        if score <= silhouette_score(input_data, labels):
            score = silhouette_score(input_data, labels)
            best_model = agg_clustering
    print(best_model.n_clusters, score)
    return best_model, best_model.labels_


def cluster_discovery(parameters, tableNames):
    if not parameters:
        print("parameters are invalid! Check the code.")
        return None
    labels = parameters[1]
    l = (tableNames, labels)
    clust = zip(*l)
    clu = list(clust)
    return clu


def cluster_Dict(clusters_list):
    cluster_dictionary = {}
    for k, v in clusters_list:
        if cluster_dictionary.get(v) is None:
            cluster_dictionary[v] = []
        cluster_dictionary[v].append(k)
    return cluster_dictionary


def clustering(input_et_embedding, table_names, number_estimate, max_clusters=None):
    parameters = AgglomerativeClustering_param_search(input_et_embedding, number_estimate, max_clusters)
    clusters = cluster_discovery(parameters, table_names)
    cluster_dict = cluster_Dict(clusters)
    return cluster_dict
