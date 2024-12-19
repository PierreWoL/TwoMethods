import os

import pandas as pd

from evaluation import bertScoreMetric, calculate_cosine_similarity
from utils import mkdir

dataset = "GDS"
store_path = f"Result/{dataset}/Column/"
mkdir(store_path)
number = "1"
# result.to_csv(os.path.join(store_path, "SampleColumnAnnotation3.csv"))
df = pd.read_csv(f"E:\Project\CurrentDataset\datasets\{dataset}\column_gt.csv", encoding="latin1")
result = pd.read_csv(os.path.join(store_path, f"SampleColumnAnnotation{number}.csv"))
data_names = []
col_names = []
dataTypes = []
groundTruthTs = []

data_names_NE = []
col_names_NE = []
dataTypes_NE = []
dataTypes_NE_G = []
groundTruth_NEs = []

for index, row in result.iterrows():
    tableName = row["table_name"]
    colName = row["col_name"]
    dataType = row["inferred type"]
    dataType_NE = row["named entity re-inferred type"]
    filtered_df = df[(df['fileName'] == tableName) & (df['colName'] == colName)]
    gt_type_list = list(filtered_df["ColumnLabel"])
    if len(gt_type_list) != 0:
        gt_type = gt_type_list[0]
        data_names.append(tableName)
        col_names.append(colName)
        dataTypes.append(dataType)
        groundTruthTs.append(gt_type)
        if pd.isna(dataType_NE) is False:  # np.isnan(dataType_NE)
            data_names_NE.append(tableName)
            col_names_NE.append(colName)
            dataTypes_NE.append(dataType_NE)
            groundTruth_NEs.append(gt_type)
            dataTypes_NE_G.append(dataType)
    else:
        continue

print(len(data_names), len(col_names), len(groundTruthTs), len(dataTypes), len(groundTruthTs))
print(len(data_names_NE), len(col_names_NE), len(groundTruth_NEs), len(dataTypes_NE), len(groundTruth_NEs))
"""eBERT_score = bertScoreMetric(dataTypes, groundTruthTs)
eSBERT_score = calculate_cosine_similarity(dataTypes, groundTruthTs)

All_types = pd.DataFrame(
    {"Table": data_names, "Column": col_names, "Inferred Types": dataTypes,
     "bert score Precision": eBERT_score['precision'],
     "bert score Recall": eBERT_score['recall'],
     "bert score F1": eBERT_score['f1'], "SBERT score": eSBERT_score})"""
evaluationBERT_score_NE = bertScoreMetric(dataTypes_NE, groundTruth_NEs)
evaluationSBERT_score_NE = calculate_cosine_similarity(dataTypes_NE, groundTruth_NEs)
evaluationBERT_score_NE_G = bertScoreMetric(dataTypes_NE_G, groundTruth_NEs)
evaluationSBERT_score_NE_G = calculate_cosine_similarity(dataTypes_NE_G, groundTruth_NEs)
NE_types = pd.DataFrame(
    {"Table": data_names_NE, "Column": col_names_NE, "Inferred Types": dataTypes_NE,
     "bert score Precision": evaluationBERT_score_NE['precision'],
     "bert score Recall": evaluationBERT_score_NE['recall'],
     "bert score F1": evaluationBERT_score_NE['f1'],
     "SBERT score": evaluationSBERT_score_NE,
     "General bert score Precision": evaluationBERT_score_NE_G['precision'],
     "General bert score Recall": evaluationBERT_score_NE_G['recall'],
     "General bert score F1": evaluationBERT_score_NE_G['f1'],
     "General SBERT score": evaluationSBERT_score_NE_G},
)
#All_types.to_csv(os.path.join(store_path, f"SampleColumnAnnotationScore{number}.csv"))
NE_types.to_csv(os.path.join(store_path, f"NESampleColumnAnnotationScore{number}.csv"))
