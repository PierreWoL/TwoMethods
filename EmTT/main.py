import argparse
import pickle

import pandas as pd

from TableCluster.tableClustering import P1, P2, P3, baselineP1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WDC")
    parser.add_argument("--embed", type=str, default='sbert')
    parser.add_argument("--baseline", dest="baseline", action="store_true")
    parser.add_argument("--clustering", type=str, default='Agglomerative')
    parser.add_argument("--iteration", type=int, default=1)

    """ This is for the testing of different steps """
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--phaseTest",dest="phaseTest", action="store_true")
    """ This is the parameter for P1"""
    parser.add_argument("--estimateNumber", type=int, default=10)
    """ Parameter for slicing the dendrogram in Step3 """
    parser.add_argument("--intervalSlice", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.15)
    """ Step 4 parameter """
    parser.add_argument("--similarity", type=float, default=0.9)
    parser.add_argument("--portion", type=float, default=0.8)
    parser.add_argument("--portionSA", type=float, default=0.05)
    parser.add_argument("--Euclidean", dest="Euclidean", action="store_true")
    """ single-column mode without table context """
    parser.add_argument("--subjectCol", dest="subjectCol", action="store_true")
    # row / column-ordered for preprocessing
    """Parameter for End To End """
    parser.add_argument("--P1Embed", type=str, default='WDC_sbert_53num50hTT_subcol.pkl')
    #cl_SCT8_lm_sbert_head_column_0_none_subCol.pkl
    parser.add_argument("--P23Embed", type=str, default='WDC_sbert_30248HeaderTT_column.pkl')
    parser.add_argument("--P4Embed", type=str, default='WDC_sbert_30248HeaderTT_column.pkl')
    # parser.add_argument("--slice_start", type=int, default=0)
    # parser.add_argument("--slice_stop", type=int, default=1)
    """ This is randomly selecting tables from the datasets """
    parser.add_argument("--SelectType", type=str, default='')
    """ This is for testing running time, the total number of tables """
    parser.add_argument("--tableNumber", type=int, default=0)

    hp = parser.parse_args()
    if hp.step == 1:
        P1(hp) if hp.baseline is False else baselineP1(hp)
    elif hp.step == 2:
        P2(hp)
    elif hp.step == 3:
        P3(hp)



'''
"Background: The cluster of tables is derived from a hierarchical clustering algorithm applied to the embeddings of subject attributes of tables."
"This clustering indicates a mutual conceptual entity type among the tables.",

"Data Provided: The cluster includes all tables in the cluster."
"Each subject attribute is separated by <sc></sc>."
"Each subject attribute follows the format: <sc> Column name : instance1, instance2, instance3,....</sc>.",

"Task: Analyze the provided subject attributes within the cluster. "
"Based on the content and pattern of the subject attributes, determine the mutual entity type of the tables. Provide a suitable name that describes this mutual entity type.",

"Expected Output: The output should be a single word or a short phrase that clearly describes the mutual "
"entity type of the tables."
"The name should be descriptive and relevant to the context of the data. AND NO OTHER TEXT!"'''