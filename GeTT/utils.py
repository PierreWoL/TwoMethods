import os
from collections import Counter

import pandas as pd


def dataframe_to_sample(df: pd.DataFrame, selectedHeader = None) -> str:
    header_transfer = list(df.columns)
    if selectedHeader is not None:
        header_transfer = [f'**{selectedHeader}**' if item == selectedHeader else item for item in header_transfer]
    markdown_str = "| " + " | ".join(header_transfer) + " |\n"
    markdown_str += "| " + " | ".join(["-" * len(col) for col in df.columns]) + " |\n"
    for index, row in df.iterrows():
        markdown_str += "| " + " | ".join(str(value) for value in row) + " |\n"
    return markdown_str


def dataframe_to_mdd(df: pd.DataFrame, selectedHeader=None, max_tokens=15, keep_token= 5) -> str:
    header_transfer = list(df.columns)
    if selectedHeader is not None:
        header_transfer = [f'**{selectedHeader}**' if item == selectedHeader else item for item in header_transfer]
    markdown_str = "| " + " | ".join(header_transfer) + " |\n"
    markdown_str += "| " + " | ".join(["-" * len(col) for col in df.columns]) + " |\n"
    for index, row in df.iterrows():
        row_str = "| "
        for col, value in row.items():
            if isinstance(value, str) and len(value.split()) > max_tokens:
                value = " ".join(value.split()[:keep_token]) + "..."
            row_str += str(value) + " | "
        markdown_str += row_str.rstrip() + "\n"
    return markdown_str

def sample(df: pd.DataFrame, max_tokens=50, keep_token= 50):
    if len(df) <= 5:
        new_df = df.copy()
    else:
        new_df = df.sample(n=5, random_state=42)
    for index, row in new_df.iterrows():
        for col, value in row.items():
            if isinstance(value, str) and len(value.split()) > max_tokens:
                value = " ".join(value.split()[:keep_token]) + "..."
                new_df.loc[index, col] = value
    return new_df
def most_frequent(list1, isFirst=True):
    """
    count the most frequent occurring annotated label in the cluster
    """

    count = Counter(list1)
    if isFirst is True:
        return count.most_common(1)[0][0]
    else:
        most_common_elements = count.most_common()
        max_frequency = most_common_elements[0][1]
        most_common_elements_list = [element for element, frequency in most_common_elements if
                                     frequency == max_frequency]
        return most_common_elements_list
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
