import random

import pandas as pd
import os
Dataset_name = "WDC"
data_path = f"E:/Project/datasets/{Dataset_name}/"
sample_names = pd.read_csv(f"sample_selection_{Dataset_name}.csv")
print(sample_names["Table_name"])
"""
###The following is to 
gt_csv_name = "groundTruth.csv"
gt_csv = pd.read_csv(os.path.join(data_path, gt_csv_name))
print(gt_csv)
class_to_filenames = gt_csv.groupby('class')['fileName'].apply(set).to_dict()
random_file_per_class = {key: random.choice(list(value)) for key, value in class_to_filenames.items()}
result_df = pd.DataFrame(list(random_file_per_class.items()), columns=['class', 'Table_name'])
result_df.to_csv(f"sample_selection_{Dataset_name}.csv")
"""
