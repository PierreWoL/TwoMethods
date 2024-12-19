import os
import pickle
import pandas as pd
from langchain_ollama import OllamaLLM
from prompt.columnPrompt import column_annotationLlama
from utils import mkdir

ollama_base_url = "http://localhost:11434/"
llm = OllamaLLM(
    base_url=ollama_base_url,
    model='llama3.1'
)
dataset = "GDS"


data_path = f"datasets/{dataset}/Test/"
#sample_names_csv = pd.read_csv(f"sample_selection_{dataset}.csv")
#sample_names = sample_names_csv["Table_name"]
sample_names = [i for i in os.listdir(data_path) if i.endswith(".csv")]
entityTypes = []
result = pd.DataFrame(columns=['table_name', 'col_name', 'data type', 'inferred type', 'named entity re-inferred type'])
print(f"table_name | col_name | data type | column inferred type | re-inferred type")
error_table_cols = []
for data_name in sample_names:
    t = pd.read_csv(os.path.join(data_path, data_name))
    for col_name in t.columns:
        try:
            dt, ct, NE_v = column_annotationLlama(t, col_name, llm, table_mention=False, size=5)
            result = result._append({'table_name': data_name,
                                     'col_name': col_name,
                                     'data type': dt,
                                     'inferred type': ct,
                                     'named entity re-inferred type': NE_v}, ignore_index=True)
            print(f"{data_name} |{col_name} | {dt} | {ct} |{NE_v}")
        except:
            error_table_cols.append((data_name, col_name))

with open(f'Result/{dataset}/errors.pkl', 'wb') as file:
    pickle.dump(error_table_cols, file)


store_path = f"Result/{dataset}/Column/"
mkdir(store_path)
number = "1"
result.to_csv(os.path.join(store_path, f"ColumnAnnotationAll{number}.csv"))
