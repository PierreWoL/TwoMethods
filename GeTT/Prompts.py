import os
import pandas as pd
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from Sampling.dataTypes import determine_data_type
from Sampling.SpecificTypes import numeric, textual
from Sampling.Table import column_context, example_data, colMetadata

"""
Later we will replace the specified one with iteration over the whole dataset
"""








"""dp2 = "E:\Project\CurrentDataset\datasets\Old\open_data\Test\\"
dname = "project_UKRI-25112022-Total-Expenditure-September-2022.csv"
t2 = pd.read_csv(os.path.join(dp2, dname))
print(t2.head(5))
"""





def tableAllAnnotations(path: str, filename: str, LLM_model: OllamaLLM, store_path):
    table = pd.read_csv(os.path.join(path, filename))
    columnAnno, promptCol = column_annotation(table, LLM_model, table_mention=True)
    table_prompt = subjectColumnPrompt(table)
    subject_col_names = LLM_model.invoke(table_prompt)
    entityTPrompt = entityTypePrompt(table, columnAnnotation=promptCol)
    entityType = LLM_model.invoke(entityTPrompt)

    columnAnno1, promptCol1 = column_annotation(table, LLM_model, table_mention=True)
    subject_col_names1 = LLM_model.invoke(table_prompt)
    entityType1 = LLM_model.invoke(entityTPrompt)

    columnAnno2, promptCol2 = column_annotation(table, LLM_model, table_mention=True)
    subject_col_names2 = LLM_model.invoke(table_prompt)
    entityType2 = LLM_model.invoke(entityTPrompt)

    mkdir(os.path.join(store_path, filename[:-4]))
    print(subject_col_names)
    print(entityType)
    summary_columns = ["Subject Column(s)", "Entity type"]
    dfSummary = pd.DataFrame(columns=summary_columns)
    slist = [{"Subject Column(s)": subject_col_names, "Entity type": entityType},
             {"Subject Column(s)": subject_col_names1, "Entity type": entityType1},
             {"Subject Column(s)": subject_col_names2, "Entity type": entityType2}]
    for item in slist:
        dfSummary = dfSummary._append(item, ignore_index=True)
    columns = ['Attribute Header', 'Response 1', 'Response 2', 'Response 3']
    df = pd.DataFrame(columns=columns)
    for col in table.columns:
        data = {'Attribute Header': col,
                'Response 1': columnAnno[col],
                'Response 2': columnAnno1[col],
                'Response 3': columnAnno2[col]}
        df = df._append(data, ignore_index=True)

    df.to_csv(os.path.join(store_path, filename[:-4], "CTA.csv"))
    dfSummary.to_csv(os.path.join(store_path, filename[:-4], "SC_ET.csv"))




### LLM


Dataset_name = "GDS"
dataPath = f"E:/Project/CurrentDataset/datasets/{Dataset_name}/Test/"
store_path = f"Result/{Dataset_name}/"

"""sample_names_csv = pd.read_csv(f"sample_selection_{Dataset_name}.csv")
sample_names = sample_names_csv["Table_name"]
for name in list(sample_names):
    print(name)
    tableAllAnnotations(dataPath, name, llm, store_path)"""
