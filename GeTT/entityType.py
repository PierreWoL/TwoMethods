from openai import OpenAI
import time
from evaluation import calculate_cosine_similarity, bertScoreMetric
from prompt.tableEntityPrompt import entityTypePrompt
import os
from response import get_model_answer
import pandas as pd
from langchain_ollama import OllamaLLM

from utils import mkdir

"""model_name = 'qwen2.5:14b'
ollama_base_url = "http://localhost:11434/"
llm = OllamaLLM(
    base_url=ollama_base_url,
    model= model_name
)"""

openai_key = ""
client = OpenAI(api_key=openai_key)

datasets = ["GDS","WDC"]
for dataset in datasets:
    data_path = f"datasets/{dataset}/Test/"
    groundTruth = pd.read_csv(f"datasets\{dataset}\groundTruth.csv")
    file_class_dict = dict(zip(groundTruth['fileName'], groundTruth['class']))

    data_names = [i for i in os.listdir(data_path) if i.endswith(".csv") and i in file_class_dict.keys()]
    groundTruthEntities = [file_class_dict[i] for i in data_names]
    for i in range(0, 5):
        start_time = time.time()
        print(i)
        entityTypes = []
        for data_name in data_names:  # [:4]
            t = pd.read_csv(os.path.join(data_path, data_name))
            entityTypeM = entityTypePrompt(t, model="gpt")
            # entityType = get_llama_answer(llm, entityTypeM)
            try:
                entityType = get_model_answer(client, entityTypeM)
            except:
                entityType = ""

            entityTypes.append(entityType)
            # print(entityType)
        end_time = time.time()  # 记录结束时间

        elapsed_time = end_time - start_time  # 计算运行时间
        print(f"{i}th running time: {elapsed_time:.4f} s")
        sbert_score = calculate_cosine_similarity(entityTypes, groundTruthEntities)
        bertScore = bertScoreMetric(entityTypes, groundTruthEntities)
        df = pd.DataFrame({
            'FileName': data_names,
            'specificType': groundTruthEntities,
            'inferred_type': entityTypes,
            'sbert_score': sbert_score,
            'bertScore_Precision': bertScore['precision'],
            'bertScore_Recall': bertScore['recall'],
            'bertScore_F1': bertScore['f1'],
        })
        # print(df)
        mkdir(f"Result/{dataset}/gpt4/")
        df.to_csv(f"Result/{dataset}/gpt4/entityTypeS1Try{i}.csv", index=False)


