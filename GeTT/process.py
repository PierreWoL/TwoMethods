import json
import pandas as pd

from utils import mkdir

numbers = ["0","1", "2", "3","4"]
dataset = "WDC"
dict = []
for number in numbers:
    data = pd.read_csv(f"Result\{dataset}\qwen14\entityTypeS1Try{number}.csv")
    df_cleaned = data[~data.apply(lambda x: x == '').any(axis=1)]
    result = list(data["inferred_type"].dropna().unique())
    print(result)
    test = {"root": "Thing", "entity_list": result}

    dict.append(test)
    print(len(result))
    print(test)
mkdir(f"Layer\dataset\processed\{dataset}\\gpt4\\")
with open(f'Layer\dataset\processed\{dataset}\\qwen32\\test.json', 'w') as f:
        for d in dict:
            json.dump(d, f)
            f.write('\n')

    # with open('E:\SILLM\Chain-of-Layer\dataset\processed\GDS\\test.json', 'w', encoding='utf-8') as file:
#   json.dump(data, file, separators=(',', ':'))
