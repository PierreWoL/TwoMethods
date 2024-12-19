import pandas as pd
from langchain_ollama import OllamaLLM
from openai import OpenAI
from utils import dataframe_to_markdown, dataframe_to_mdd
from Sampling.SpecificTypes import numeric, textual
from Sampling.Table import column_context, example_data, colMetadata
from Sampling.dataTypes import determine_data_type
from response import get_llama_answer


def isNamedEntity(col: pd.Series, chatgpt=False):
    # TODO not important this is just a test
    col_str = '; '.join(col.head(5).astype(str))
    col_text = "Column name: " + str(col.name) + f"\n,  cells:{col_str}."

    # print(col_text)
    system = ("You are an expert at identifying named entities of column cells."
              "A column should be considered a named entity column only if each cell is an entity.")
    user = (f"Please judge if the given column is a named entity column based on the column header and cells. "
            f".\n {col_text}. Let's think step by step.")  # and give your reasons
    systemR = "Do not add any EXTRA comments. Only answer Yes or No for the whole column."

    if chatgpt:
        messages = [
            {"role": "system", "content": system, },
            {"role": "user", "content": user, },
            {"role": "system", "content": systemR, }
        ]
    else:
        messages = [
            ("system", system),
            ("human", user),
            ("system", systemR),
        ]
    # judge = llm.invoke(messages)
    # print(judge)
    return messages


def columnDataType(table, col_name, table_mention=False, size=5, chatgpt=False):
    sampleT = example_data(table, sample_size=size)
    sampleT_MD = dataframe_to_markdown(sampleT)
    s_t = f"sampled table: \n {sampleT_MD} \n" if table_mention is True else ""
    col = table[col_name]
    dataType = determine_data_type(col)
    # if dataType in ["Boolean", "Date", "Time", "DateTime"]:
    # return dataType, None
    # else:
    # selections = ", ".join(numeric) if dataType == "Number" else ", ".join(textual)
    selections = ", ".join(numeric) + ", ".join(textual)
    # System = "You are an expert that specializes in identifying data types of columns."
    colMeta = colMetadata(col, sample=size)
    user = (
        "Please pick the data type which best describes the column,"
        " based on the column content and its metadata. "
        f"Options: {selections}"
        f"{colMeta}."
        f"\n{s_t}."
    )
    restriction = "Do not add any EXTRA comments. ONLY OUTPUT selected type from the options."
    if chatgpt:
        messages = [
            # {"role": "system", "content": System, },
            {"role": "user", "content": user, },
            {"role": "system", "content": restriction, }
        ]
    else:
        messages = [
            # ("system", System),
            ("human", user),
            ("system", restriction),
        ]
    # print( user, restriction)
    return dataType, messages


# Test


def columnTypePrompt(table, col_name, table_mention=False, size=5, chatgpt=False):  # dataTypeDetail="",
    sampleT = example_data(table, sample_size=size)
    sampleT_MD = dataframe_to_mdd(sampleT, selectedHeader=col_name)
    col = table[col_name]
    dataType = determine_data_type(col)
    # if dataType in ["Boolean", "Date", "Time", "DateTime"]:
    #   return dataType, None
    # else:
    # if dataType in ["Boolean", "Number"]:
    #    table_mention = True
    s_t = f"Table: \n {sampleT_MD} \n" if table_mention is True else ""
    System = "Please infer the semantic type of the column based on the column content and its metadata."
    colMeta = colMetadata(col, sample=size)
    user = (
        f"Please infer semantic type of the column."
        f"{colMeta}."
        f"\n{s_t}."
    )
    restriction = ("Do not add any EXTRA comments. "
                   "ONLY OUTPUT inferred type that best describes the column."
                   )
    if chatgpt:
        messages = [
            {"role": "system", "content": System, },
            {"role": "user", "content": user, },
            {"role": "system", "content": restriction, }
        ]
    else:
        messages = [
            ("system", System),
            ("human", user),
            ("system", restriction),
        ]
    return dataType, messages


def columnNETypes_prompt(table, col_name, table_mention=False, size=5, chatgpt=False):
    sampleT = example_data(table)
    sampleT_MD = dataframe_to_mdd(sampleT)
    col = table[col_name]
    prompt_col = column_context(col, fraction=size)

    context_mention = (" Leverage the cells provided by the other columns "
                       "in the table to refine and restrict the semantic types.") if table_mention is True else ""
    ins_mention = "a table and " if table_mention is True else ""
    User_mention = f"Table: \n{sampleT_MD} \n" if table_mention is True else ""
    instruction_prompt = (
            "You are an AI language model that specializes in inferring semantic types of columns in a table. " +
            f"When provided with {ins_mention} a specific column with header," +
            " analyze the data in that column to infer possible semantic types. " +
            "Express these types as natural language phrases (multi-labels), and each of them should not repeat; " +
            "ensuring each type corresponds to a semantic domain represented in the cells." +
            f" Use the information from the given cells to support your inferences. {context_mention}")
    User_prompt = (f"{User_mention}"
                   + f"Column: \n{prompt_col}\n" +
                   f"Please infer the semantic type of the column {col_name} based on the data provided. " +
                   "List several possible natural language phrases as multi-labels,"
                   " with each label corresponding to an possible entity type represented in the cells.")
    Restriction_prompt = "Do not add any EXTRA comments. ONLY OUTPUT the inferred types separated by commas."
    if chatgpt:
        message = [
            {"role": "system", "content": instruction_prompt, },
            {"role": "user", "content": User_prompt, },
            {"role": "system", "content": Restriction_prompt, }
        ]
    else:
        message = [
            ("system", instruction_prompt),
            ("human", User_prompt),
            ("system", Restriction_prompt),
        ]
    # print(instruction_prompt, User_prompt, Restriction_prompt)
    return message


def subjectColumnPrompt(table, chatgpt=False):
    # TODO this needs another thoughts
    instructions_prompt = (f"When provided with a table, "
                           "analyze the information to determine the subject column(s) of the table."  # the overall entity type
                           " Select the subject column name(s) from the given column names, "
                           "considering all relevant aspects from the data and semantic types.")
    sample_table = example_data(table)
    User_prompt = (f"Column names: {list(sample_table.columns)}. \nTable: {sample_table}. \n " +
                   "Given column names, please select the subject column(s) name of the table based on table itself ")
    Restriction_prompt = ('Do not add any EXTRA comments. Output only subject column name(s) of the table separated '
                          'by commas.')
    if chatgpt:
        message = [
            {"role": "system", "content": instructions_prompt, },
            {"role": "user", "content": User_prompt, },
            {"role": "system", "content": Restriction_prompt, }
        ]
    else:
        message = [
            ("system", instructions_prompt),
            ("human", User_prompt),
            ("system", Restriction_prompt),
        ]

    return message


def column_annotationLlama(given_table, col_name, LLM_model: OllamaLLM, table_mention=False, size=5):
    dataType, cdt_prompt = columnDataType(given_table, col_name, table_mention=table_mention, size=size)
    if cdt_prompt is not None:
        dataType = get_llama_answer(LLM_model, cdt_prompt)
    columnType, columnType_prompt = columnTypePrompt(given_table, col_name, table_mention=table_mention, size=size)
    if columnType_prompt is not None:
        columnType = get_llama_answer(LLM_model, columnType_prompt)
    NE_variations = ""
    # if "named entity" in dataType:
    NEType_prompt = columnNETypes_prompt(given_table, col_name, table_mention=table_mention, size=size)
    NE_variations = get_llama_answer(LLM_model, NEType_prompt)
    return dataType, columnType, NE_variations


### Example here
"""ollama_base_url = "http://localhost:11434/"
llm = OllamaLLM(
    base_url=ollama_base_url,
    model='llama3.1'
)
table = pd.read_csv("E:\SILLM\open_data\Test\\restaurant_Top250.csv")
print(table.columns)
dataType, columnType, NE_variations = column_annotationLlama(table, "Segment_Category", llm, table_mention=True)
print(f"dataType:{dataType} \n columnType:{columnType} \n , NE_variations: {NE_variations} \n ")
"""
# apiKey = ""
# gpt = OpenAI(api_key=apiKey)
