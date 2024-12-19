
from Sampling.Table import example_data
from utils import dataframe_to_markdown

Restriction_prompt = 'Do not add any EXTRA comments. Output only the name of entity type.'


def entityTypePrompt(Table, columnAnnotation=None, model="qwen"):
    if columnAnnotation is None:
        colAnnotation_instruction = " and the inferred semantic types of each column, in the format of [column name]: type1, type2..."
        end_extra = " and semantic types"
    else:
        colAnnotation_instruction = ""
        end_extra = ""
    instructions_prompt = ("You are an AI language model that specializes in inferring the entity types of tables. "
                           + f"When provided with a table, {colAnnotation_instruction}"
                             "analyze the information to determine the overall entity type of the table."  # 
                             f"considering all relevant aspects from the data{end_extra}.")
    sample_table = example_data(Table)
    table_input = dataframe_to_markdown(sample_table)

    user_colAnnotationP = f"Inferred Semantic Types of Each Column: {columnAnnotation} \n" \
        if columnAnnotation is not None else ""
    tip = "and the inferred semantic types of each column. " if columnAnnotation is not None else ". "
    User_prompt = (f"Table: {table_input} \n " +
                   f"{user_colAnnotationP}" +
                   "Please infer the entity type of the table based on the data " +
                   f"{tip}")  # +
    if model=="gpt":
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
    #print(instructions_prompt, User_prompt,Restriction_prompt )
    return message


def tableTypePrompt(Table, chatgpt=False):
    instructions_prompt = ("Analyze the following table to determine the most suitable entity type"
                           " for the data represented. "
                           "The table includes columns that act as attributes of this entity type."
                           " Based on the context provided by these columns, "
                           "please identify the best-fit entity type that describes the rows as a collective group.")
    sample_table = example_data(Table)
    table_input = dataframe_to_markdown(sample_table)
    User_prompt = ("Sample table:\n" + table_input +
                   "Determine the entity type based on the attributes in the columns.")
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
    #print(instructions_prompt, User_prompt,Restriction_prompt )
    return message

