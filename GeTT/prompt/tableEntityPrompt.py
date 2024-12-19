
from Sampling.Table import example_data
from utils import dataframe_to_sample,sample

Restriction_prompt = 'Do not add any EXTRA comments. Output only the name of entity type.'


def entityTypePrompt(Table, model="qwen"):

    instructions_prompt = ("You are an AI language model specialized in inferring table entity types. "
                           "Given a table, identify the entity in each row and infer the overall entity type they represent.")
    sample_table = example_data(Table)
    table_input = sample(Table)
    User_prompt = (f"Table: {table_input} \n " +

                   "Please infer the entity type of the table")  # +
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
    table_input = sample(Table)
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

