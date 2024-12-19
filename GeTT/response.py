from openai import OpenAI


def get_chatgpt_response(gpt_model, messages):
    response = gpt_model.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
    )
    new_assistant_reply = response.choices[0].message.content
    return new_assistant_reply


def get_llama_answer(model, messages):
    response = model.invoke(messages)
    return response


def get_model_answer(model, messages):
    response = ""
    if isinstance(model, OpenAI):
        response = get_chatgpt_response(model, messages)
    else:
        response = get_llama_answer(model, messages)
    return response
