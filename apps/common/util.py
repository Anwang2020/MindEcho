from langchain_core.prompts import load_prompt


def custom_load_prompt(path):
    try:
         prompt = load_prompt(path, encoding='utf-8')
    except:
         prompt = load_prompt(path, encoding='gbk')

    return prompt