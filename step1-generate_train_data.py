import os
import random
import openai
import requests
import wikipediaapi
import itables
import numpy as np
import pandas as pd
import plotly.express as px
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

openai.api_key = "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"
options_set = set(("option_1", "option_2", "option_3", "option_4", "option_5"))
response_keys_set = set(("question", "option_1", "option_2", "option_3", "option_4", "option_5", "answer"))
delimiter = "####"
system_message = f"""
You will be provided with TEXT from wikipedia. \
The TEXT will be delimited with {delimiter} characters.
Output a python list of 5 dict objects, where each object is \
a multiple choice question whom answers should be in \
the given TEXT and that has 5 choices each and has the following format:
    'question': <question on the TEXT>
    'option_1': <question answer option>
    'option_2': <question answer option>
    'option_3': <question answer option>
    'option_4': <question answer option>
    'option_5': <question answer option>
    'answer': <answer option key label>

You should tell me which one of your proposed options is right \
by assigning the corresponding option's key label in the 'answer' field.

The question, the answer and question answer options should be broad, \
challenging, long, detailed and based on the TEXT provided.

Only output the list of objects, with nothing else.
"""


def get_completion_messages(wiki_text):
    return [
        {
            'role': 'system',
            'content': system_message
        },

        {
            'role': 'user',
            'content': f"{delimiter}{wiki_text}{delimiter}"
        },
    ]


def get_completion_from_messages(
        messages,
        model="gpt-3.5-turbo-0613",
        temperature=0.8,
        max_tokens=6000
):
    try:

        response = openai.ChatCompletion.create(

            model=model,

            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,

        )

        completion = {'role': '', 'content': ''}

        for event in response:

            if event['choices'][0]['finish_reason'] == 'stop':
                # print(f'收到的完成数据: {completion}')

                break

            for delta_k, delta_v in event['choices'][0]['delta'].items():
                # print(f'流响应数据: {delta_k} = {delta_v}')

                completion[delta_k] += delta_v

        messages.append(completion)  # 直接在传入参数 messages 中追加消息

        return (True, '')

    except Exception as err:

        return (False, f'OpenAI API 异常: {err}')


def is_correctly_formatted(mcq) -> bool:
    return all([
        len(el) == len(response_keys_set) and response_keys_set == set(list(el.keys()))
        for el in mcq
    ])


def main():
    # https://www.kaggle.com/code/nbroad/create-science-wikipedia-dataset
    files = list(map(str, Path("data/wiki-20220301-en-sci").glob("*.parquet")))
    ds = load_dataset("parquet", data_files=files, split="train")

    max_completion_attempts = 5
    multiple_choice_questions = []
    count = 0
    thread = 10000
    for index, row in tqdm(enumerate(ds), total=len(ds)):
        # get wiki passage
        text = row['text']
        message = get_completion_messages(text)
        attempts_counter = 0
        if count >= thread:
            break
        while True:
            try:
                get_completion_from_messages(
                    message)  # according to every passage to generate (prompt, five options, answer)
                assert len(message) == 3
                chatgpt_response = message[-1]['content']
                # print(chatgpt_response)
                mcq = eval(chatgpt_response)

                if not isinstance(mcq, list) or len(mcq) < 5 or not is_correctly_formatted(mcq):
                    raise Exception

                for i in range(len(mcq)):
                    mcq[i]["ori_index"] = index
                    mcq[i]["ori_wiki"] = text
                    if mcq[i]["answer"] in options_set:
                        continue
                    else:
                        # index method will raise an error if answer isn't in list
                        answ_indx = [v.lower() for v in mcq[i].values()].index(mcq[i]["answer"].lower())
                        mcq[i]["answer"] = list(mcq[i].keys())[answ_indx]

                multiple_choice_questions += mcq
                # print("Generated count:", index+1)
                break
            except Exception:
                attempts_counter += 1
                print("Attempts count:", attempts_counter)
                if attempts_counter > max_completion_attempts:
                    break
        count += 1

    print(f'length of multiple_choice_questions: {multiple_choice_questions}')

    df_mcq = pd.DataFrame.from_records(multiple_choice_questions)
    df_mcq.to_csv("/data/generation_data/train_qa.csv", index=None)

    ori_df = pd.DataFrame(ds)
    ori_df.to_csv("/data/generation_data/ori_dataset.csv", index=None)

    df_final = df_mcq.merge(ori_df, how='left', left_on='ori_text', right_on='text')
    df_final = df_final[['question', 'text', 'url', 'title']]
    df_final.to_csv("/data/generation_data/retrive_dataset.csv", index=False)

    neg = []
    for row in tqdm(df_final.iterrows(), total=len(df_final)):
        row = row[1]
        url = row.url
        question = row.question
        candidates = ori_df[ori_df['url'] != url]
        candidates = candidates.sample(n=10)
        candidates['ori_url'] = url
        candidates['question'] = question
        neg.append(candidates)

    neg = pd.concat(neg)
    neg = neg.drop_duplicates()
    neg.to_csv('/data/generation_data/neg.csv', index=None)

    val_id = ori_df.url.unique()
    val_id = val_id[:1000]
    np.save('/data/generation_data/val_id', val_id)


if __name__ == "__main__":
    main()
