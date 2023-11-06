import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import openai
import ast

from aiohttp import ClientSession
import asyncio

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

openai.api_key = "ViUJCDzcLEvG8QP7Ka5tT3BlbkFJfL8WhqdZyuBQPFOhrTyn"

# 创建一个信号量对象，限制同时进行地并发请求数量为10
request_semaphore = asyncio.Semaphore(10)


# @retry是一个装饰器,用于对get_completion_from_meassage函数进行自动重试
# 指定了在失败时如何进行重试,包括等待时间和最大尝试次数
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    """使用Python的异步编程实现并发地向OpenAId地聊天模型发送请求并获取响应"""
    ## max number of concurrent requests
    async with request_semaphore:
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            temperature=temperature,  # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]


def map3_score(y_true, y_pred):
    score = 0
    for y, pred in zip(y_true, y_pred):
        if y == pred[0]:
            score += 1
        elif y == pred[1]:
            score += 2 / 3
        elif y == pred[2]:
            score += 1 / 3

    return score / len(y_true)


def accuracy_score(y_true, y_pred):
    score = 0
    for y, pred in zip(y_true, y_pred):
        if y == pred[0]:
            score += 1

    return score / len(y_true)


def get_best_answers(json_response):
    try:
        x = json_response["best_answers"]
        if len(x) == 3:
            x = [a.strip()[0:1] for a in x]
    except:
        x = [None, None, None]
    return x


###############################################################
#############       Basic/Naive           #####################
###############################################################

def create_messages_basic(df, idx):
    question = df.loc[idx, "prompt"]
    A = df.loc[idx, "A"]
    B = df.loc[idx, "B"]
    C = df.loc[idx, "C"]
    D = df.loc[idx, "D"]
    E = df.loc[idx, "E"]
    answer = df.loc[idx, "answer"]

    instructions = """
You need to provide the best possible answer to the user's question  by selecting the first char (A,B,C,D,E) of three best options  for the question ranked from best to worst.
Don't write anything but provide result in a single JSON with the following format 
{
    best_answers = [ ".",".", "." ],

}
"""

    messages = [
        {'role': 'user', 'content': f"""
{instructions}

User's question:
{question}
Options:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}
"""
         }
    ]
    return messages


async def get_completetion(df, idx):
    messages = create_messages_basic(df, idx)

    try:
        raw_response = await get_completion_from_messages(messages)
    except:
        raw_response = None

    try:
        json_response = ast.literal_eval(raw_response)
    except:
        json_response = None

    return raw_response, json_response


###############################################################
##########   Give the model time to “think”   #################
###############################################################
def create_messages_advanced(df, idx):
    question = df.loc[idx, "prompt"]
    A = df.loc[idx, "A"]
    B = df.loc[idx, "B"]
    C = df.loc[idx, "C"]
    D = df.loc[idx, "D"]
    E = df.loc[idx, "E"]
    answer = df.loc[idx, "answer"]

    instructions = """
You are a group of experts in STEM subjects.
You need to provide the best possible answer to the user's question.

First, choose the STEM expert who will answer the question. You can choose beetween: Science, Technology, Engineering, and Mathematics expert.
Then, let the expert thinks step by steps to answer the question with in-depth explanation.
Then, summarize the answer in less then 100 words.
Finally, let the expert select first char (A,B,C,D,E) of the three best options for the question ranked from best to worst.

Don't write anything but provide result in a single JSON with the following format 
{
	expert="...",
	expert_answer_summarized ="...",
	best_answers = [ ".",".", "." ],
 	best_answers_explanations = ["...","...","..."]

}
"""

    messages = [
        {'role': 'user', 'content': f"""
{instructions}

User's question:
{question}
Options:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}
"""
         }
    ]
    return messages


async def get_completetion(df, idx):
    messages = create_messages_advanced(df, idx)

    try:
        raw_response = await get_completion_from_messages(messages)
    except:
        raw_response = None

    try:
        json_response = ast.literal_eval(raw_response)
    except:
        json_response = None

    return raw_response, json_response


###############################################################
#############       Two Steps               ###################
###############################################################
def create_messages_firststep(df, idx):
    question = df.loc[idx, "prompt"]

    instructions = """
You are a group of experts in STEM subjects.
You need to provide the best possible answer to the user's question.

First, choose the STEM expert who will answer the question. You can choose beetween: Science, Technology, Engineering, and Mathematics expert.
Then, let the expert thinks step by steps to answer the question with in-depth explanation.
Then, summarize the answer in less then 100 words.

Don't write anything but provide result in a single JSON with the following format 
{
	expert="...",
	expert_answer_summarized ="...",
}

"""
    messages = [
        {'role': 'user', 'content': f"""
{instructions}

User's question:
{question}
"""
         }
    ]
    return messages


def create_messages_secondstep(df, idx):
    question = df.loc[idx, "prompt"]
    A = df.loc[idx, "A"]
    B = df.loc[idx, "B"]
    C = df.loc[idx, "C"]
    D = df.loc[idx, "D"]
    E = df.loc[idx, "E"]

    expert = df.loc[idx, "expert_response"]
    expert_answer_summarized = df.loc[idx, "expert_answer_summarized_response"]

    instructions = """
You must think step by steps to answer the following user's question with in-depth explanation.
Then, summarize the answer in less then 50 words.

Finally, you must select first char (A,B,C,D,E) of the three best options for the question ranked from best to worst.

Don't write anything but provide result in a single JSON with the following format 
{
    best_answers = [ ".",".", "." ],
    best_answers_explanations = ["...","...","..."]

}
"""

    messages = [
        {'role': 'user', 'content': f"""
You're a {expert} expert.

Context:
{expert_answer_summarized}

{instructions}

User's question:
{question}
Options:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}
"""
         }
    ]

    return messages


async def get_completetion_firststep(df, idx):
    messages = create_messages_firststep(df, idx)

    try:
        raw_response = await get_completion_from_messages(messages)
    except:
        raw_response = None

    try:
        json_response = ast.literal_eval(raw_response)
    except:
        json_response = None

    return raw_response, json_response


async def get_completetion_secondstep(df, idx):
    messages = create_messages_secondstep(df, idx)

    try:
        raw_response = await get_completion_from_messages(messages)
    except:
        raw_response = None

    try:
        json_response = ast.literal_eval(raw_response)
    except:
        json_response = None

    return raw_response, json_response

df = pd.read_csv("data/train.csv")
###############################################################
#############       Basic/Naive           #####################
###############################################################
basic_df = df.copy()
openai.aiosession.set(ClientSession())
results = await asyncio.gather(*[get_completetion(basic_df, idx) for idx in range(basic_df.shape[0])])
await openai.aiosession.get().close()
raw_responses = [raw_response for (raw_response, json_response) in results]
json_responses = [json_response for (raw_response, json_response) in results]
basic_df["raw_response"] = raw_responses
basic_df["json_response"] = json_responses
basic_df["best_answers_response"] = basic_df["json_response"].map(lambda x: get_best_answers(x))
basic_df.to_csv("data/basic.csv", index=False)
map3 = map3_score(basic_df["answer"].values, basic_df["best_answers_response"].values)
accuracy = accuracy_score(basic_df["answer"].values, basic_df["best_answers_response"].values)
print(f"accuracy: {accuracy:.5f} map3 : {map3:.5f}")

###############################################################
##########   Give the model time to “think”   #################
###############################################################
advanced_df = df.copy()
openai.aiosession.set(ClientSession())
results = await asyncio.gather(*[get_completetion(advanced_df, idx) for idx in range(advanced_df.shape[0])])
await openai.aiosession.get().close()

raw_responses = [raw_response for (raw_response, json_response) in results]
json_responses = [json_response for (raw_response, json_response) in results]

advanced_df["raw_response"] = raw_responses
advanced_df["json_response"] = json_responses
advanced_df["best_answers_response"] = advanced_df["json_response"].map(lambda x: get_best_answers(x))
advanced_df["expert_response"] = advanced_df["json_response"].map(lambda x: x["expert"])
advanced_df["expert_answer_summarized"] = advanced_df["json_response"].map(lambda x: x["expert_answer_summarized"])

advanced_df.to_csv("advanced.csv", index=False)

map3 = map3_score(advanced_df["answer"].values, advanced_df["best_answers_response"].values)
accuracy = accuracy_score(advanced_df["answer"].values, advanced_df["best_answers_response"].values)
print(f"accuracy: {accuracy:.5f} map3 : {map3:.5f}")

###############################################################
#############       Two Steps               ###################
###############################################################
twosteps_df = df.copy()

openai.aiosession.set(ClientSession())
results = await asyncio.gather(
    *[get_completetion_firststep(twosteps_df, idx) for idx in range(twosteps_df.shape[0])])

raw_responses = [raw_response for (raw_response, json_response) in results]
json_responses = [json_response for (raw_response, json_response) in results]

twosteps_df["raw_response_1"] = raw_responses
twosteps_df["json_response_1"] = json_responses
twosteps_df["expert_response"] = twosteps_df["json_response_1"].map(lambda x: x["expert"])
twosteps_df["expert_answer_summarized_response"] = twosteps_df["json_response_1"].map(
    lambda x: x["expert_answer_summarized"])

results = await asyncio.gather(
    *[get_completetion_secondstep(twosteps_df, idx) for idx in range(twosteps_df.shape[0])])

raw_responses = [raw_response for (raw_response, json_response) in results]
json_responses = [json_response for (raw_response, json_response) in results]

twosteps_df["raw_response_2"] = raw_responses
twosteps_df["json_response_2"] = json_responses

await openai.aiosession.get().close()

twosteps_df["best_answers_response"] = twosteps_df["json_response_2"].map(lambda x: get_best_answers(x))

twosteps_df.to_csv("/data/twosteps.csv", index=False)
map3 = map3_score(twosteps_df["answer"].values, twosteps_df["best_answers_response"].values)
accuracy = accuracy_score(twosteps_df["answer"].values, twosteps_df["best_answers_response"].values)
print(f"accuracy: {accuracy:.5f} map3 : {map3:.5f}")
