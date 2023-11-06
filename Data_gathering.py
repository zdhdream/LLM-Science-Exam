import os
import random
import openai
import requests
import itables
import wikipediaapi
import pandas as pd
import numpy as np
import plotly.express as px

openai.api_key = "ViUJCDzcLEvG8QP7Ka5tT3BlbkFJfL8WhqdZyuBQPFOhrTyn"
pages_count = 2
max_completion_attempts = 10
wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (zdh20211025@gmail.com)', 'en')

###############################################################
##############    Form a list of STEM topics    ###############
###############################################################

# probabilities: S -> 0.294; T,E,M -> 0.235
STEM_WEIGHTS = [1.25, 1, 1, 1]

STEM = {
    "S": ["Category:Applied_sciences", "Category:Biotechnology", "Category:Biology", "Category:Natural_history"],
    "T": [
        "Category:Technology_strategy", "Category:Technical_specifications", "Category:Technology_assessment",
        "Category:Technology_hazards", "Category:Technology_systems", "Category:Hypothetical_technology",
        "Category:Mobile_technology", "Category:Obsolete_technologies", "Category:Philosophy_of_technology",
        "Category:Real-time_technology", "Category:Software", "Category:Technology_development",
        "Category:Computing", "Category:Artificial_objects", "Category:Technological_change",
        "Category:Technical_communication", "Category:Technological_comparisons"
    ],
    "E": ["Category:Engineering_disciplines", "Category:Engineering_concepts", "Category:Industrial_equipment",
          "Category:Manufacturing"],
    "M": ["Category:Fields_of_mathematics", "Category:Physical_sciences"]
}

EXCLUDE_CATEGORIES = set([
    "Category:Technology", "Category:Mathematics", "Category:Works about technology",
    "Category:Technology evangelism", "Category:Artificial objects", "Category:Fictional physical scientists"
])

###############################################################
#########    Randomly select a category or a page    ##########
###############################################################
def split_category_members(members):
    """将给定成员列表分割成两个列表,一个列表包含类别(Category),另一个包含页面(Page)成员"""
    category_list, page_list = [], []

    for member_name, member_page in members:
        if member_name.startswith('Category') and member_name not in EXCLUDE_CATEGORIES:
            category_list.append((member_name, member_page))
        else:
            page_list.append((member_name, member_page))

    return category_list, page_list


def get_wiki_random_page(deep_subcategories=True):
    """从维基百科中随机获取一个页面(Page)或类别(Category)"""
    stem_label, stem_categories = random.choices(list(STEM.items()), weights=STEM_WEIGHTS, k=1)[0]
    category = random.choice(stem_categories)
    category_page = wiki_wiki.page(category)
    while True:
        # 获取当前类别页面
        chosen_list = list(category_page.categorymembers.items())
        # 将类别成员列表分成类别列表和页面列表
        if deep_subcategories:
            category_list, page_list = split_category_members(chosen_list)
            chosen_list = []
        else:
            category_list, page_list = [], []

        # 50% change to select category or page list if one of them isn't empty
        # helps to go deeper into subcategories because there're more pages than categories
        if not (category_list or page_list) and not chosen_list:
            continue
        elif not category_list:  # 若当前类别没有类别列表选择页面
            chosen_list = page_list
        elif not page_list:  # 若当前类别没有页面则选择类别
            chosen_list = category_list
        else:  # 如果都有则随机选择一个页面或者类别
            chosen_list = random.choice([category_list, page_list])
        # 从选定的列表中随机选择一个页面或类别
        # select random page from chosen list
        selected_page_name, selected_page = random.choice(chosen_list)

        if not selected_page_name.startswith("Category"):
            break
        # 将当前类别页面更新为所选类别页面
        category_page = selected_page

    return selected_page, stem_label


###############################################################
#########    Extract the text from selected page    ###########
###############################################################
def get_wiki_text(seen_pages, min_page_length=6, sentences_include=3):
    """用于从维基百科获取随机页面的文本
    seen_pages: 已看过的页面列表
    min_page_length: 页面最小长度限制
    sentences_include: 提取句子的数量
    """
    while True:
        wiki_page, stem_label = get_wiki_random_page()

        if wiki_page.pageid in seen_pages:
            continue

        page_sentences = wiki_page.text.split(". ")

        # check is the page is long enought
        if len(page_sentences) >= min_page_length:
            # main information about the topic usualy described within first 3 sentences
            wiki_text = ". ".join(page_sentences[:sentences_include]) + "."
            break

    return wiki_text, wiki_page.pageid, wiki_page.title, stem_label


###############################################################
#########    Compose a message to the LLM model    ##########
###############################################################
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
    """
    生成包含对象消息的列表，这些消息可以在对话系统中使用
    模拟用户和系统之间的交互
    wiki_text: 用户在对话中提供的文本
    """
    return [
        # 系统消息
        {
            'role': 'system',
            'content': system_message
        },
        # 用户消息
        {
            'role': 'user',
            'content': f"{delimiter}{wiki_text}{delimiter}"
        },
    ]


def get_completion_from_messages(
        messages,
        model="gpt-3.5-turbo",
        temperature=0.8,
        max_tokens=3000
):
    """通过OpenAI的GPT3.5 Turbo模型进行对话生成
    message: list:包含对话消息的字典,每个字典具有role和content字段,表示消息的角色和内容
    model: 用于生成对话的模型，这里模型使用gpt-3.5-turbo
    temperature: 控制生成文本的随机性，较高的值会产生更随机的文本，较低的值会更加确定性
    max_tokens: 生成的最大标记数（令牌数），用于限制生成文本的长度
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]


###############################################################
#########    Combine all elements of the pipeline    ##########
###############################################################
def is_correctly_formatted(mcq) -> bool:
    """检查一组多项选择题(MCQ)是巨头正确的格式"""
    return all([
        len(el) == len(response_keys_set) and response_keys_set == set(list(el.keys()))
        for el in mcq
    ])


def gather_multiple_choice_question_dataset(
        pages_count: int,
        max_completion_attempts: int = 10,
        seen_pages: list = []
):
    """生成多选题数据集
    pages_count: 要生成的多选题数量
    max_completion_attempts: 生成每个页面的最大尝试次数，默认为10次
    seen_pages: 已看过的页面列表，默认为列表
    """
    attempts_list = []  # 记录每个页面的尝试次数
    multiple_choice_questions = []  # 存储生成的多选题

    generated_count = 0  # 记录已生成的多选题数量
    while generated_count < pages_count:
        wiki_text, page_id, page_title, stem_label = get_wiki_text(seen_pages, sentences_include=7)
        print(
            f"\nStart multiple choice questions generation: page_id={page_id}, page_title={page_title}, stem_label={stem_label}")

        messages = get_completion_messages(wiki_text)

        attempts_counter = 0
        while True:
            try:
                chatgpt_response = get_completion_from_messages(messages)
                mcq = eval(chatgpt_response)

                if not isinstance(mcq, list) or len(mcq) < 5 or not is_correctly_formatted(mcq):
                    raise Exception

                for i in range(len(mcq)):  # 为每个多选题添加额外的信息
                    mcq[i]["wiki_text"] = wiki_text
                    mcq[i]["page_id"] = page_id
                    mcq[i]["page_title"] = page_title
                    mcq[i]["stem_label"] = stem_label

                    if mcq[i]["answer"] in options_set:
                        continue
                    else:
                        # index method will raise an error if answer isn't in list
                        answ_indx = [v.lower() for v in mcq[i].values()].index(mcq[i]["answer"].lower())
                        mcq[i]["answer"] = list(mcq[i].keys())[answ_indx]

                multiple_choice_questions += mcq
                seen_pages.append(page_id)
                generated_count += 1
                print("Generated count:", generated_count)
                break
            except Exception:
                attempts_counter += 1
                print("Attempts count:", attempts_counter)
                attempts_list.append(attempts_counter)
                if attempts_counter > max_completion_attempts:
                    break

    return multiple_choice_questions, seen_pages, attempts_list


def conver_df_to_compet_format(df):
    """To convert df_mcq to competition format"""
    df_compet = df.copy(deep=True)
    df_compet.insert(0, "id", list(range(len(df_compet))))
    df_compet.rename(
        columns = {
            'question': 'prompt',
            'option_1': 'A',
            'option_2': 'B',
            'option_3': 'C',
            'option_4': 'D',
            'option_5': 'E'
        },
        inplace = True
    )

    answer_subjects = {
        'option_1': 'A',
        'option_2': 'B',
        'option_3': 'C',
        'option_4': 'D',
        'option_5': 'E'
    }
    df_compet["answer"] = df_compet["answer"].map(answer_subjects)
    df_compet = df_compet.drop(columns=["wiki_text", "page_id", "page_title", "stem_label"])

    return df_compet



def main():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    multiple_choice_questions, seen_pages, attempts_list = gather_multiple_choice_question_dataset(
        pages_count, max_completion_attempts
    )
    # Let's examine the output provided by the GPT3.5-turbo model.
    df_mcq = pd.DataFrame.from_records(multiple_choice_questions)
    df_compet = conver_df_to_compet_format(df_mcq)
    df_compet.to_csv("/data/stem_dataset.csv", index=False)



if __name__ == "__main__":
    main()
