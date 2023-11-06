import re
import random
import pandas as pd
import numpy as np
from config import CFG1
from random import shuffle
from data import OptionShuffle, tokenizer
from nltk.corpus import wordnet
from transformers import pipeline

cfg  = CFG1


###############################################################
##################    Augmentation    #########################
###############################################################
def augment_fn(df):
    new_df = []
    for index, row in df.iterrows():
        shuffled_row = OptionShuffle(row)
        new_df.append(shuffled_row)
    new_df = pd.concat(new_df, axis=1)
    return new_df.T


# Initialize the pipeline for masked language modeling using BERT
mlm_fill_mask = pipeline(task="fill-mask", model="model/bert-base-uncased")


# Function to perform data augmentation using different techniques
def augmentation_data(original_df, num_augmented_rows):
    augmented_data = []  # 用于存储增强后的数据
    original_rows = original_df.shape[0]  # 原始数据集的行数

    # Function for contextual word embeddings augmentation
    # 上下文词嵌入增强功能
    def contextual_embeddings(text):
        # Tokenize the text
        tokenized_text = tokenizer(text, return_tensors="pt")

        # Find masked positions in the tokenized text
        # 在被分词的文本中寻找mask标记
        masked_positions = [i for i, token in enumerate(tokenized_text["input_ids"][0]) if
                            token == tokenizer.mask_token_id]

        # If no masked positions found, return the original text
        if not masked_positions:
            return text

        # Randomly select one of the masked positions
        # 随机选择一个mask位置
        random_masked_position = random.choice(masked_positions)

        # Predict the masked word using masked language modeling
        # 使用掩码语言建模来预测掩码词
        masked_text = text.replace("[MASK]", tokenizer.mask_token)
        # 利用MLM pipline预测掩码文本中被掩码的词语,并从预测结果中获取预测的词语的字符串形式
        predicted_word = mlm_fill_mask(masked_text)[0]["token_str"]

        # Replace the masked word in the text with the predicted word
        # 这个预测的词语将被用于替换原始文本中的掩码
        augmented_text = text.replace(tokenizer.mask_token, predicted_word, 1)

        return augmented_text

    # Function for synonym replacement augmentation
    def augment_with_synonyms(text):
        # 你可以在这里使用自己的同义词替换逻辑
        # 为了简单起见，假设我们有一个预定义的同义词字典
        synonym_dict = {
            "good": ["excellent", "great", "superb", "fine"],
            "bad": ["poor", "terrible", "awful", "horrible"]
            # 根据需要添加更多的同义词
        }

        words = text.split()
        # 对每个词语进行同义词替换
        augmented_tokens = []
        for token in words:
            if token in synonym_dict:
                # 如果词语在同义词字典中,随机选择一个同义词进行替换
                synonym = random.choice(synonym_dict[token])
                augmented_tokens.append(synonym)
            else:
                augmented_tokens.append(token)
        # 将替换后的词语重新组合为增强后的文本
        augmented_text = " ".join(augmented_tokens)
        return augmented_text

    for _ in range(num_augmented_rows):
        original_row = original_df.iloc[random.randint(0, original_rows - 1)]  # 选择一个原始样本
        augmented_row = original_row.copy()  # 复制原始行以创建增强行

        # Apply augmentation techniques to "prompt"
        # 对"prompt"应用上下文词嵌入增强
        augmented_row["prompt"] = contextual_embeddings(original_row["prompt"])

        # Apply synonym replacement to answer choices (A, B, C, D, E)
        # 答案选项 (A, B, C, D, E) 应用同义词替换
        for choice in ["A", "B", "C", "D", "E"]:
            augmented_row[choice] = augment_with_synonyms(original_row[choice])

        augmented_data.append(augmented_row)

    return augmented_data


def get_only_chars(line):
    """从输入的文本中提取只包含小写英文字母和空格的纯净文本"""

    clean_line = ""

    line = line.replace("’", "")  # 将单引号替换为空字符串
    line = line.replace("'", "")
    line = line.replace("-", " ")  # 将-替换为空格
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'abcdefghijklmnopqrstuvwxyz$%0123456789':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if clean_line and clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

def synonym_replacement(words, n):
    """进行同义词替换
    n: 要替换的单词数目
    """
    new_words = words.copy()
    # 选择有意义的单词来进行同义词替换
    random_word_list = list(set([word for word in words if word not in cfg.stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):  # 遍历wordnet中给定单词的各种含义
        for l in syn.lemmas():  # 对于每个含义，遍历其各种同义词
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in 'abcdefghijklmnopqrstuvwxyz$%0123456789'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################
def random_deletion(words, p):
    # obviously, if there's only one word, don't delete it
    if len(words) <= 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    if len(new_words) <=0:
        return new_words
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    max_attempts = 10  # 设置尝试次数的阈值
    while len(synonyms) < 1:
        if len(new_words) == 0:
            return
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


########################################################################
# main data augmentation function
########################################################################

def eda(original_df, num_augment_rows, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1):
    augmented_data = []  # 用于存储增强后的数据
    original_rows = original_df.shape[0]  # 原始数据集的行数

    num_new_per_technique = int(num_augment_rows / 4) + 1  # 计算每种增强结束需要生成的新句子数量

    def aug_with_synonym_replacement(sentence):
        sentence = get_only_chars(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word is not '']  # 移除列表中为空的单词
        num_words = len(words)  # 计算单词列表的长度
        n_sr = max(1, int(alpha_sr * num_words))  # 计算用于同义词替换的单词数量
        a_words = synonym_replacement(words, n_sr)
        sentence = get_only_chars(' '.join(a_words))
        return sentence

    def aug_with_random_insert(sentence):
        sentence = get_only_chars(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word is not '']  # 移除列表中为空的单词
        num_words = len(words)  # 计算单词列表的长度
        n_ri = max(1, int(alpha_ri * num_words))  # 计算用于随机插入的单词数量
        a_words = random_insertion(words, n_ri)
        sentence = get_only_chars(' '.join(a_words))
        return sentence

    def aug_with_random_swap(sentence):
        sentence = get_only_chars(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word is not '']  # 移除列表中为空的单词
        num_words = len(words)  # 计算单词列表的长度
        n_rs = n_rs = max(1, int(alpha_rs * num_words))  # 计算用于随机交换的单词数量
        a_words = random_swap(words, n_rs)
        sentence = get_only_chars(' '.join(a_words))
        return sentence

    def aug_with_random_deletion(sentence):
        sentence = get_only_chars(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word is not '']  # 移除列表中为空的单词
        num_words = len(words)  # 计算单词列表的长度
        n_rs = n_rs = max(1, int(alpha_rs * num_words))  # 计算用于随机交换的单词数量
        a_words = random_deletion(words, n_rs)
        sentence = get_only_chars(' '.join(a_words))
        return sentence

    for _ in range(num_new_per_technique):
        original_row = original_df.iloc[random.randint(0, original_rows - 1)]  # 选择一个原始样本
        augmented_row = original_row.copy()  # 复制原始行以创建增强行
        augmented_row['prompt'] = aug_with_synonym_replacement(original_row['prompt'])

        for choice in ["A", "B", "C", "D", "E"]:
            augmented_row[choice] = aug_with_synonym_replacement(original_row[choice])

        augmented_data.append(augmented_row)

    for _ in range(num_new_per_technique):
        original_row = original_df.iloc[random.randint(0, original_rows - 1)]  # 选择一个原始样本
        augmented_row = original_row.copy()  # 复制原始行以创建增强行
        augmented_row['prompt'] = aug_with_random_insert(original_row['prompt'])

        for choice in ["A", "B", "C", "D", "E"]:
            augmented_row[choice] = aug_with_random_insert(original_row[choice])

        augmented_data.append(augmented_row)

    for _ in range(num_new_per_technique):
        original_row = original_df.iloc[random.randint(0, original_rows - 1)]  # 选择一个原始样本
        augmented_row = original_row.copy()  # 复制原始行以创建增强行
        augmented_row['prompt'] = aug_with_random_swap(original_row['prompt'])

        for choice in ["A", "B", "C", "D", "E"]:
            augmented_row[choice] = aug_with_random_swap(original_row[choice])

        augmented_data.append(augmented_row)

    for _ in range(num_new_per_technique):
        original_row = original_df.iloc[random.randint(0, original_rows - 1)]  # 选择一个原始样本
        augmented_row = original_row.copy()  # 复制原始行以创建增强行
        augmented_row['prompt'] = aug_with_random_deletion(original_row['prompt'])

        for choice in ["A", "B", "C", "D", "E"]:
            augmented_row[choice] = aug_with_random_deletion(original_row[choice])

        augmented_data.append(augmented_row)

    shuffle(augmented_data)

    return augmented_data
