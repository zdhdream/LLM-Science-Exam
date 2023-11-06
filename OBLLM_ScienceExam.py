import os
import gc
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import blingfire as bf
from collections.abc import Iterable
import faiss
from faiss import write_index, read_index
from sentence_transformers import SentenceTransformer
import torch
import ctypes

libc = ctypes.CDLL("libc.so.6")

torch.cuda.is_available()
###############################################################
#############    Code to Sentence Text    #####################
###############################################################
def process_documents(documents: Iterable[str],
                      document_ids: Iterable,
                      split_sentences: bool = True,
                      filter_len: int = 3,
                      disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Main helper function to process documents from the EMR.
    处理EMR文档数据的辅助函数

    :param documents: Iterable containing documents which are strings
    包含多个文档的可迭代对象,每个文档是一个字符串
    :param document_ids: Iterable containing document unique identifiers
    包含每个文档的唯一标识符的可迭代对象
    :param document_type: String denoting the document type to be processed
    :param document_sections: List of sections for a given document type to process
    是否将文档的部分(sentions)进一步划分
    :param split_sentences: Flag to determine whether to further split sections into sentences
    # 过滤掉长度小于指定字符数的句子，默认为3
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    # 是否禁用进度条
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        df = sentencize(df.text.values,
                        df.document_id.values,
                        df.offset.values,
                        filter_len,
                        disable_progress_bar)
    return df


def sectionize_documents(documents: Iterable[str],
                         document_ids: Iterable,
                         disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Obtains the sections of the imaging reports and returns only the
    selected sections (defaults to FINDINGS, IMPRESSION, and ADDENDUM).

    :param documents: Iterable containing documents which are strings
    包含多个文档的可迭代对象
    :param document_ids: Iterable containing document unique identifiers
    包含每个文档的唯一标识符的可迭代对象
    :param disable_progress_bar: Flag to disable tqdm progress bar
    是否禁用进度条
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `offset`
    """
    processed_documents = []
    for document_id, document in tqdm(zip(document_ids, documents), total=len(documents), disable=disable_progress_bar):
        row = {}
        text, start, end = (document, 0, len(document))
        # 将文档的id、text、位置信息存储在row字典中
        row['document_id'] = document_id
        row['text'] = text
        row['offset'] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:  # 如果行数>0,按照id和offset列进行排序
        return _df.sort_values(['document_id', 'offset']).reset_index(drop=True)
    else:
        return _df


def sentencize(documents: Iterable[str],
               document_ids: Iterable,
               offsets: Iterable[tuple[int, int]],
               filter_len: int = 3,
               disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Split a document into sentences. Can be used with `sectionize_documents`
    to further split documents into more manageable pieces. Takes in offsets
    to ensure that after splitting, the sentences can be matched to the
    location in the original documents.
    将文档拆分为句子，并根据句子的位置信息对拆分后的句子进行匹配，然后将拆分后的句子和相关信息存储在一个pd中

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param offsets: Iterable tuple of the start and end indices
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    document_sentences = []
    for document, document_id, offset in tqdm(zip(documents, document_ids, offsets), total=len(documents),
                                              disable=disable_progress_bar):
        try:
            # 将文档拆分为句子,并获取每个句子的位置信息
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1] - o[0] > filter_len:  # 若句子长度大于filter_len
                    sentence = document[o[0]:o[1]]  # 获取句子的内容
                    abs_offsets = (o[0] + offset[0], o[1] + offset[0])  # 获取句子的绝对位置信息
                    row = {}
                    row['document_id'] = document_id
                    row['text'] = sentence
                    row['offset'] = abs_offsets
                    document_sentences.append(row)
        except:
            continue
    return pd.DataFrame(document_sentences)


###############################################################
#####################   Configurations    #####################
###############################################################
MODEL = 'model/gte-small'
DEVICE = 0
MAX_LENGTH = 384
BATCH_SIZE = 32


def main():
    # Load Data
    WIKI_PATH = "data/wikipedia-20230701"
    wiki_files = os.listdir(WIKI_PATH)
    # df_train = pd.read_csv("data/train.csv")
    # trn = pd.read_csv("data/train.csv")
    # stem_df = pd.read_csv("data/stem_1k_v1.csv")
    # stem_df = stem_df.drop(columns="id")
    # trn = pd.concat([
    #     pd.read_csv("data/6000_train_examples.csv"),
    #     pd.read_csv("data/extra_train_set.csv"),
    #     pd.read_csv("data/5900_examples.csv"),
    #     pd.read_csv("data/15k_gpt3.5-turbo.csv"),
    #     stem_df
    # ])
    trn = pd.read_csv('data/validation-500-gpt4-manually-reviewed/val_500_enhanced.csv')
    trn["id"] = range(len(trn))
    trn = trn.drop_duplicates()

    # 删除ext_df中存在于df_train中的row
    # values_to_exclude = df_train['prompt'].values
    # mask = trn['prompt'].isin(values_to_exclude)
    # ext_df = trn[~mask]

    # Load model
    model = SentenceTransformer(MODEL, device="cuda:0")
    model.max_seq_length = MAX_LENGTH
    # 将模型的数据类型设置为半精度浮点数(float16)类型
    model = model.half()

    # Using precomputed index of the Wikipedia 2023-07 dump
    sentence_index = read_index("data/index/gte-small.index")
    trn['answer_all'] = trn.apply(lambda x: " ".join([x['A'], x['B'], x['C'], x['D'], x['E']]), axis=1)
    trn['prompt_answer_stem'] = trn['prompt'] + " " + trn['answer_all']

    # Encoder the prompts 将trn中的所有prompt转化为embedding
    prompt_embeddings = model.encode(trn.prompt_answer_stem.values, batch_size=BATCH_SIZE, device=DEVICE, show_progress_bar=True,
                                     convert_to_tensor=True, normalize_embeddings=True).half()
    prompt_embeddings = prompt_embeddings.detach().cpu().numpy()

    _ = gc.collect()

    ## Get the top 3 pages that are likely to contain the topic of interest 获取每个prompt可能包含感兴趣主题的前 3 个页面
    search_score, search_index = sentence_index.search(prompt_embeddings, 12)

    ## Save memory - delete sentence_index since it is no longer necessary
    del sentence_index
    del prompt_embeddings
    _ = gc.collect()
    # 释放程序中不再使用的内存块
    libc.malloc_trim(0)

    # Load the Wikipedia Index File
    df = pd.read_parquet("data/wikipedia-20230701/wiki_2023_index.parquet", columns=['id', 'file'])

    ## Get the article and associated file location using the index 存储从搜索结果中提取的文章及其文件有关的位置
    wikipedia_file_data = []

    for i, (scr, idx) in tqdm(enumerate(zip(search_score, search_index)), total=len(search_score)):
        ## Get indices by score threshold
        # scr_idx = idx[np.where(scr <= 0.85)]
        scr_idx = idx
        _df = df.loc[scr_idx].copy()
        _df['prompt_id'] = i
        wikipedia_file_data.append(_df)
    wikipedia_file_data = pd.concat(wikipedia_file_data).reset_index(drop=True)
    wikipedia_file_data = wikipedia_file_data[['id', 'prompt_id', 'file']].drop_duplicates().sort_values(
        ['file', 'id']).reset_index(drop=True)

    ## Save memory - delete df since it is no longer necessary
    del df
    _ = gc.collect()
    libc.malloc_trim(0)

    ## Get the full text data
    wiki_text_data = []

    for file in tqdm(wikipedia_file_data.file.unique(), total=len(wikipedia_file_data.file.unique())):
        _id = [str(i) for i in wikipedia_file_data[wikipedia_file_data['file'] == file]['id'].tolist()]
        _df = pd.read_parquet(f"{WIKI_PATH}/{file}", columns=['id', 'text'])

        _df_temp = _df[_df['id'].isin(_id)].copy()  # 得到当前所有id对应的文本
        del _df
        _ = gc.collect()
        libc.malloc_trim(0)
        wiki_text_data.append(_df_temp)
    wiki_text_data = pd.concat(wiki_text_data).drop_duplicates().reset_index(drop=True)
    _ = gc.collect()

    # Split full-text Wikipedia Documents into Sentences
    ## Parse documents into sentences
    processed_wiki_text_data = process_documents(wiki_text_data.text.values, wiki_text_data.id.values)

    ## Get embeddings of the wiki text data
    wiki_data_embeddings = model.encode(processed_wiki_text_data.text, batch_size=BATCH_SIZE, device=DEVICE,
                                        show_progress_bar=True, convert_to_tensor=True,
                                        normalize_embeddings=True).half()
    wiki_data_embeddings = wiki_data_embeddings.detach().cpu().numpy()

    ## Combine all answers
    trn['answer_all'] = trn.apply(lambda x: " ".join([str(x['A']), str(x['B']), str(x['C']), str(x['D']), str(x['E'])]),
                                  axis=1)

    ## Search using the prompt and answers to guide the search
    trn['prompt_answer_stem'] = trn['prompt'] + " " + trn['answer_all']

    question_embeddings = model.encode(trn.prompt_answer_stem.values, batch_size=BATCH_SIZE, device=DEVICE,
                                       show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True).half()
    question_embeddings = question_embeddings.detach().cpu().numpy()

    ## Parameter to determine how many relevant sentences to include
    NUM_SENTENCES_INCLUDE = 22

    ## List containing Question, Choices, Context
    prompt_contexts = []

    ## List containing just Context
    contexts = []

    for r in trn.itertuples():
        prompt_context = ""

        prompt_id = r.id

        prompt_indices = processed_wiki_text_data[processed_wiki_text_data['document_id'].isin(
            wikipedia_file_data[wikipedia_file_data['prompt_id'] == prompt_id][
                'id'].values)].index.values  # 获取与当前prompt相关的id（3个文本以及被切分为多个句子）
        prompt_context += "Question: " + trn.prompt.iloc[prompt_id] + "\n"  # 获取当前数据的prompt

        prompt_context += "Choices:\n"  # 获取当前数据的五个选项
        prompt_context += "(A) " + str(trn.A.iloc[prompt_id]) + "\n"
        prompt_context += "(B) " + str(trn.B.iloc[prompt_id]) + "\n"
        prompt_context += "(C) " + str(trn.C.iloc[prompt_id]) + "\n"
        prompt_context += "(D) " + str(trn.D.iloc[prompt_id]) + "\n"
        prompt_context += "(E) " + str(trn.E.iloc[prompt_id]) + "\n"

        if prompt_indices.shape[0] > 0:
            prompt_context += "Context:\n"
            ## Per Prompt Index
            prompt_index = faiss.index_factory(wiki_data_embeddings.shape[1], "Flat")
            prompt_index.add(wiki_data_embeddings[prompt_indices])

            context = ""

            ## Get the top matches
            ss, ii = prompt_index.search(question_embeddings,
                                         NUM_SENTENCES_INCLUDE)  # 从文档中搜索三条与question_embeddings（包含每条数据的prompt+5options）的有关的句子embedding
            for _s, _i in zip(ss[prompt_id], ii[prompt_id]):
                ## Threshold on the score
                if _s < 2:  # 筛选相似性分数低于 2 的句子，认为相似性分数低于阈值的句子与问题相关度较高
                    context += processed_wiki_text_data.loc[prompt_indices]['text'].iloc[_i] + "\n"
            prompt_context += context

        contexts.append(context)
        prompt_contexts.append(prompt_context)

    trn['context'] = contexts
    trn.to_csv("data/validation-500-gpt4-manually-reviewed/val_500_context.csv", index=False)
    model.cpu()
    del model
    del question_embeddings, wiki_data_embeddings
    _ = gc.collect()
    libc.malloc_trim(0)
    torch.cuda.empty_cache()

    # Open Book Test Taking!
    for i, p in enumerate(prompt_contexts[:10]):
        print(f"Question {i}")
        print(p)
        print()


if __name__ == "__main__":
    main()
