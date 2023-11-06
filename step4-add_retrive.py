import os
import re
import gc
import sys
import multiprocessing
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm
from copy import deepcopy
import torch
import blingfire as bf
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import tokenizers
import transformers

print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from pathlib import Path
from glob import glob
import warnings

warnings.simplefilter('ignore')
model = "model/gte-small"
tokenizer = AutoTokenizer.from_pretrained(model)
device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else torch.device('cuda:0')


# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            # self.config.hidden_dropout = 0.
            # self.config.hidden_dropout_prob = 0.
            # self.config.attention_dropout = 0.
            # self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        # if self.cfg.gradient_checkpointing:
        #     self.model.gradient_checkpointing_enable

        self.pool = MeanPooling()
        self.fc_dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        # feature = F.normalize(feature, p=2, dim=1)
        return feature


class TestDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        # text = self.texts[item].replace('[SEP]', '</s>')
        inputs = self.tokenizer(text,
                                max_length=512,
                                pad_to_max_length=True,
                                add_special_tokens=True,
                                return_offsets_mapping=False)

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs


def get_sentences(document):
    """将文档拆分成句子,并返回句子的列表"""
    res = []
    # 每个句子的偏移量
    _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
    for o in sentence_offsets:
        if o[1] - o[0] < 20:
            continue
        sentence = document[o[0]:o[1]]
        res.append(sentence)
    return res


def get_model_feature(model, texts):
    """利用预训练模型从一组文本中提取特征"""
    feature_outs_all = []
    test_dataset = TestDataset(texts)
    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='longest'),
                             num_workers=0, pin_memory=True, drop_last=False)

    # tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tqdm(test_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            feature_outs = model(inputs)  # (bs, hidden_size)
            feature_outs_all.append(feature_outs.cpu())

    feature_outs_all_final = torch.cat(feature_outs_all, dim=0)
    # print(feature_outs_all_final.shape)

    return feature_outs_all_final  # (200, hidden_size)


def read_csv(file_path):
    """读取文件并删除id列"""
    df = pd.read_csv(file_path)
    if 'id' in df.columns:
        df = df.drop(columns='id')
    return df


def split_long_doc(text):
    max_lenth = 128
    window_size = 16
    text_list = [i for i in text.split() if i]
    res = []

    i = 0
    while i + window_size < len(text_list):
        res.append(' '.join(text_list[i:min(i + max_lenth, len(text_list))]))
        i += 64

    return res


TOP_K = 5
N_RECALLS = 30
MAX_SEQ_LEN = 512
MODEL_NAME = "output_simcse_model"


def main():
    test = pd.read_csv("data/dataset_wiki_new_1/dataset_wiki_new_1_balanced.csv")
    final_res = deepcopy(test)
    files = list(map(str, Path("data/wiki-20220301-en-sci").glob("*.parquet")))
    ds = load_dataset("parquet", data_files=files, split="train")
    content_df = pd.DataFrame(ds)
    model = CustomModel(cfg=None, config_path=MODEL_NAME + '/config.pth', pretrained=False)
    state = torch.load(MODEL_NAME + '/model-gte-small_fold0_best.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    model.eval()
    model.to(device)

    content_df['sentence'] = content_df['text'].apply(lambda x: get_sentences(x))
    content_df = content_df.explode('sentence')
    print(f'content_df: {content_df.shape}')
    content_df.reset_index(drop=True, inplace=True)

    test_embedding_list = get_model_feature(model, test['prompt'].values)
    print('question embedding done')
    print(test_embedding_list.shape)

    corpus_embeddings = get_model_feature(model, content_df['sentence'].values)
    print('content embedding done')
    print(corpus_embeddings.shape)

    np.save('output/text_sentence_embedding', corpus_embeddings.cpu().numpy())
    content_df.to_parquet('output/wiki_sci_text_sentence.parquet')


def main1():
    N_RECALLS = 30
    pred_final = []
    pred_text = []
    model = CustomModel(cfg=None, config_path=MODEL_NAME + '/config.pth', pretrained=False)
    state = torch.load(MODEL_NAME + '/model-gte-small_fold0_best.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    model.eval()
    model.to(device)

    # test = pd.read_csv("data/test.csv")
    test = pd.read_csv("data/Split-60k-Data/New_val2.csv")
    test = test.drop(columns=["source", 'Unnamed: 0'])
    final_res = deepcopy(test)

    test_embedding_list = get_model_feature(model, test['prompt'].values)
    print('question embedding done')
    print(test_embedding_list.shape)

    content_df = pd.read_parquet('output/wiki_sci_text_sentence.parquet')
    corpus_embeddings = np.load('output/text_sentence_embedding.npy', allow_pickle=True)

    for idx, row in tqdm(test.iterrows(), total=len(test)):

        query_embedding = test_embedding_list[idx, :]

        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_k = min([N_RECALLS, len(corpus_embeddings)])
        top_results = torch.topk(cos_scores, k=top_k)
        # print(top_results)
        indics = top_results[1].cpu().numpy()

        # threshold = 0.8
        # score_top = top_results[0].cpu().numpy()
        # in_use = np.where(score_top > threshold)
        # indics = indics[in_use]

        # pid = content_dict[lang]['id'][indics]
        try:
            pid = content_df['url'][indics]
            pred_final.append(' '.join(pid))

            pid = content_df['sentence'][indics]
            pred_text.append('<recall_wiki_text>'.join(pid))
        except:
            pred_final.append('')
            pred_text.append('')

    test['recall_ids'] = pred_final
    test['recall_text'] = pred_text

    test['length'] = test['recall_text'].apply(lambda x: len(x.split()))
    print(test['recall_text'].isna().sum())
    prompt_values = test['prompt'].values.tolist()
    corpus = pd.read_parquet('output/wiki_sci_text_sentence.parquet')

    test['recall_text'] = test['recall_text'].apply(lambda x: x.split('<recall_wiki_text>'))
    test = test.explode('recall_text')
    test['recall_sentence'] = test['recall_text'].apply(lambda x: split_long_doc(x))
    test = test.explode('recall_sentence')
    test = test.fillna("")

    sentence_embeddings = get_model_feature(model, test['recall_sentence'].values)
    test.reset_index(drop=True, inplace=True)

    pred_final = []
    N_RECALLS = 10
    prompt_length = len(prompt_values)
    for idx in tqdm(range(prompt_length)):
        query_text = prompt_values[idx]
        # 获取test文件的prompt's embedding
        query_embedding = test_embedding_list[idx, :]
        sentence_embeddings_index = test[test['prompt'] == query_text].index
        cos_scores = util.cos_sim(query_embedding.cuda(), sentence_embeddings[sentence_embeddings_index].cuda())[0]
        top_k = min([N_RECALLS, len(corpus_embeddings)])
        top_results = torch.topk(cos_scores, k=top_k)
        # print(top_results)
        indics = top_results[1].cpu().numpy()

        pid = test['recall_sentence'][sentence_embeddings_index[indics]]
        pred_final.append('<new_recall_wiki_sep>'.join(pid))
    final_res['context'] = pred_final
    final_res.to_csv('data/dataset_wiki_new_1/dataset_wiki_new_1_balanced.csv')


if __name__ == "__main__":
    main1()
