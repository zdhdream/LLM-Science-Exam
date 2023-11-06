TOP_K = 5
N_RECALLS = 10
MAX_SEQ_LEN = 512


MODEL_NAME = "output_simcse_model"

import warnings
warnings.simplefilter('ignore')

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
# 加载数据

# DATA_DIR = './learning-equality-curriculum-recommendations'

# 验证集 topic_id
#final_name = './external_train_data/stem_dataset_gpt4.csv'
df = pd.read_csv('data/ext-data/retrive_dataset.csv') # 正样本
dev_ids = np.load('data/ext-data/val_id.npy',allow_pickle=True)
dev_df =  df[df['url'].isin(dev_ids)]
#dev_df = pd.read_csv(final_name)
dev_df.reset_index(drop=True, inplace=True)
final_res = deepcopy(dev_df)
files = list(map(str, Path("data/wiki-20220301-en-sci").glob("*.parquet")))
ds = load_dataset("parquet", data_files=files, split="train")
content_df = pd.DataFrame(ds)
# 加载预训练模型

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
        #feature = F.normalize(feature, p=2, dim=1)
        return feature

#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = 'model/gte-small'
tokenizer = AutoTokenizer.from_pretrained(model)

model = CustomModel(cfg=None, config_path=MODEL_NAME + '/config.pth', pretrained=False)
state = torch.load(MODEL_NAME + '/model-gte-small_fold0_best.pth',
                   map_location=torch.device('cpu'))
model.load_state_dict(state['model'])

device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else torch.device('cuda:0')
model.eval()
model.to(device)


class TestDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        # text = self.texts[item].replace('[SEP]', '</s>')
        inputs = tokenizer(text,
                           max_length=512,
                           pad_to_max_length=True,
                           add_special_tokens=True,
                           return_offsets_mapping=False)

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs

def get_model_feature(model, texts):
    feature_outs_all = []
    test_dataset = TestDataset(texts)
    test_loader = DataLoader(test_dataset,
                             batch_size=256,
                             shuffle=False,
                             collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='longest'),
                             num_workers=0, pin_memory=True, drop_last=False)

    # tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tqdm(test_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            feature_outs = model(inputs)
            feature_outs_all.append(feature_outs.cpu())

    feature_outs_all_final = torch.cat(feature_outs_all, dim=0)
    #print(feature_outs_all_final.shape)

    return feature_outs_all_final




topic_embedding_list = get_model_feature(model, dev_df['question'].values)
print('question embedding done')
print(topic_embedding_list.shape)
corpus_embeddings = get_model_feature(model, content_df['text'].values)
# corpus_embeddings = torch.as_tensor(np.load('text_embedding.npy')) #get_model_feature(model, content_df['text'].values)
print('content embedding done')
print(corpus_embeddings.shape)

N_RECALLS = 10
pred_final = []
pred_text = []
for idx, row in tqdm(dev_df.iterrows(), total=len(dev_df)):
    query_embedding = topic_embedding_list[idx, :]

    cos_scores = util.cos_sim(query_embedding.cuda(), corpus_embeddings.cuda())[0]
    top_k = min([N_RECALLS, len(corpus_embeddings)])
    top_results = torch.topk(cos_scores, k=top_k)
    # print(top_results)
    indics = top_results[1].cpu().numpy()

    # threshold = 0.8
    # score_top = top_results[0].cpu().numpy()
    # in_use = np.where(score_top > threshold)
    # indics = indics[in_use]

    # pid = content_dict[lang]['id'][indics]
    pid = content_df['url'][indics]
    pred_final.append(' '.join(pid))

    pid = content_df['text'][indics]
    pred_text.append('<recall_wiki_text>'.join(pid))

dev_df['recall_ids'] = pred_final
dev_df['recall_text'] = pred_text


# 算分环节
dev_df['recall_ids'] = pred_final
df_metric = dev_df.copy()
df_metric['content_ids'] = df_metric['url']


def get_pos_score(y_true, y_pred, top_n):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()[:top_n]))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)

pos_score = get_pos_score(df_metric['content_ids'], df_metric['recall_ids'], 50)
print(f'Our max positive score top 50 is {pos_score}')

pos_score = get_pos_score(df_metric['content_ids'], df_metric['recall_ids'], 70)
print(f'Our max positive score top 70 is {pos_score}')

pos_score = get_pos_score(df_metric['content_ids'], df_metric['recall_ids'], 100)
print(f'Our max positive score top 100 is {pos_score}')

pos_score = get_pos_score(df_metric['content_ids'], df_metric['recall_ids'], 150)
print(f'Our max positive score top 150 is {pos_score}')

pos_score = get_pos_score(df_metric['content_ids'], df_metric['recall_ids'], 200)
print(f'Our max positive score top 200 is {pos_score}')

df_metric['content_ids'] = df_metric['content_ids'].astype(str).apply(lambda x: x.split())
df_metric['recall_ids'] = df_metric['recall_ids'].astype(str).apply(lambda x: x.split())
f2_scores = []

N_RECALLS = [3, 5, 10, 30, 50, 100, 200, 300, 400, 500, 600]
N_TOP_F2 = [5, 10, 15]
# for n_top in N_TOP_F2:
#     for _, row in tqdm(df_metric.iterrows(), total=len(df_metric)):
#         true_ids = set(row['content_ids'])
#         pred_ids = set(row['recall_ids'][:n_top])
#         tp = len(true_ids.intersection(pred_ids))
#         fp = len(pred_ids - true_ids)
#         fn = len(true_ids - pred_ids)
#         if pred_ids:
#             precision = tp / (tp + fp)
#             recall = tp / (tp + fn)
#             f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
#         else:
#             f2 = 0
#         f2_scores.append(f2)
#     print(f'Average F2@{n_top}:', np.mean(f2_scores))
for n_recall in N_RECALLS:
    total = 0
    correct = 0
    for _, row in tqdm(df_metric.iterrows(), total=len(df_metric)):
        y_trues = row['content_ids']
        y_preds = row['recall_ids'][:n_recall]
        for y_true in y_trues:
            total += 1
            if y_true in y_preds:
                correct += 1
    print(f'hitrate@{n_recall}:', correct/total)