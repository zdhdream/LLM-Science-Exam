# ====================================================
# CFG
# ====================================================
# ====================================================
class CFG:
    wandb = True
    competition = 'llm-science'
    debug = False
    apex = False
    print_freq = 20
    num_workers = 4
    model = "model/gte-small"
    # model = "microsoft/deberta-v3-base"
    # model = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    # model = 'model/all-mpnet-base-v2'
    # model = "sentence-transformers/all-MiniLM-L6-v2"
    # model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # model = "mpnet_basev2_first_pretrain"
    # model = "output_simcse_model_with_pretrain_sep_epo66_945"
    # model = "bert-large-multilingual-cased"

    gradient_checkpointing = True
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 30
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    layerwise_learning_rate_decay = 0.9
    adam_epsilon = 1e-6

    betas = (0.9, 0.999)
    batch_size = 120
    max_len = 250
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    n_fold = 1
    trn_fold = [0]
    train = True


if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]

import ast
import copy
import gc
import itertools
import json
import math
# ====================================================
# Library
# ====================================================
import os
import pickle
import random
import re
import shutil
import string
import sys
import time
import warnings
from pathlib import Path

import joblib

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy as sp

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import torch
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import (GroupKFold, KFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

print(f"torch.__version__: {torch.__version__}")
# os.system('pip uninstall -y transformers')
# os.system('pip uninstall -y tokenizers')
# os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels transformers')
# os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels tokenizers')
import tokenizers
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn import Parameter
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.utils.data import DataLoader, Dataset

print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from adv_utils import *
from adv_utils import AWP, EMA, FGM, PGD
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else torch.device('cuda:0')
# device = torch.device('cpu')

INPUT_DIR = './data/'
OUTPUT_DIR = './output_simcse_model/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ====================================================
# Utils
# ====================================================

def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores


def get_logger(filename=OUTPUT_DIR + 'train'):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=888)


def display(tmp):
    print(tmp)


# ====================================================
# Data Loading
# ====================================================
df = pd.read_csv('data/ext-data/all-data-gpt3.5/retrive_dataset.csv')  # 正样本
# df = df.drop_duplicates(subset=['text'])
ori_dataset = pd.read_csv('data/ext-data/all-data-gpt3.5/ori_dataset.csv')  # 原始数据集
neg_dataset = pd.read_csv('data/ext-data/all-data-gpt3.5/neg_sample_10.csv')  # 负样本
df['label'] = 1  # 正样本标签为1
dev_ids = np.load('data/ext-data/all-data-gpt3.5/val_id.npy', allow_pickle=True)
train_df = df[-df['url'].isin(dev_ids)]  # 确保训练集中所有的context的来源于验证集中context的来源不同
dev_df = df[df['url'].isin(dev_ids)]  # 将训练集中所有context来源于验证集相同的数据当作验证集

train_df = train_df.reset_index(drop=True)

dev_df_ids = dev_df.url.unique().tolist()  # 获取验证集中经过去重的维基百科url
# 将负样本中原始context的来源ori_url,如果存在于验证集中当作验证集数据
dev_df_neg = neg_dataset[neg_dataset['ori_url'].isin(dev_df_ids)].reset_index(drop=True)
# 随机抽取 5 * len(dev_df) 的数据
dev_df_neg = dev_df_neg.sample(n=5 * len(dev_df))
dev_df_neg = dev_df_neg[['question', 'text', 'url', 'title']]
# dev_df_neg = dev_df_neg.drop_duplicates(subset=['question'])
# 将负样本的标签设置为0
dev_df_neg['label'] = 0
dev_df = dev_df.copy().reset_index(drop=True)
dev_df = pd.concat([dev_df, dev_df_neg])
dev_df = dev_df.sample(frac=1)

print(f"train.shape: {train_df.shape}")
display(train_df.head())
print(f"dev.shape: {dev_df.shape}")
print(f"dev pos.shape: {dev_df[dev_df['label'] == 1].shape}")
display(dev_df.head())

# ====================================================
# tokenizer
# ====================================================
# tokenizer = AutoTokenizer.from_pretrained(CFG.model+'/tokenizer/')
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained(OUTPUT_DIR + 'tokenizer/')
CFG.tokenizer = tokenizer


# ====================================================
# Define max_len
# ====================================================
# lengths = []
# for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
#     length = len(tokenizer(row['topic_text'], add_special_tokens=False)['input_ids'])
#     lengths.append(length)
#     length = len(tokenizer(row['content_text'], add_special_tokens=False)['input_ids'])
#     lengths.append(length)
#
# pd_tmp = pd.DataFrame()
# pd_tmp['Text_len'] = lengths
# print(pd_tmp['Text_len'].describe([.90, .95, .99, .995]))
# LOGGER.info(f"max_len: {CFG.max_len}")


# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    # text = text.replace('[SEP]', '</s>')
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,  # add <SEP>/<EOS>/<STA>
        max_length=CFG.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)  # transfer numpy to tensor
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text_topic = df['topic_text'].values
        self.text_content = df['content_text'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, [self.text_topic[item], self.text_content[item]])
        return inputs


class DevDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text_topic = df['question'].values
        self.text_content = df['text'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs_topic = prepare_input(self.cfg, self.text_topic[item])  # 将数据转换为tensor
        inputs_content = prepare_input(self.cfg, self.text_content[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs_topic, inputs_content, label


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        """
        :param last_hidden_state: (bs, sentence_len, hidden_size)
        :param attention_mask: (bs, sentence_len)
        :return:
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()).float()  # (bs, sentence_len, hidden_size)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)  # (bs, hidden_size)
        sum_mask = input_mask_expanded.sum(1)  # (bs, hidden_size)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask  # (bs, hidden_size)
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
            LOGGER.info(self.config)
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
        outputs = self.model(**inputs)  # (bs, sentence_len, hidden_size)
        last_hidden_states = outputs[0]  # (bs, sentence_len, hidden_size)
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature


# ====================================================
# Loss
# ====================================================

def simcse_sup_loss(feature_topic, feature_content) -> 'tensor':
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """
    y_true = torch.arange(0, feature_topic.size(0), device=device)
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(feature_topic.unsqueeze(1), feature_content.unsqueeze(0), dim=2)
    # # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    # sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    loss = torch.mean(loss)
    return loss


def in_batch_loss(feature_topic, feature_content):
    """计算两个特征矩阵之间的余弦相似度,并基于这些相似度计算损失"""
    cosine_sim = torch.matmul(feature_topic, feature_content.T)  # (bs, bs)
    # 64 * 64 --> batch * batch 其中 cosine_sim[i][j] 第i个topic和第j个content
    # substract margin from all positive samples cosine_sim()
    # 一个填充值为0.3的对角矩阵
    margin_diag = torch.full(
        [feature_topic.shape[0]], fill_value=0.3
    )
    # margin_diag --》batch * batch 元素全是fill_value的一个矩阵

    cosine_sim = cosine_sim - torch.diag(margin_diag).to('cuda:1')

    # scale cosine to ease training converge
    cosine_sim *= 30

    ## batch * batch 

    labels = torch.arange(0, feature_topic.shape[0])
    labels = torch.reshape(labels, shape=[-1]).to('cuda:1')

    # print(cosine_sim.shape)
    # print(labels.shape)

    loss = F.cross_entropy(cosine_sim, labels)

    return loss


# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device,
             valid_loader, valid_labels, best_score, fgm, awp, ema_inst):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    save_step = int(len(train_loader) / 1)

    for step, (inputs_topic, inputs_content, labels) in enumerate(train_loader):
        inputs_topic = collate(inputs_topic)  # 将所有的question进行截断
        for k, v in inputs_topic.items():
            inputs_topic[k] = v.to(device)

        inputs_content = collate(inputs_content)
        for k, v in inputs_content.items():
            inputs_content[k] = v.to(device)

        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            feature_topic = model(inputs_topic)  # 双塔模型
            feature_content = model(inputs_content)
            # print(feature.shape)
            loss = in_batch_loss(feature_topic, feature_content)
            # print(loss)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        # ---------------------fgm-------------
        # fgm.attack(epsilon=1.0)  # embedding被修改了
        # with torch.cuda.amp.autocast(enabled=CFG.apex):
        #     feature_topic = model(inputs_topic)
        #     feature_content = model(inputs_content)
        #     loss_avd = simcse_sup_loss(feature_topic, feature_content)
        # if CFG.gradient_accumulation_steps > 1:
        #     loss_avd = loss_avd / CFG.gradient_accumulation_steps
        # losses.update(loss_avd.item(), batch_size)
        # scaler.scale(loss_avd).backward()
        # fgm.restore()  # 恢复Embedding的参数
        # ---------------------fgm-------------

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

            if ema_inst:
                ema_inst.update()

            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        if CFG.wandb and step % 40 == 0:
            print({f"[fold{fold}] loss": losses.val,
                   f"[fold{fold}] lr": scheduler.get_lr()[0]})

        if (step + 1) % save_step == 0 and epoch > -1:
            if ema_inst:
                ema_inst.apply_shadow()

            # eval
            score = valid_fn(valid_loader, model, criterion, device)
            # # scoring
            # score, scores = get_score(valid_labels, predictions)

            LOGGER.info(f'Epoch {epoch + 1} - step: {step:.4f}  score: {score:.4f}')
            if CFG.wandb:
                print({f"[fold{fold}] epoch": epoch + 1,
                       f"[fold{fold}] score": score,
                       f"[fold{fold}] best_score": best_score})

            if score >= best_score:
                best_score = score
                LOGGER.info(f'Epoch {epoch + 1} - Save Best loss: {best_score:.4f} Model')
                torch.save({'model': model.state_dict()},
                           # 'predictions': predictions},
                           OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

            if ema_inst:
                ema_inst.restore()

    return losses.avg, best_score


from scipy import stats


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])

    start = time.time()
    for step, (inputs_topic, inputs_content, labels) in enumerate(valid_loader):
        inputs_topic = collate(inputs_topic)
        for k, v in inputs_topic.items():
            inputs_topic[k] = v.to(device)

        inputs_content = collate(inputs_content)
        for k, v in inputs_content.items():
            inputs_content[k] = v.to(device)

        labels = labels.to('cpu').numpy()
        with torch.no_grad():
            feature_topic = model(inputs_topic)
            feature_content = model(inputs_content)
            sim = F.cosine_similarity(feature_topic, feature_content, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(labels))
            # sim_tmp = sim.cpu().numpy()
            # print(labels)
            # print(sim_tmp)
            # score_tmp = stats.spearmanr(labels, sim_tmp)

    end = time.time()
    print('Eval cost time : ', end - start)

    score = stats.spearmanr(label_array, sim_tensor.cpu().numpy()).correlation
    return score


def get_optimizer_grouped_parameters(
        model, model_type,
        learning_rate, weight_decay,
        layerwise_learning_rate_decay
):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.config.num_hidden_layers
    layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters


# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    # train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    # valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    # valid_labels = valid_folds[CFG.target_cols].values

    train_folds = train_df
    valid_folds = dev_df
    train_dataset = DevDataset(CFG, train_folds)
    valid_dataset = DevDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)

    # model = CustomModel(cfg=None, config_path=CFG.model + '/config.pth', pretrained=False)
    # state = torch.load(CFG.model + '/mpnet_basev2_first_pretrain_fold0_best.pth',
    #                    map_location=torch.device('cpu'))
    # model.load_state_dict(state['model'])

    torch.save(model.config, OUTPUT_DIR + 'config.pth')
    model.to(device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr,
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            cfg.num_warmup_steps = num_train_steps * 0.05
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles
            )
        return scheduler

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    # criterion = nn.SmoothL1Loss(reduction='mean')
    # #criterion = RMSELoss(reduction="mean")
    criterion = None
    # loss = loss_fn(rep_a=rep_a, rep_b=rep_b, label=label)

    best_score = 0
    fgm = FGM(model)
    awp = None
    ema_inst = EMA(model, 0.999)
    ema_inst.register()

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # train
        valid_labels = None
        avg_loss, best_score = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device,
                                        valid_loader, valid_labels, best_score, fgm, awp, ema_inst)

    # predictions = torch.load(OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
    #                          map_location=torch.device('cpu'))['predictions']

    # valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


if __name__ == '__main__':

    def get_result(oof_df):
        labels = oof_df[CFG.target_cols].values
        preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')


    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):  # 1
            if fold in CFG.trn_fold:  # [0]
                _oof_df = train_loop(train_df, fold)
