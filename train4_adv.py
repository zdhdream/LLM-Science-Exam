import logging
import os
import argparse
import torch
import random
import gc
import optuna
import torch.nn.functional as F
import pandas as pd
import numpy as np
from utils import map3, compute_metrics, set_seed
from tqdm import tqdm
from config4 import CFG4
from adv_utils import FGM
from datasets import Dataset
from model import AWP, CustomModel
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
import torch.nn as nn
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel, \
    get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoConfig
from data import preprocess, DataCollatorForMultipleChoice, tokenizer, EarlyStoppingCallback, RemoveOptimizerCallback
from colorama import Fore, Back, Style
from sklearn.model_selection import StratifiedKFold
import math
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
cfg = CFG4


def apk(actual, predicted, k=5):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """

    # requires all elements are unique
    assert (len(np.unique(predicted)) == len(predicted))

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


###############################################################
##################    SettingParameters   #####################
###############################################################
def get_optimizer_params(model):
    if CFG4.is_settingParameters1:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        other = ["pooler", "classifier", "bias", "LayerNorm.bias", "LayerNorm.weight"]

        # Define parameter groups based on DeBERTa-large architecture
        group1 = ['deberta.encoder.layer.0.', 'deberta.encoder.layer.1.', 'deberta.encoder.layer.2.',
                  'deberta.encoder.layer.3.',
                  'deberta.encoder.layer.4.', 'deberta.encoder.layer.5.', 'deberta.encoder.layer.6.',
                  'deberta.encoder.layer.7.']

        group2 = ['deberta.encoder.layer.8.', 'deberta.encoder.layer.9.', 'deberta.encoder.layer.10.',
                  'deberta.encoder.layer.11.',
                  'deberta.encoder.layer.12.', 'deberta.encoder.layer.13.', 'deberta.encoder.layer.14.',
                  'deberta.encoder.layer.15.']

        group3 = ['deberta.encoder.layer.16.', 'deberta.encoder.layer.17.', 'deberta.encoder.layer.18.',
                  'deberta.encoder.layer.19.',
                  'deberta.encoder.layer.20.', 'deberta.encoder.layer.21.', 'deberta.encoder.layer.22.',
                  'deberta.encoder.layer.23.']

        group_all = ['deberta.encoder.layer.0.', 'deberta.encoder.layer.1.', 'deberta.encoder.layer.2.',
                     'deberta.encoder.layer.3.',
                     'deberta.encoder.layer.4.', 'deberta.encoder.layer.5.', 'deberta.encoder.layer.6.',
                     'deberta.encoder.layer.7.',
                     'deberta.encoder.layer.8.', 'deberta.encoder.layer.9.', 'deberta.encoder.layer.10.',
                     'deberta.encoder.layer.11.', \
                     'deberta.encoder.layer.12.', 'deberta.encoder.layer.13.', 'deberta.encoder.layer.14.',
                     'deberta.encoder.layer.15.', 'deberta.encoder.layer.16.', 'deberta.encoder.layer.17.',
                     'deberta.encoder.layer.18.', 'deberta.encoder.layer.19.',
                     'deberta.encoder.layer.20.', 'deberta.encoder.layer.21.', 'deberta.encoder.layer.22.',
                     'deberta.encoder.layer.23.']

        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in other) and not any(nd in n for nd in group_all)],
             'weight_decay': 0.01},
            # encoder 0~7
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': 0.01,
             'lr': CFG4.learning_rate / 2.6},
            # encoder 8~15
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.01,
             'lr': CFG4.learning_rate},
            # encoder 16~23
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.01,
             'lr': CFG4.learning_rate * 2.6},
            # 设置出了encoder.layer层之外所有的bias, LayerNorm
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], 'weight_decay': 0.0},
            # encoder 0~7 bias and LayerNorm
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': 0.0,
             'lr': CFG4.learning_rate / 2.6},
            # encoder 8~15 bias and LayerNorm
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.0,
             'lr': CFG4.learning_rate},
            # encoder 16~23 bias and LayerNorm
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.0,
             'lr': CFG4.learning_rate * 2.6},
            # set classifier
            {'params': [p for n, p in model.named_parameters() if
                        "deberta" not in n and not any(nd in n for nd in no_decay)], 'lr': CFG4.decoder_lr,
             "momentum": 0.99},
        ]
        return optimizer_parameters

    if CFG4.is_settingParameters2:
        optimizer_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "base_model.embeddings" not in n],
                "lr": 1e-5,  # Example learning rate for top layers
                "weight_decay": 0.01,  # Example weight decay
            },

            {
                "params": [p for n, p in model.named_parameters() if "base_model.embeddings" in n],
                "lr": 1e-4,  # Example learning rate for bottom layers
                "weight_decay": 0.001,  # Example weight decay
            },
        ]

        return optimizer_parameters


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


###############################################################
####################    R-drop         ########################
###############################################################
def compute_kl_loss(p, q, pad_mask):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


###############################################################
##################    Train/Valid Dataset   ###################
###############################################################
def get_datasets(df, ext_df, fold):
    train_df = ext_df
    # valid_ext_df = ext_df.query("fold==@fold")
    # valid_df = pd.concat([df, valid_ext_df], axis=0).reset_index(drop=True)
    valid_df = df
    valid_labels = valid_df['answer']
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(preprocess,
                                      remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    valid_dataset = Dataset.from_pandas(valid_df)
    valid_dataset = valid_dataset.map(preprocess,
                                      remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    return train_dataset, valid_dataset, valid_labels


def get_logger(filename=cfg.adv_output + 'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
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


def train_fn_adv(train_loader, model, optimizer, epoch, scheduler, criterion, device):
    model.train()
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    fgm = FGM(model)

    with tqdm(train_loader, leave=True) as pbar:
        for step, batch in enumerate(pbar):
            inp_ids = batch['input_ids'].to(cfg.device)
            att_mask = batch['attention_mask'].to(cfg.device)
            token_type_ids = batch['token_type_ids'].to(cfg.device)
            label = batch['labels'].to(cfg.device)
            batch_size = label.size(0)

            y_pred = model(input_ids=inp_ids,
                           attention_mask=att_mask,
                           token_type_ids=token_type_ids)

            loss = criterion(y_pred, label.view(-1))

            losses.update(loss.item(), batch_size)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG4.MAX_GRAD_NORM or 1e9)
            # adv
            fgm.attack()  # embedding被修改了,攻击对应层的梯度
            # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
            y_adv = model(input_ids=inp_ids,
                          attention_mask=att_mask,
                          token_type_ids=token_type_ids)
            loss_adv = criterion(y_adv, label.view(-1))
            loss_adv.backward()
            fgm.restore()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            scheduler.step()
            end = time.time()
            if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):
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
    return losses.avg


def valid_fn(vaild_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    y_preds = []
    labels = []
    start = end = time.time()
    with tqdm(vaild_loader, leave=True) as pbar:
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                inp_ids = batch['input_ids'].to(device)
                att_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                label = batch['labels'].to(device)

                y_pred = model(input_ids=inp_ids,
                               attention_mask=att_mask,
                               token_type_ids=token_type_ids)
                loss = criterion(y_pred, label.view(-1))

                y_pred = y_pred.to(torch.float)
                y_preds.append(y_pred.cpu())
                labels.append(label.cpu)
                end = time.time()

                if step % cfg.print_freq == 0 or step == (len(pbar) - 1):
                    print('EVAL: [{0}/{1}] '
                          'Elapsed {remain:s} '
                          'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                          .format(step, len(pbar),
                                  loss=losses,
                                  remain=timeSince(start, float(step + 1) / len(pbar))))

        y_preds = torch.cat(y_preds).numpy()
        labels = torch.cat(labels).numpy()
        # y_preds = np.argsort(-y_preds, 1)
        y_preds = np.argsort(-y_preds, 1)

        map3 = mapk(labels.reshape(-1, 1), y_preds.reshape(-1, 5), k=3)

    return losses.avg, map3


def main():
    set_seed(cfg.seed)
    device = cfg.device
    train_df = pd.read_csv("data/Split-60k-Data/New_val2.csv")  # 验证集
    ext_df = pd.read_csv("data/Split-60k-Data/New_train2.csv")  # 训练集
    test_df = pd.read_csv("data/Split-60k-Data/test_context.csv")  # 训练集
    valid_df = pd.read_csv('data/train_with_context2.csv')  # 验证集
    test_df = test_df[['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer']]
    ext_df = ext_df.drop(columns=["source", 'Unnamed: 0'])
    train_df = train_df.drop(columns=["source", 'Unnamed: 0'])
    ext_df = pd.concat([
        ext_df,
        test_df
    ])
    train_df = pd.concat([
        train_df,
        valid_df
    ])
    ext_df = ext_df.fillna('')
    # ext_df['context'] = ext_df['context'].apply(lambda x: x[:1750])
    ##########################################################
    ext_df = ext_df.drop_duplicates()

    # 删除ext_df中存在于df_train中的row
    values_to_exclude = train_df['prompt'].values
    mask = ext_df['prompt'].isin(values_to_exclude)
    ext_df = ext_df[~mask]
    del values_to_exclude, mask

    ext_len = len(ext_df)

    ###############################################################
    ##################    Data Split    ###########################
    ###############################################################
    skf = StratifiedKFold(n_splits=CFG4.num_folds, shuffle=True, random_state=CFG4.seed)  # Initialize K-Fold
    train_df = train_df.reset_index(drop=True)
    train_df['fold'] = -1
    for fold, [train_idx, val_idx] in enumerate(skf.split(train_df, train_df['answer'])):
        train_df.loc[val_idx, 'fold'] = fold

    train_df = train_df.sample(frac=1, random_state=CFG4.seed).reset_index(drop=True)
    ext_df = ext_df.sample(frac=1, random_state=CFG4.seed).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(CFG4.model_path)
    tokenizer.save_pretrained(CFG4.adv_output + '/tokenizer/')
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    ###############################################################
    ##################    Training    #############################
    ###############################################################
    cv_list = []
    for fold in cfg.selected_folds:
        train_dataset, valid_dataset, valid_label = get_datasets(train_df, ext_df, fold=fold)
        train_dl = DataLoader(
            train_dataset,
            batch_size=cfg.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True
        )

        val_dl = DataLoader(
            valid_dataset,
            batch_size=cfg.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=False
        )

        model = CustomModel(cfg.model_path)
        model = model.to(device)
        model_name_seq = cfg.adv_output.split('/')[-1]
        torch.save(model.config, cfg.adv_output + f'{model_name_seq}_config.pth')
        optimizer_grouped_parameters = get_optimizer_params(model)
        optimizer = AdamW(optimizer_grouped_parameters, lr=CFG4.learning_rate,
                          weight_decay=cfg.weight_decay, eps=1e-6, betas=(0.9, 0.999))
        # Create a cosine learning rate scheduler
        num_training_steps = int(len(train_dl) / cfg.per_device_train_batch_size * cfg.epochs)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=cfg.warmup_ratio * num_training_steps,
                                                    num_training_steps=num_training_steps)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
        for epoch in range(cfg.epochs):
            best_map3 = 0.
            start_time = time.time()
            avg_loss = train_fn_adv(train_dl, model, optimizer, epoch, scheduler, criterion, device)
            elapsed = time.time() - start_time

            LOGGER.info(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')

            avg_valid_loss, map3 = valid_fn(val_dl, model, criterion, device)

            if map3 > best_map3:
                best_map3 = map3
                best_epoch = epoch + 1
                LOGGER.info(
                    f'Epoch {epoch + 1} - Save Best Score: {best_map3:.4f} Model'
                )
                fname = f'{cfg.adv_output}/best_model_awp.pt'
                torch.save(model.state_dict(), fname)


if __name__ == "__main__":
    main()
