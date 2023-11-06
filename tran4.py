import logging
import os
import argparse
import torch
import random
import gc
import optuna
import pandas as pd
import numpy as np
from utils import map3, compute_metrics, set_seed
from tqdm import tqdm
from config4 import CFG4
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
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, PeftConfig, TaskType, \
    PeftModelForSequenceClassification
from colorama import Fore, Back, Style
from sklearn.model_selection import StratifiedKFold


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


def main():
    set_seed(CFG4.seed)
    device = CFG4.device
    train_df = pd.read_csv("data/llm-science-exam-dataset-w-context/train.csv")
    ext_df = pd.concat([
        pd.read_csv("data/llm-science-exam-dataset-w-context/15k_gpt3.5-turbo.csv"),
        pd.read_csv("data/llm-science-exam-dataset-w-context/5900_examples.csv"),
        pd.read_csv("data/llm-science-exam-dataset-w-context/6000_train_examples.csv"),
        pd.read_csv("data/llm-science-exam-dataset-w-context/extra_train_set.csv"),
        pd.read_csv("data/llm-science-exam-dataset-w-context/stem_1k_v1.csv")
    ])
    # ext_df['context'] = ext_df['context'].apply(lambda x: x.replace(',', '', regex=True))
    ext_df.reset_index(inplace=True, drop=True)
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
    tokenizer.save_pretrained(CFG4.output_dir + '/tokenizer/')
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    ###############################################################
    ##################    Training    #############################
    ###############################################################
    cv_list = []
    for fold in CFG4.selected_folds:
        train_dataset, valid_dataset, valid_label = get_datasets(train_df, ext_df, fold=fold)
        train_dl = DataLoader(
            train_dataset,
            batch_size=CFG4.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True
        )

        val_dl = DataLoader(
            valid_dataset,
            batch_size=CFG4.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=False
        )

        model = CustomModel(CFG4.model_path)
        model_name_sep = CFG4.output_dir.split('/')[-1]
        torch.save(model.config, CFG4.output_dir + f'/{model_name_sep}_config.pth')
        optimizer_grouped_parameters = get_optimizer_params(model)
        optimizer = AdamW(optimizer_grouped_parameters, lr=CFG4.learning_rate,
                          weight_decay=CFG4.weight_decay, eps=1e-6, betas=(0.9, 0.999))

        # Create a cosine learning rate scheduler
        num_training_steps = CFG4.epochs * (ext_len // (CFG4.per_device_train_batch_size * 2))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=CFG4.warmup_ratio * num_training_steps,
                                                    num_training_steps=num_training_steps)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
        awp = AWP(model, optimizer, adv_lr=CFG4.adv_lr, adv_eps=CFG4.adv_eps)
        awp_start_epoch = 1.0
        autocast = torch.cuda.amp.autocast(enabled=CFG4.USE_AMP,
                                           dtype=torch.half)  # dtype is recommended torch.bfloat16 if you use newer GPU
        scaler = torch.cuda.amp.GradScaler(enabled=CFG4.USE_AMP, init_scale=4096)

        for epoch in range(CFG4.epochs):
            model.train()
            total_loss = 0
            best_map3 = 0.
            best_epch = -1
            es_step = 0
            with tqdm(train_dl, leave=True) as pbar:
                optimizer.zero_grad()
                for idx, batch in enumerate(pbar):
                    inp_ids = batch['input_ids'].to(CFG4.device)
                    att_mask = batch['attention_mask'].to(CFG4.device)
                    token_type_ids = batch['token_type_ids'].to(CFG4.device)
                    label = batch['labels'].to(CFG4.device)

                    if epoch >= awp_start_epoch:
                        awp.perturb(inp_ids, att_mask, token_type_ids, label, criterion)

                    with autocast:
                        y_pred = model(input_ids=inp_ids,
                                       attention_mask=att_mask,
                                       token_type_ids=token_type_ids)

                        loss = criterion(y_pred, label.view(-1))

                        total_loss += loss.item()
                        if CFG4.GRAD_ACC > 1:
                            loss = loss / CFG4.GRAD_ACC

                    pbar.set_postfix(
                        OrderedDict(
                            epoch=f'{epoch + (idx + 1) / len(train_dl):.2f}',
                            loss=f'{loss.item() * CFG4.GRAD_ACC:.4f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )

                    scaler.scale(loss).backward()
                    awp.restore()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG4.MAX_GRAD_NORM or 1e9)

                    if (idx + 1) % CFG4.GRAD_ACC == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()

                train_loss = total_loss / len(train_dl)
                print(f'Epoch:{epoch + 1:02d}, train_loss:{train_loss:.4f}')

            # validation
            total_loss = 0
            y_preds = []
            labels = []
            model.eval()
            with tqdm(val_dl, leave=False) as pbar:
                with torch.no_grad():
                    for idx, batch in enumerate(pbar):
                        inp_ids = batch['input_ids'].to(device)
                        att_mask = batch['attention_mask'].to(device)
                        token_type_ids = batch['token_type_ids'].to(device)
                        label = batch['labels'].to(device)

                        with autocast:
                            y_pred = model(input_ids=inp_ids,
                                           attention_mask=att_mask,
                                           token_type_ids=token_type_ids)
                            loss = criterion(y_pred, label.view(-1))
                            total_loss += loss.item()

                        y_pred = y_pred.to(torch.float)

                        y_preds.append(y_pred.cpu())
                        labels.append(label.cpu())

                predictions = torch.cat(y_preds).numpy()
                labels_ids = torch.cat(labels).numpy()

                # y_preds = np.argsort(-y_preds, 1)
                predictions = np.argsort(-predictions, 1)

                eval_predictions = (labels_ids.reshape(-1, 1), predictions.reshape(-1, 5))

                map3 = compute_metrics(eval_predictions)["MAP@3"]

                val_loss = total_loss / len(val_dl)

                print(f'Epoch:{epoch + 1:02d}, val_loss:{val_loss:.4f}, val_map3:{map3:.4f}')

                if map3 > best_map3:
                    best_map3 = map3
                    best_epoch = epoch + 1

                    fname = f'{CFG4.output_dir}/best_model_awp.pt'
                    torch.save(model.state_dict(), fname)

                    es_step = 0

                else:
                    es_step += 1
                    if es_step >= CFG4.EARLY_STOPPING_EPOCH:
                        print('early stopping')
                        break


if __name__ == "__main__":
    main()
