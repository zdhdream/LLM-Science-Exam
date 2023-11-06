import os.path
import re
import pandas as pd
import numpy as np
import random
import evaluate
import torch
import optuna
from config import CFG
from utils import map3, compute_metrics, set_seed
from datasets import Dataset
from torch.optim import AdamW
from data import preprocess, DataCollatorForMultipleChoice, tokenizer, EarlyStoppingCallback, RemoveOptimizerCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel, \
    get_cosine_schedule_with_warmup
from eda import augment_fn, augmentation_data, eda
from colorama import Fore, Back, Style
from sklearn.model_selection import StratifiedKFold


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
    train_dataset = train_dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
    valid_dataset = Dataset.from_pandas(valid_df)
    valid_dataset = valid_dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
    return train_dataset, valid_dataset, valid_labels


def main():
    set_seed(CFG.seed)
    ###############################################################
    ##################    Processing    ###########################
    ###############################################################
    deberta_v3_large = 'model/deberta-v3-large-hf-weights'
    df_train = pd.read_csv("./data/train.csv")
    df_train = df_train.drop(columns="id")
    df_train.dropna(inplace=True)
    df_train = df_train.reset_index(drop=True)
    stem_df = pd.read_csv("./data/stem_1k_v1.csv")
    stem_df = stem_df.drop(columns="id")
    ext_df = pd.concat([
        pd.read_csv("data/6000_train_examples.csv"),
        pd.read_csv("data/extra_train_set.csv"),
        stem_df
    ])
    ext_df = ext_df.drop_duplicates()
    # 将缺失值替换为NaN
    ext_df = ext_df.fillna(np.nan)
    ext_df = ext_df.dropna().reset_index(drop=True)

    # 删除ext_df中存在于df_train中的row
    values_to_exclude = df_train['prompt'].values
    mask = ext_df['prompt'].isin(values_to_exclude)
    ext_df = ext_df[~mask]

    ###############################################################
    ##################    Data Split    ###########################
    ###############################################################
    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)  # Initialize K-Fold
    df_train.reset_index(drop=True)
    df_train['fold'] = -1
    for fold, [train_idx, val_idx] in enumerate(skf.split(df_train, df_train['answer'])):
        df_train.loc[val_idx, 'fold'] = fold

    train_df = df_train.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    ext_df = ext_df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)

    ###############################################################
    ##################    Training    #############################
    ###############################################################
    model = AutoModelForMultipleChoice.from_pretrained(CFG.model_path)
    cv_list = []
    for fold in CFG.selected_folds:
        train_dataset, valid_dataset, valid_label = get_datasets(train_df, ext_df, fold=fold)

        output_dir = CFG.output_dir + f'/fold_{fold}'
        best_model_dir = CFG.best_dir

        training_args = TrainingArguments(
            warmup_ratio=CFG.warmup_ratio,
            learning_rate=CFG.learning_rate,
            weight_decay=CFG.weight_decay,
            per_device_train_batch_size=CFG.per_device_train_batch_size,
            per_device_eval_batch_size=CFG.per_device_eval_batch_size,
            num_train_epochs=CFG.epochs,
            report_to='none',
            output_dir=CFG.output_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=1,
            load_best_model_at_end=True,
            seed=CFG.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
            callbacks=[RemoveOptimizerCallback()],
        )

        trainer.train()

        valid_pred = trainer.predict(valid_dataset).predictions
        valid_pred_ids = np.argsort(-valid_pred, 1)
        valid_pred_letters = np.array(list('ABCDE'))[valid_pred_ids][:, :3]
        valid_map3 = map3(valid_label, valid_pred_letters)
        print(f"{Fore.RED}{Style.BRIGHT}Fold {fold}: MAP@3 = {valid_map3:.5f}{Style.RESET_ALL}")
        cv_list.append(valid_map3)
        del model, trainer, train_dataset, valid_dataset, valid_label
        cv = np.mean(cv_list)
        print(f"{Fore.RED}{Style.BRIGHT}Global MAP@3 = {cv:.5f}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
