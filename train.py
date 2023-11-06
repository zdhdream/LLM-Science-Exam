import re

import pandas as pd
import numpy as np
import random
import evaluate
import torch
import optuna
from config import CFG1
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
from peft import get_peft_config, get_peft_model, PeftModel, PeftConfig, LoraConfig, TaskType


###############################################################
##################    Train/Valid Dataset   ###################
###############################################################
def get_datasets(df, ext_df, fold):
    if CFG1.only_valid:
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
    else:
        train_df = df.query("fold!=@fold")  # Get training fold data
        if CFG1.is_merage:
            if CFG1.external_data:
                train_df = pd.concat([train_df, ext_df], axis=0)  # Add external data texts
                train_df = train_df.reset_index(drop=True)
        train_dataset = Dataset.from_pandas(train_df)
        train_dataset = train_dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

        valid_df = df.query("fold==@fold")  # Get validation fold data
        valid_labels = valid_df['answer']
        valid_dataset = Dataset.from_pandas(valid_df)
        valid_dataset = valid_dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

        return train_dataset, valid_dataset, valid_labels


def main():
    set_seed(CFG1.seed)
    ###############################################################
    ##################    Processing    ###########################
    ###############################################################
    df_train = pd.read_csv("./data/train.csv")
    df_train = df_train.drop(columns="id")
    df_train.dropna(inplace=True)
    df_train = df_train.reset_index(drop=True)
    stem_df = pd.read_csv("./data/stem_1k_v1.csv")
    stem_df = stem_df.drop(columns="id")
    ext_df = pd.concat([
        pd.read_csv("data/6000_train_examples.csv"),  # 6000
        pd.read_csv("data/extra_train_set.csv"),
        pd.read_csv("data/llm-science-3k-data-test.csv"), # 3000
        # pd.read_csv("data/15k_gpt3.5-turbo.csv"), # 15000
        stem_df # 1000
    ])
    ext_df = ext_df.drop_duplicates()

    # 删除ext_df中存在于df_train中的row
    values_to_exclude = df_train['prompt'].values
    mask = ext_df['prompt'].isin(values_to_exclude)
    ext_df = ext_df[~mask]
    # cleaning the data by getting rid of weird characters
    # df_train = df_train.replace('[^ -~]+', '', regex=True)  # clean data; get rid of weird characters
    df_train = df_train.replace('_', '', regex=True)  # remove underscores
    # df_train = df_train.replace('\d+', '', regex=True)  # remove numbers

    # ext_df = ext_df.replace('[^ -~]+', '', regex=True)
    ext_df = ext_df.replace('_', '', regex=True)
    # ext_df = ext_df.replace('\d+', '', regex=True)

    ###############################################################
    ##################    Augmentation    #########################
    ###############################################################
    if CFG1.is_addition:
        # Calculate the number of rows needed to achieve the desired total rows
        additional_ext_rows = CFG1.desired_rows - ext_df.shape[0]

        # Augment the data
        augmented_ext_data = augmentation_data(ext_df, additional_ext_rows)

        # Convert the augmented data to a dataframe
        augmented_df = pd.DataFrame(augmented_ext_data)

        ext_df = pd.concat([ext_df, augmented_df], ignore_index=True)

    if CFG1.use_eda:
        augmented_eda_date = eda(ext_df, CFG1.eda_rows)
        augmented_eda_df = pd.DataFrame(augmented_eda_date)
        ext_df = pd.concat([ext_df, augmented_eda_df], ignore_index=True)

    if CFG1.use_shuffle_options:
        shuffle_df_train = augment_fn(df_train)
        df_train = pd.concat([df_train, shuffle_df_train], axis=0)
        shuffle_ext_df = augment_fn(ext_df)
        ext_df = pd.concat([ext_df, shuffle_ext_df], axis=0)

    ###############################################################
    ##################    Data Split    ###########################
    ###############################################################
    skf = StratifiedKFold(n_splits=CFG1.num_folds, shuffle=True, random_state=CFG1.seed)  # Initialize K-Fold
    if CFG1.is_merage:
        df_merge = pd.concat([df_train, ext_df], axis=0).reset_index(drop=True)
        df_merge['fold'] = -1
        for fold, [train_idx, val_idx] in enumerate(skf.split(df_merge, df_merge['answer'])):
            df_merge.loc[val_idx, 'fold'] = fold
        df = df_merge
    else:
        df_train = df_train.reset_index(drop=True)  # Reset dataframe index
        df_train["fold"] = -1  # New 'fold' index
        # Assign folds using StratifiedKFold
        for fold, [train_idx, val_idx] in enumerate(skf.split(df_train, df_train['answer'])):
            df_train.loc[val_idx, 'fold'] = fold
        df = df_train

    df = df.sample(frac=1, random_state=CFG1.seed).reset_index(drop=True)
    ext_df = ext_df.sample(frac=1, random_state=CFG1.seed).reset_index(drop=True)

    ###############################################################
    ##################    Training    #############################
    ###############################################################
    # test_df = pd.read_csv("./data/test.csv")
    cv_list = []
    for fold in CFG1.selected_folds:
        train_dataset, valid_dataset, valid_label = get_datasets(df, ext_df, fold=fold)

        model = AutoModelForMultipleChoice.from_pretrained(CFG1.model_path)

        if CFG1.is_freezingEmbedding:
            # Freeze the embeddings
            for param in model.base_model.embeddings.parameters():
                param.requires_grad = False

        if CFG1.use_self_optimizer:
            # Create optimizer and learning rate scheduler
            # Define different learning rates and weight decay for different layers
            optimizer_grouped_parameters = [
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
            optimizer = AdamW(optimizer_grouped_parameters)

            # Create a cosine learning rate scheduler
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=CFG1.warmup_ratio * CFG1.total_training_steps,
                                                        num_training_steps=CFG1.total_training_steps)

        if CFG1.SelfDefine1:
            for i in range(11):  # freezing the first six layers for better generation
                for param in model.deberta.encoder.layer[i].parameters():
                    param.requires_grad = False
            # Applying different learning rates to different layers
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    # set lr, weight_decay except bias LayerNorm.bias LayeNorm.weight
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "lr": 8e-5,
                    "weight_decay": 0.1,
                },

                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "lr": 8e-4,
                    "weight_decay": 0.1,
                },
            ]

            optimizer = AdamW(optimizer_grouped_parameters)
            # Create a cosine learning rate scheduler
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=CFG1.warmup_ratio * CFG1.total_training_steps,
                                                        num_training_steps=CFG1.total_training_steps)
        # if CFG1.SelfDefine2:
        #     optimizer_parameters = get_optimizer_params(model)
        #     optimizer = AdamW(optimizer_parameters)
        #     # Create a cosine learning rate scheduler
        #     scheduler = get_cosine_schedule_with_warmup(optimizer,
        #                                                 num_warmup_steps=CFG1.warmup_ratio * CFG1.total_training_steps,
        #                                                 num_training_steps=CFG1.total_training_steps)

        training_args = TrainingArguments(
            # warmup_ratio=CFG.warmup_ratio,
            # learning_rate=CFG.learning_rate,
            # weight_decay=CFG.weight_decay,
            per_device_train_batch_size=CFG1.per_device_train_batch_size,
            per_device_eval_batch_size=CFG1.per_device_eval_batch_size,
            num_train_epochs=CFG1.epochs,
            report_to='none',
            output_dir=CFG1.output_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            # logging_dir="./log",
            # logging_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            seed=CFG1.seed

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
            # Other Trainer configurations...
            optimizers=(optimizer, scheduler)  # Pass the optimizer and scheduler as a tuple
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
