import torch
import random
import gc
import optuna
import pandas as pd
import numpy as np
from utils import map3, compute_metrics, set_seed
from config3 import CFG
from config import CFG1
from datasets import Dataset
from torch.optim import AdamW
from pathlib import Path
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel, \
    get_cosine_schedule_with_warmup
from data import preprocess, DataCollatorForMultipleChoice, tokenizer, EarlyStoppingCallback, RemoveOptimizerCallback
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, PeftConfig, TaskType, \
    PeftModelForSequenceClassification
from colorama import Fore, Back, Style
from sklearn.model_selection import StratifiedKFold
from eda import augment_fn, augmentation_data, eda


def model_init(trial):
    return AutoModelForMultipleChoice.from_pretrained("model/deberta-v3-large-hf-weights")


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


###############################################################
##################    Hyperparameter Search   #################
###############################################################
def optuna_hp_space(trial):
    # 定义需要调优的超参数空间
    hyperparameters = {
        # Floating point parameter (log)
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 1e-2, log=True),
        # Floating point parameter (log)
        "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.01, log=True),
        # Floating point parameter (log)
        "warm_up_radio": trial.suggest_float("warmup_ratio", 0.1, 0.8, log=True),
        # Integer parameter(step)
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 2, 10, step=2)
    }
    train_df = pd.read_csv("./data/train_context.csv")
    ext_df = pd.read_csv("./data/ext_train_context.csv")[:1500]
    ext_df["prompt"] = ext_df["context"][:100] + " #### " + ext_df["prompt"]
    ext_df = ext_df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    train_dataset, valid_dataset, valid_label = get_datasets(train_df, ext_df, fold=0)
    model = AutoModelForMultipleChoice.from_pretrained(CFG.model_path)
    training_args = TrainingArguments(
        warmup_ratio=hyperparameters["warm_up_radio"],
        learning_rate=hyperparameters["learning_rate"],
        weight_decay=hyperparameters["weight_decay"],
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        per_device_eval_batch_size=CFG.per_device_eval_batch_size,
        num_train_epochs=CFG.epochs,
        report_to='none',
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
        output_dir=CFG.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        seed=CFG.seed,
        fp16=True,
        lr_scheduler_type='cosine'
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
    return valid_map3


def optuna_hp_space_train1(trial):
    # 定义需要调优的超参数空间
    hyperparameters = {
        # Floating point parameter (log)
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 1e-2, log=True),
        # Floating point parameter (log)
        "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.01, log=True),
        # Floating point parameter (log)
        "warm_up_radio": trial.suggest_float("warmup_ratio", 0.1, 0.8, log=True),
        # Integer parameter(step)
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 2, 10, step=2)
    }

    return hyperparameters


def optuna_hp_space_way2(trial):
    return {
        # Floating point parameter (log)
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 1e-2, log=True),
        # Floating point parameter (log)
        "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.01, log=True),
        # Floating point parameter (log)
        "warm_up_radio": trial.suggest_float("warmup_ratio", 0.1, 0.8, log=True),
        # Integer parameter(step)
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 2, 10, step=2)
    }


def main0():
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_hp_space, n_trials=10)
    # 输出最优的超参数组合和性能指标
    print('Best hyperparameters: {}'.format(study.best_params))
    print('Best performance: {:.4f}'.format(study.best_value))
    best_params = study.best_params


def main1():
    set_seed(CFG.seed)

    df_train = pd.read_csv("./data/train.csv")
    df_train = df_train.drop(columns="id")
    df_train.dropna(inplace=True)
    df_train = df_train.reset_index(drop=True)
    stem_df = pd.read_csv("./data/stem_1k_v1.csv")
    stem_df = stem_df.drop(columns="id")
    ext_df = pd.concat([
        pd.read_csv("data/6000_train_examples.csv"),  # 6000
        pd.read_csv("data/extra_train_set.csv"),
        pd.read_csv("data/llm-science-3k-data-test.csv"),  # 3000
        stem_df  # 1000
    ])
    ext_len = len(ext_df) // 3
    ext_df = ext_df[:ext_len]
    del stem_df
    ext_df = ext_df.drop_duplicates()

    # 删除ext_df中存在于df_train中的row
    values_to_exclude = df_train['prompt'].values
    mask = ext_df['prompt'].isin(values_to_exclude)
    ext_df = ext_df[~mask]
    del values_to_exclude, mask

    if CFG1.use_shuffle_options:
        shuffle_df_train = augment_fn(df_train)
        df_train = pd.concat([df_train, shuffle_df_train], axis=0)
        shuffle_ext_df = augment_fn(ext_df)
        ext_df = pd.concat([ext_df, shuffle_ext_df], axis=0)

    ext_df = ext_df.sample(frac=1, random_state=CFG1.seed).reset_index(drop=True)
    df_train = df_train.sample(frac=1, random_state=CFG1.seed).reset_index(drop=True)
    train_dataset, valid_dataset, valid_label = get_datasets(df_train, ext_df, fold=0)

    # model = AutoModelForMultipleChoice.from_pretrained(CFG1.model_path)
    #
    # if CFG1.is_freezingEmbedding:
    #     # Freeze the embeddings
    #     for param in model.base_model.embeddings.parameters():
    #         param.requires_grad = False
    #
    # if CFG1.use_self_optimizer:
    #     # Create optimizer and learning rate scheduler
    #     # Define different learning rates and weight decay for different layers
    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [p for n, p in model.named_parameters() if "base_model.embeddings" not in n],
    #             "lr": 1e-5,  # Example learning rate for top layers
    #             "weight_decay": 0.01,  # Example weight decay
    #         },
    #
    #         {
    #             "params": [p for n, p in model.named_parameters() if "base_model.embeddings" in n],
    #             "lr": 1e-4,  # Example learning rate for bottom layers
    #             "weight_decay": 0.001,  # Example weight decay
    #         },
    #     ]
    #     optimizer = AdamW(optimizer_grouped_parameters, lr=CFG1.learning_rate,
    #                       weight_decay=CFG1.weight_decay)
    #
    #     # Create a cosine learning rate scheduler
    #     num_training_steps = CFG1.epochs * (ext_len // (CFG1.per_device_train_batch_size * 2))
    #     scheduler = get_cosine_schedule_with_warmup(optimizer,
    #                                                 num_warmup_steps=CFG1.warmup_ratio * num_training_steps,
    #                                                 num_training_steps=num_training_steps)

    training_args = TrainingArguments(
        learning_rate=CFG1.learning_rate,
        weight_decay=CFG1.weight_decay,
        warmup_ratio=CFG1.warmup_ratio,
        per_device_train_batch_size=CFG1.per_device_train_batch_size,
        per_device_eval_batch_size=CFG1.per_device_eval_batch_size,
        num_train_epochs=CFG1.epochs,
        report_to='none',
        output_dir=CFG1.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        seed=CFG1.seed

    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[RemoveOptimizerCallback()],
    )

    trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space_way2,
        n_trials=10,
        compute_objective=compute_metrics
    )


def main2():
    train_df = pd.read_csv("./data/train_context.csv")
    ext_df = pd.read_csv("./data/ext_train_context.csv")[:1500]
    ext_df["prompt"] = ext_df["context"] + " #### " + ext_df["prompt"]
    ext_df = ext_df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    train_dataset, valid_dataset, valid_label = get_datasets(train_df, ext_df, fold=0)

    training_args = TrainingArguments(
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        per_device_eval_batch_size=CFG.per_device_eval_batch_size,
        num_train_epochs=CFG.epochs,
        report_to='none',
        output_dir=CFG.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        seed=CFG.seed

    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[RemoveOptimizerCallback()],
    )

    trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space_way2,
        n_trials=10,
        compute_objective=compute_metrics
    )


if __name__ == "__main__":
    main2()
