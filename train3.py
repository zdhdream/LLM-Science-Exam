import torch
import random
import gc
import pandas as pd
import numpy as np
from utils import map3, compute_metrics, set_seed
from config3 import CFG
from datasets import Dataset
from torch.optim import AdamW
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel, \
    get_cosine_schedule_with_warmup
from data import preprocess, DataCollatorForMultipleChoice, tokenizer, EarlyStoppingCallback, RemoveOptimizerCallback
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, PeftConfig, TaskType, \
    PeftModelForSequenceClassification
from colorama import Fore, Back, Style
from sklearn.model_selection import StratifiedKFold


###############################################################
##################    SettingParameters   #####################
###############################################################
def get_optimizer_params(model):
    if CFG.is_settingParameters1:
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
             'lr': CFG.learning_rate / 2.6},
            # encoder 8~15
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.01,
             'lr': CFG.learning_rate},
            # encoder 16~23
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.01,
             'lr': CFG.learning_rate * 2.6},
            # 设置出了encoder.layer层之外所有的bias, LayerNorm
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], 'weight_decay': 0.0},
            # encoder 0~7 bias and LayerNorm
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': 0.0,
             'lr': CFG.learning_rate / 2.6},
            # encoder 8~15 bias and LayerNorm
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.0,
             'lr': CFG.learning_rate},
            # encoder 16~23 bias and LayerNorm
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.0,
             'lr': CFG.learning_rate * 2.6},
            # set classifier
            {'params': [p for n, p in model.named_parameters() if
                        "deberta" not in n and not any(nd in n for nd in no_decay)], 'lr': CFG.decoder_lr,
             "momentum": 0.99},
        ]
        return optimizer_parameters

    if CFG.is_settingParameters2:
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
    train_dataset = train_dataset.map(preprocess, num_proc=10,
                                      remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    valid_dataset = Dataset.from_pandas(valid_df)
    valid_dataset = valid_dataset.map(preprocess, num_proc=10,
                                      remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    return train_dataset, valid_dataset, valid_labels


def main():
    set_seed(CFG.seed)

    ext_df = pd.read_csv("data/60k-data-with-context-v2/all_12_with_context2.csv")
    print(ext_df.shape)
    train_df = pd.read_csv('data/60k-data-with-context-v2/train_with_context2.csv')
    ext_df = ext_df.drop(columns="source")
    ext_df = ext_df.fillna('').sample(CFG.NUM_TARIN_SAMPLES)
    # ext_df['context'] = ext_df['context'].apply(lambda x: x[:1750])
    ##########################################################
    ext_df = ext_df.drop_duplicates()

    # 删除ext_df中存在于df_train中的row
    values_to_exclude = train_df['prompt'].values
    mask = ext_df['prompt'].isin(values_to_exclude)
    ext_df = ext_df[~mask]

    ext_len = len(ext_df)

    ###############################################################
    ##################    Data Split    ###########################
    ###############################################################
    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)  # Initialize K-Fold
    train_df = train_df.reset_index(drop=True)
    train_df['fold'] = -1
    for fold, [train_idx, val_idx] in enumerate(skf.split(train_df, train_df['answer'])):
        train_df.loc[val_idx, 'fold'] = fold

    train_df = train_df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    ext_df = ext_df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)

    ###############################################################
    ##################    Training    #############################
    ###############################################################
    cv_list = []
    for fold in CFG.selected_folds:
        train_dataset, valid_dataset, valid_label = get_datasets(train_df, ext_df, fold=fold)

        model = AutoModelForMultipleChoice.from_pretrained(CFG.model_path)

        if CFG.is_freezingEmbedding1:
            # Freeze the embeddings
            for param in model.base_model.embeddings.parameters():
                param.requires_grad = False

        if CFG.is_freezingEmbedding2:
            if CFG.FREEZE_EMBEDDINGS:
                print('Freezing embeddings.')
                for param in model.deberta.embeddings.parameters():
                    param.requires_grad = False

            if CFG.FREEZE_LAYERS > 0:
                print(f'Freezing {CFG.FREEZE_LAYERS} layers.')
                for layer in model.deberta.encoder.layer[:CFG.FREEZE_LAYERS]:
                    for param in layer.parameters():
                        param.requires_grad = False

        if CFG.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=CFG.r,
                lora_alpha=CFG.lora_alpha,
                lora_dropout=CFG.lora_dropout

            )

            lora_model = get_peft_model(model, peft_config=peft_config)

        if CFG.use_SelfParameters:
            optimizer_grouped_parameters = get_optimizer_params(model)
            optimizer = AdamW(optimizer_grouped_parameters, lr=CFG.learning_rate,
                              weight_decay=CFG.weight_decay, eps=1e-6,
                              betas=(0.9, 0.999))  # eps=1e-6, betas=(0.9, 0.999)

            # Create a cosine learning rate scheduler
            num_training_steps = ext_len // (CFG.per_device_train_batch_size * 2) * CFG.epochs
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=CFG.warmup_ratio * num_training_steps,
                                                        num_training_steps=num_training_steps)

            training_args = TrainingArguments(
                # warmup_ratio=CFG.warmup_ratio,
                # learning_rate=CFG.learning_rate,
                # weight_decay=CFG.weight_decay,
                per_device_train_batch_size=CFG.per_device_train_batch_size,
                per_device_eval_batch_size=CFG.per_device_eval_batch_size,
                num_train_epochs=CFG.epochs,
                report_to='none',
                gradient_accumulation_steps=CFG.gradient_accumulation_steps,
                output_dir=CFG.output_dir,
                evaluation_strategy="steps",
                logging_steps=25,
                eval_steps=25,
                save_strategy="steps",
                save_steps=25,
                save_total_limit=2,
                load_best_model_at_end=True,
                seed=CFG.seed,
                fp16=True,
                lr_scheduler_type='cosine',
                metric_for_best_model='eval_loss'
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
                optimizers=[optimizer, scheduler],

            )

            trainer.train()
        else:
            training_args = TrainingArguments(
                warmup_ratio=CFG.warmup_ratio,
                learning_rate=CFG.learning_rate,
                weight_decay=CFG.weight_decay,
                per_device_train_batch_size=CFG.per_device_train_batch_size,
                per_device_eval_batch_size=CFG.per_device_eval_batch_size,
                num_train_epochs=CFG.epochs,
                report_to='none',
                gradient_accumulation_steps=CFG.gradient_accumulation_steps,
                output_dir=CFG.output_dir,
                evaluation_strategy="steps",
                save_strategy="steps",
                save_total_limit=2,
                load_best_model_at_end=True,
                seed=CFG.seed,
                fp16=True,
                lr_scheduler_type='cosine',
                metric_for_best_model='eval_loss',
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
