import torch
import numpy as np
import pandas as pd
import random
import tqdm
import os
import gc
from dataclasses import dataclass, field
from typing import Optional
from data import preprocess, OptionShuffle, DataCollatorForMultipleChoice, tokenizer
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, PeftConfig, TaskType, \
    PeftModelForSequenceClassification
from transformers import (
    AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorWithPadding
)
from colorama import Fore, Back, Style
from transformers import pipeline
from trl import RewardTrainer
from utils import set_seed, get_top_3_winners, MAP_at_3
from data import preprocess_function
from config import CFG


###############################################################
##################    Augmentation    #########################
###############################################################
def augment_fn(df):
    new_df = []
    for index, row in df.iterrows():
        shuffled_row = OptionShuffle(row)
        new_df.append(shuffled_row)
    new_df = pd.concat(new_df, axis=1)
    return new_df.T


# Initialize the pipeline for masked language modeling using BERT
mlm_fill_mask = pipeline(task="fill-mask", model="model/bert-base-uncased")


# Function to perform data augmentation using different techniques
def augmentation_data(original_df, num_augmented_rows):
    augmented_data = []  # 用于存储增强后的数据
    original_rows = original_df.shape[0]  # 原始数据集的行数

    # Function for contextual word embeddings augmentation
    # 上下文词嵌入增强功能
    def contextual_embeddings(text):
        # Tokenize the text
        tokenized_text = tokenizer(text, return_tensors="pt")

        # Find masked positions in the tokenized text
        # 在被分词的文本中寻找mask标记
        masked_positions = [i for i, token in enumerate(tokenized_text["input_ids"][0]) if
                            token == tokenizer.mask_token_id]

        # If no masked positions found, return the original text
        if not masked_positions:
            return text

        # Randomly select one of the masked positions
        # 随机选择一个mask位置
        random_masked_position = random.choice(masked_positions)

        # Predict the masked word using masked language modeling
        # 使用掩码语言建模来预测掩码词
        masked_text = text.replace("[MASK]", tokenizer.mask_token)
        # 利用MLM pipline预测掩码文本中被掩码的词语,并从预测结果中获取预测的词语的字符串形式
        predicted_word = mlm_fill_mask(masked_text)[0]["token_str"]

        # Replace the masked word in the text with the predicted word
        # 这个预测的词语将被用于替换原始文本中的掩码
        augmented_text = text.replace(tokenizer.mask_token, predicted_word, 1)

        return augmented_text

    # Function for synonym replacement augmentation
    def augment_with_synonyms(text):
        # 你可以在这里使用自己的同义词替换逻辑
        # 为了简单起见，假设我们有一个预定义的同义词字典
        synonym_dict = {
            "good": ["excellent", "great", "superb", "fine"],
            "bad": ["poor", "terrible", "awful", "horrible"]
            # 根据需要添加更多的同义词
        }

        words = text.split()
        # 对每个词语进行同义词替换
        augmented_tokens = []
        for token in words:
            if token in synonym_dict:
                # 如果词语在同义词字典中,随机选择一个同义词进行替换
                synonym = random.choice(synonym_dict[token])
                augmented_tokens.append(synonym)
            else:
                augmented_tokens.append(token)
        # 将替换后的词语重新组合为增强后的文本
        augmented_text = " ".join(augmented_tokens)
        return augmented_text

    for _ in range(num_augmented_rows):
        original_row = original_df.iloc[random.randint(0, original_rows - 1)]  # 选择一个原始样本
        augmented_row = original_row.copy()  # 复制原始行以创建增强行

        # Apply augmentation techniques to "prompt"
        # 对"prompt"应用上下文词嵌入增强
        augmented_row["prompt"] = contextual_embeddings(original_row["prompt"])

        # Apply synonym replacement to answer choices (A, B, C, D, E)
        # 答案选项 (A, B, C, D, E) 应用同义词替换
        for choice in ["A", "B", "C", "D", "E"]:
            augmented_row[choice] = augment_with_synonyms(original_row[choice])

        augmented_data.append(augmented_row)

    return augmented_data


def generate_new_dataframe(df):
    new_rows = []

    # Iterate through each row in the original DataFrame
    for _, row in df.iterrows():
        prompt = row['prompt']
        answer = row['answer']
        chosen_option = row[row['answer']]  # Get the text of the chosen option based on the 'answer' column

        # Iterate through each option
        for option in ['A', 'B', 'C', 'D', 'E']:
            if option != row['answer']:
                rejected_option = row[option]  # Get the text of the rejected option
                new_row = {'chosen': prompt + ' ' + chosen_option, 'rejected': prompt + ' ' + rejected_option, 'answer': answer}
                new_rows.append(new_row)

    # Create a new DataFrame from the new_rows list
    new_df = pd.DataFrame(new_rows)
    return new_df


###############################################################
##################    Train/Valid Dataset   ###################
###############################################################
def get_datasets(df, ext_df, fold):
    """
        1. external data as training dataset、original data as valid dataset
        2. merge external data and original data and do split
        3. first, choice some data from original data as valid dataset, and the spare original data and external data as training dataset
    """
    if CFG.only_valid:
        train_df = ext_df
        valid_df = df
        # valid_labels = valid_df['answer']
        train_dataset = Dataset.from_pandas(train_df)
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
        )
        valid_dataset = Dataset.from_pandas(valid_df)
        valid_dataset = valid_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
        )
        train_dataset = train_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= 256
                      and len(x["input_ids_rejected"]) <= 256
        )
        valid_dataset = valid_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= 2048
                      and len(x["input_ids_rejected"]) <= 2048
        )
        return train_dataset, valid_dataset, valid_df
    else:
        train_df = df.query("fold!=@fold")  # Get training fold data
        if CFG.is_merage:
            if CFG.external_data:
                train_df = pd.concat([train_df, ext_df], axis=0)  # Add external data texts
                train_df = train_df.reset_index(drop=True)
        train_dataset = Dataset.from_pandas(train_df)
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
        )

        valid_df = df.query("fold==@fold")  # Get validation fold data
        # valid_labels = valid_df['answer']
        valid_dataset = Dataset.from_pandas(valid_df)
        valid_dataset = valid_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
        )

        return train_dataset, valid_dataset, valid_df


def main():
    set_seed(CFG.seed)
    ###############################################################
    ##################    Processing    ###########################
    ###############################################################
    df_train = pd.read_csv("./data/train.csv")
    df_train = df_train.drop(columns="id")
    df_train.dropna(inplace=True)
    df_train = df_train.reset_index(drop=True)

    ext_df = pd.concat([
        pd.read_csv("data/6000_train_examples.csv"),
        pd.read_csv("./data/llm-science-3k-data-test.csv"),
        pd.read_csv("./data/stem_1k_v1.csv")
    ])
    ext_df = ext_df.drop(columns="id")
    ext_df = ext_df.dropna().reset_index(drop=True)

    # cleaning the data by getting rid of weird characters
    df_train = df_train.replace('[^ -~]+', '', regex=True)  # clean data; get rid of weird characters
    df_train = df_train.replace('_', '', regex=True)  # remove underscores
    df_train = df_train.replace('\d+', '', regex=True)  # remove numbers

    ext_df = ext_df.replace('[^ -~]+', '', regex=True)
    ext_df = ext_df.replace('_', '', regex=True)
    ext_df = ext_df.replace('\d+', '', regex=True)

    df_train = generate_new_dataframe(df_train)
    ext_df = generate_new_dataframe(ext_df)

    ###############################################################
    ##################    Data Split    ###########################
    ###############################################################
    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)  # Initialize K-Fold
    if CFG.is_merage:
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

    df = df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    ext_df = ext_df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)

    ###############################################################
    ##################    Training    #############################
    ###############################################################
    cv_list = []
    for fold in CFG.selected_folds:
        train_dataset, valid_dataset, valid_df = get_datasets(df, ext_df, fold=fold)
        train_dataset = train_dataset
        valid_dataset = valid_dataset
        model = AutoModelForSequenceClassification.from_pretrained(CFG.model_path, num_labels=1, device_map="auto")
        peft_config = LoraConfig(
            r=8, lora_alpha=CFG.lora_alpha, task_type=TaskType.SEQ_CLS, lora_dropout=CFG.lora_dropout,
            bias="none", inference_mode=False, target_modules=["query_proj", "value_proj"]
        )
        model = get_peft_model(model, peft_config)

        training_args = TrainingArguments(
            output_dir='./save_checkpoints',  # 模型保存的文件夹
            overwrite_output_dir=True,
            warmup_ratio=0.1,
            lr_scheduler_type='cosine',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=CFG.epochs,
            gradient_accumulation_steps=2,  # 每两个批次更新一次梯度
            learning_rate=2e-4,
            remove_unused_columns=False,  # 在训练期间删除未使用的列
            optim="adafactor",  # 优化器类型
            logging_steps=250,  # 每隔250个训练步骤记录一次日志
            eval_steps=250,  # 每个250个训练步骤进行一次评估
            evaluation_strategy='steps',
            load_best_model_at_end=True,
            save_strategy='steps',
            save_total_limit=2,  # 最多保存两个模型
            fp16=True,  # 是否使用混合精度训练
            bf16=False,  # 是否使用 Brain Floating Point 16（BF16）精度
            weight_decay=CFG.weight_decay,  # 权重衰减参数
            report_to="none",
        )

        trainer = RewardTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            peft_config=peft_config,
            max_length=256,
        )

        model.config.use_cache = False
        trainer.train()

        preds = []
        for _, row in tqdm(valid_df.iterrows()):
            prompt = row['promot']
            response_options = [
                row['A'],
                row['B'],
                row['C'],
                row['D'],
                row['E']
            ]
            top_3_winners = get_top_3_winners(trainer.model, tokenizer, prompt, response_options)
            preds.append(top_3_winners)
        final_preds = [' '.join(pred) for pred in preds]

        valid_map3 = MAP_at_3(final_preds, valid_df['answer'])
        print(f"{Fore.RED}{Style.BRIGHT}Fold {fold}: MAP@3 = {valid_map3:.5f}{Style.RESET_ALL}")
        cv_list.append(valid_map3)
        cv = np.mean(cv_list)
        print(f"{Fore.RED}{Style.BRIGHT}Global MAP@3 = {cv:.5f}{Style.RESET_ALL}")
        gc.collect()
        torch.cuda.empty_cache()
        del model, trainer, train_dataset, valid_dataset


if __name__ == "__main__":
    main()
