import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
from typing import Optional, Union
import pandas as pd, numpy as np, torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from config5 import CFG5
from data import DataCollatorForMultipleChoice, preprocess, preprocess_as_4


def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1 * np.array(predictions), axis=1)[:, :3]
    for x, y in zip(pred, labels):
        z = [1 / i if y == j else 0 for i, j in zip([1, 2, 3], x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)


def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}


def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k


def MAP_at_3(predictions, true_items):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u].split()
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k + 1) * user_results[k]
    return map_at_3 / U


def main():
    config = CFG5
    # data/60k-data-with-context-v2/train_with_context2.csv
    df_valid = pd.read_csv("data/60k-data-with-context-v2/train_with_context2.csv")
    print('Validation data size:', df_valid.shape)
    # data/60k-data-with-context-v2/all_12_with_context2.csv
    df_train = pd.read_csv('data/60k-data-with-context-v2/all_12_with_context2.csv')
    print(df_train.shape)
    # df_train = df_train.drop(columns="source")
    df_train = df_train.fillna('').sample(config.NUM_TRAIN_SAMPLES)
    print('Train data size:', df_train.shape)

    ##########################################################
    df_train = df_train.drop_duplicates()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL)
    dataset_valid = Dataset.from_pandas(df_valid)
    dataset = Dataset.from_pandas(df_train)
    dataset = dataset.remove_columns(["__index_level_0__"])

    tokenized_dataset_valid = dataset_valid.map(preprocess, num_proc=12,
                                                remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    tokenized_dataset = dataset.map(preprocess, num_proc=12,
                                    remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])

    model = AutoModelForMultipleChoice.from_pretrained(config.MODEL)

    if config.USE_PEFT:
        print('We are using PEFT.')
        from peft import LoraConfig, get_peft_model, TaskType
        peft_config = LoraConfig(
            r=8, lora_alpha=4, task_type=TaskType.SEQ_CLS, lora_dropout=0.1,
            bias="none", inference_mode=False,
            target_modules=["query_proj", "value_proj"],
            modules_to_save=['classifier', 'pooler'],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if config.FREEZE_EMBEDDINGS:
        print('Freezing embeddings.')
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False
    if config.FREEZE_LAYERS > 0:
        print(f'Freezing {config.FREEZE_LAYERS} layers.')
        for layer in model.deberta.encoder.layer[:config.FREEZE_LAYERS]:
            for param in layer.parameters():
                param.requires_grad = False

    training_args = TrainingArguments(
        warmup_ratio=0.1,
        learning_rate=2e-5,  # [2e-5, 2e-4, 2e-6]
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        report_to='wandb',
        output_dir=f'./checkpoints_{config.VER}',
        overwrite_output_dir=True,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=25,
        evaluation_strategy='steps',
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model='map@3',
        lr_scheduler_type='cosine',
        weight_decay=0.01,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset_valid,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.save_model(f'model_v{config.VER}')

    tokenized_test_dataset = Dataset.from_pandas(df_valid).map(
        preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E'])

    test_predictions = trainer.predict(tokenized_test_dataset).predictions

    predictions_as_ids = np.argsort(-test_predictions, 1)
    predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
    predictions_as_string = df_valid['prediction'] = [
        ' '.join(row) for row in predictions_as_answer_letters[:, :3]
    ]

    m = MAP_at_3(df_valid.prediction.values, df_valid.answer.values)
    print('CV MAP@3 =', m)


if __name__ == "__main__":
    main()
