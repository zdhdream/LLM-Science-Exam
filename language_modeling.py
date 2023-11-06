import os
import math
import wandb
import transformers
from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

wandb.login(key="8709e1618ea1b1bcb08eda9caa51225a390f87d0")
block_size = 128
model_checkpoint = "model/deberta-v3-large-hf-weights"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))


def tokenize_function(examples):
    return tokenizer(examples["text"])


def group_texts(examples):
    """将一组文本实例重新组合成一个适合训练的数据格式"""
    # Concatenate all texts.
    concatenated_examples = {k: [] for k in examples.keys()}
    for k in examples.keys():
        for text in examples[k]:
            concatenated_examples[k].extend(text)
    # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    files = list(map(str, Path("data/wiki-20220301-en").glob("*.parquet")))
    datasets = load_dataset("parquet", data_files=files, split="train")

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=12,
                                      remove_columns=["text", "url", "title"])

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=12,
    )

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    model_name = model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        f"{model_name}-finetuned-wiki-sci",
        evaluation_strategy="epoch",
        warmup_ratio=0.1,
        learning_rate=50e-7,
        weight_decay=0.01,
        bf16=False,
        fp16=True,
        save_total_limit=1,
        save_strategy="epoch",
        auto_find_batch_size=True,
        num_train_epochs=20,
        load_best_model_at_end=True,
        per_device_train_batch_size=512,
        # output_dir="./save_checkpoints"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    train_testvalid = lm_datasets.train_test_split(test_size=0.05)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_testvalid["train"],
        eval_dataset=train_testvalid["test"],
        data_collator=data_collator,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")



if __name__ == "__main__":
    main()
