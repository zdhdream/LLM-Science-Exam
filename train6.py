import re
from typing import List
import math
import html
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from datasets import Dataset
from dataclasses import dataclass
import torch
from config6 import CFG
from typing import Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, PeftConfig, TaskType, \
    PeftModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import transformers
import numpy as np
import warnings

cfg = CFG
# we split individual characters inside special tokens like [START_DNA]
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"


def get_optimizer_params(model):
    if cfg.is_settingParameters1:
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
                        "deberta" not in n and not any(nd in n for nd in no_decay)], 'lr': cfg.decoder_lr,
             "momentum": 0.99},
        ]
        return optimizer_parameters

    if cfg.is_settingParameters2:
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


def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)


def sanitize_text(input_text, new_doc=True):
    """
    Apply custom preprocessing to input texts and tokenize them.

    Returns
    -------
        input_text : list[str]  context + question + 5*question
            Texts to be tokenized
        new_doc : bool
            If True, prepends the end-of-document (</s>) token to each sequence and fixes
            padding.
    """
    text = escape_custom_split_sequence(input_text)
    if not text:
        warnings.warn(
            "Found an empty input text. Changing to end-of-document token instead.",
            UserWarning
        )
        text = "</s>"

    if new_doc:
        pad_token = "<pad>"
        text = pad_token + text

    return text


option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
index_to_option = {v: k for k, v in option_to_index.items()}
choice_prefixes = ["A", "B", "C", "D", "E"]  # A-Z


def format_options(options, choice_prefixes):
    return ' '.join([f'({c}) {o}' for c, o in zip(choice_prefixes, options)])  # 将五个选择拼接到一起


def format_prompt(r, choice_prefixes):
    options = format_options(r.loc[choice_prefixes], choice_prefixes)
    options_len = len(options)
    prompt_len = len(r["prompt"])
    context_len = cfg.MAX_LEN - prompt_len - options_len - 1
    if context_len < 0:
        prompt = f'''Context: None\nQuestion: {r["prompt"]}\nOptions:{options}\nAnswer:'''
    else:
        prompt = f'''Context: {r["context"][:context_len]}\nQuestion: {r["prompt"]}\nOptions:{options}\nAnswer:'''
    return sanitize_text(prompt)


def format_response(r):
    return option_to_index[r['answer']]


def convert_dataset(ds):
    prompts = []
    labels = []
    for _, i in ds.iterrows():
        prompts.append(format_prompt(i, choice_prefixes))
        labels.append(format_response(i))

    df = pd.DataFrame.from_dict({'prompt': prompts, 'answer': labels})  # 这里的prompt包含context+question+5*question
    return df


def set_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # setup padding
    tokenizer.pad_token_id = 1
    tokenizer.pad_token = "<pad>"
    tokenizer.padding_side = "left"
    # setup truncation
    tokenizer.truncation_side = "left"
    # setup special tokens
    tokenizer.bos_token_id = 0
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token_id = 2
    tokenizer.eos_token = "</s>"
    tokenizer.unk_token = "<unk>"
    tokenizer.unk_token_id = 3
    return tokenizer


response_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def preprocess(row):
    output = tokenizer(row["prompt"])
    output["label"] = row["answer"]
    return output


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # print(features)
        labels = [feature.pop("label") for feature in features]
        # _ = [feature.pop("answer") for feature in features]
        _ = [feature.pop("token_type_ids") for feature in features]
        # _ = [feature.pop("response") for feature in features]
        # _ = [feature.pop("prompt") for feature in features]
        batch_size = len(features)

        flattened_features = [[{k: v for k, v in feature.items()} for feature in features]]
        flattened_features = sum(flattened_features, [])
        # print(flattened_features)
        batch = self.tokenizer.pad(flattened_features, padding=self.padding, max_length=self.max_length,
                                   pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt", )
        batch = {k: v.view(batch_size, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


# https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/435602
def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1 * np.array(predictions), axis=1)[:, :3]
    for x, y in zip(pred, labels):
        z = [1 / i if y == j else 0 for i, j in zip([1, 2, 3], x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)


def compute_metrics(p):
    # print(p)
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}


tokenizer = set_tokenizer(cfg.model_name)


def main():
    train_df = pd.read_csv("data/Split-60k-Data/New_val2_recall_info.csv")
    ext_df = pd.read_csv("data/Split-60k-Data/New_train2_recall_info.csv")
    # test_df = pd.read_csv("data/Split-60k-Data/test_context.csv")
    valid_df = pd.read_csv('data/train_with_context2.csv')
    # test_df = test_df[['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer']]
    train_df = pd.concat([
        train_df,
        valid_df
    ])
    ext_df = ext_df.fillna('')
    ext_df = ext_df.drop_duplicates()

    ext_len = len(ext_df)

    ext_df = ext_df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    val_df = valid_df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)

    processed_train_df = convert_dataset(ext_df)
    processed_val_df = convert_dataset(val_df)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=5)

    if cfg.is_freezingEmbedding1:
        # Freeze the embeddings
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False

    if cfg.is_freezingEmbedding2:
        if CFG.FREEZE_EMBEDDINGS:
            print('Freezing embeddings.')
            for param in model.deberta.embeddings.parameters():
                param.requires_grad = False

        if cfg.FREEZE_LAYERS > 0:
            print(f'Freezing {cfg.FREEZE_LAYERS} layers.')
            for layer in model.deberta.encoder.layer[:cfg.FREEZE_LAYERS]:
                for param in layer.parameters():
                    param.requires_grad = False

    if cfg.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=CFG.r,
            lora_alpha=CFG.lora_alpha,
            lora_dropout=CFG.lora_dropout

        )

        lora_model = get_peft_model(model, peft_config=peft_config)

    if cfg.use_SelfParameters:
        optimizer_grouped_parameters = get_optimizer_params(model)
        optimizer = AdamW(optimizer_grouped_parameters, lr=CFG.learning_rate,
                          weight_decay=CFG.weight_decay, eps=1e-6,
                          betas=(0.9, 0.999))  # eps=1e-6, betas=(0.9, 0.999)

        # Create a cosine learning rate scheduler
        num_training_steps = cfg.epoch * (ext_len // (CFG.per_device_train_batch_size * 2))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=cfg.warmup_ratio * num_training_steps,
                                                    num_training_steps=num_training_steps)

    model.gradient_checkpointing_enable()

    model.enable_input_require_grads()

    tokenized_train_dataset = Dataset.from_pandas(processed_train_df).map(preprocess,
                                                                          remove_columns=['prompt', 'answer'])
    collate = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    # train_dataloader = torch.utils.data.DataLoader(tokenized_train_dataset, batch_size=2, collate_fn=collate)
    tokenized_val_dataset = Dataset.from_pandas(processed_val_df).map(preprocess, remove_columns=['prompt', 'answer'])

    training_args = TrainingArguments(
        # warmup_ratio=CFG.warmup_ratio,
        # learning_rate=CFG.learning_rate,
        # weight_decay=CFG.weight_decay,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        per_device_eval_batch_size=CFG.per_device_eval_batch_size,
        num_train_epochs=cfg.epoch,
        report_to='none',
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        output_dir=CFG.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        seed=cfg.seed,
        fp16=True,
        lr_scheduler_type='cosine',
        metric_for_best_model='eval_loss'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        optimizers=[optimizer, scheduler]
    )

    trainer.train()


if __name__ == "__main__":
    main()
