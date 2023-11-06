import torch
import os
import random
from config import CFG1
from config3 import CFG
from config5 import CFG5
from config7 import CFG7
from typing import Optional, Union
from dataclasses import dataclass
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback, TrainerState, TrainerControl
from transformers import AutoTokenizer, DebertaConfig, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy


cfg = CFG5
option_to_index = {option: idx for idx, option in enumerate("ABCDE")}
index_to_option = {v: k for k, v in option_to_index.items()}
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL)


def preprocess(example):
    first_sentence = ["[CLS] " + str(example['context'])] * 5
    second_sentences = [" #### " + str(example['prompt']) + " [SEP] " + str(example[option]) + " [SEP]" for option in
                        'ABCDE']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation='only_first',
                                  max_length=cfg.MAX_INPUT, add_special_tokens=False)
    tokenized_example['label'] = option_to_index[example['answer']]

    return tokenized_example


def preprocess_as_4(example):
    first_sentence = ["[CLS] " + example['context']] * 4
    second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in
                        'ABCD']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation='only_first',
                                  max_length=cfg.MAX_INPUT, add_special_tokens=False)
    tokenized_example['label'] = option_to_index[example['answer']]
    return tokenized_example


def prepare_answering_input(
        example,  # example that contains, context, question, and options
        tokenizer = tokenizer,  # longformer_tokenizer
        max_length=768,
):
    context = example["context"].replace("\n", " ").strip()
    question = example["prompt"].replace("\n", " ").strip()
    options = [example[option].replace("\n", " ").strip() for option in "ABCDE"]
    c_5 = [context] * len(options)
    q_plus_o = [
        " " + tokenizer.bos_token + " " + question + " " + option for option in options
    ]
    tokenized_example = tokenizer(
        c_5,
        q_plus_o,
        truncation="only_first",
        max_length=max_length,
    )
    tokenized_example["label"] = option_to_index[example["answer"].strip()]
    return tokenized_example


def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_j = tokenizer(chosen, truncation=True)
        tokenized_k = tokenizer(rejected, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples


def OptionShuffle(row, prob=0.50, seed=None):
    if random.random() > prob:
        return row

    # 复制样本行
    shuffled_row = row.copy()

    # 设置随机种子
    if seed is not None:
        random.seed(seed)

    # 获取选项列的值
    options = shuffled_row[['A', 'B', 'C', 'D', 'E']].tolist()

    # 获取正确答案
    answer = shuffled_row['answer']
    answer = [index for index, word in enumerate(cfg.Choices) if answer == word]
    answer = options[answer[0]]

    # 打乱选项
    random.shuffle(options)
    # indices = torch.randperm(len(options))

    # new_options = options[indices]

    # 更新选项列的值
    shuffled_row[['A', 'B', 'C', 'D', 'E']] = options

    # 更新正确答案
    shuffled_row['answer'] = cfg.Choices[options.index(answer)]

    return shuffled_row


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


# We don't need the optimizer.pt so we delete it regularly not to run out of space
class RemoveOptimizerCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        if os.path.exists(state.best_model_checkpoint + '/optimizer.pt'):
            os.remove(state.best_model_checkpoint + '/optimizer.pt')
