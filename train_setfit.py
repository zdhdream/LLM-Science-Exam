
import json
import os
from datasets import Dataset
import pandas as pd
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitTrainer, SetFitModel

# Can only train on one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# a good sentence embedding model
model_id = "intfloat/e5-small-v2"
model = SetFitModel.from_pretrained(model_id)


# All of the examples in train.csv from Kaggle will get the label "science"
df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv")
train_texts = []
options = list("ABCDE")
for *cols, prompt, answer in df[list("ABCDE") + ["prompt", "answer"]].values:
    train_texts.append(prompt + "\n" + cols[options.index(answer)])
    

with open("/kaggle/working/wiki_samples.json") as fp:
    wiki_samples = json.load(fp)

with open("/kaggle/working/idx2label.json") as fp:
    idx2label = json.load(fp)
    

idxs = list(idx2label.keys())

texts = [wiki_samples[int(i)] for i in idxs] + train_texts
labels = [idx2label[i] for i in idxs] + [1]*len(train_texts)

ds = Dataset.from_dict({"text": texts, "label": labels})
split_ds = ds.train_test_split(0.2)
split_ds["train"].to_json("train.json")
split_ds["test"].to_json("eval.json")

trainer = SetFitTrainer(
    model=model,
    train_dataset=split_ds["train"],
    eval_dataset=split_ds["test"],
    loss_class=CosineSimilarityLoss,
    num_iterations=10, # might get better results by training on more pairs, more epochs
    column_mapping={"text": "text", "label": "label"},
    batch_size=8,
    use_amp=True,
)

# progress bar looks terrible in kaggle
trainer.train(show_progress_bar=False)

model.save_pretrained("setfit-sciwiki")

print(trainer.evaluate())
