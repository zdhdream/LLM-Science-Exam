import faiss
import pickle
import pandas as pd
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from sentence_transformers import SentenceTransformer
import subprocess
from IPython.display import FileLink, display

model = "model/gte-small"
tokenizer = AutoTokenizer.from_pretrained(model)


# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            # self.config.hidden_dropout = 0.
            # self.config.hidden_dropout_prob = 0.
            # self.config.attention_dropout = 0.
            # self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        # if self.cfg.gradient_checkpointing:
        #     self.model.gradient_checkpointing_enable

        self.pool = MeanPooling()
        self.fc_dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        # feature = F.normalize(feature, p=2, dim=1)
        return feature


class TestDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        # text = self.texts[item].replace('[SEP]', '</s>')
        inputs = self.tokenizer(text,
                                max_length=512,
                                pad_to_max_length=True,
                                add_special_tokens=True,
                                return_offsets_mapping=False)

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs


def get_model_feature(model, texts):
    """利用预训练模型从一组文本中提取特征"""
    feature_outs_all = []
    test_dataset = TestDataset(texts)
    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='longest'),
                             num_workers=0, pin_memory=True, drop_last=False)

    # tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tqdm(test_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            feature_outs = model(inputs)  # (bs, hidden_size)
            feature_outs_all.append(feature_outs.cpu())

    feature_outs_all_final = torch.cat(feature_outs_all, dim=0)
    # print(feature_outs_all_final.shape)

    return feature_outs_all_final  # (200, hidden_size)


def download_file(path, file_name):
    os.chdir('data/wiki-20230909-embedding')
    zip = f"data/wiki-20230909-embedding/{file_name}.zip"
    command = f"zip {zip} {path} -r"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Unable to run zip command!")
        print(result.stderr)
        return
    display(FileLink(f'{file_name}.zip'))


MODEL_NAME = "output_simcse_model"
device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else torch.device('cuda:0')
model = CustomModel(cfg=None, config_path=MODEL_NAME + '/config.pth', pretrained=False)
state = torch.load(MODEL_NAME + '/model-gte-small_fold0_best.pth', map_location=torch.device('cpu'))
model.load_state_dict(state['model'])
model.eval()
model.to(device)

parquet_folder = "data/wikipedia-20230701"
faiss_index_path = "data/wiki-20230909-embedding/wikipedia_embeddings.index"

document_embeddings = []
for idx, filename in enumerate(os.listdir(parquet_folder)):
    # number, other and wiki_2023_index files are not what we need
    if filename.endswith(".parquet") and not (
            filename.endswith("number.parquet") or filename.endswith("other.parquet") or filename.endswith(
        "wiki_2023_index.parquet")):
        print(f"Processing file_id: {idx} - file_name: {filename} ......")
        parquet_path = os.path.join(parquet_folder, filename)
        df = pd.read_parquet(parquet_path)
        df.text = df.text.apply(lambda x: x.split("==")[0])  # we trim an article to an abstract in this line
        sentences = df.text.tolist()
        test_embedding_list = get_model_feature(model, sentences)
        # embeddings = sentence_transformer.encode(sentences, normalize_embeddings=True)
        del df, sentences  # free some memory
        document_embeddings.extend(test_embedding_list)
document_embeddings = np.array(document_embeddings)
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)
faiss.write_index(index, faiss_index_path)
print(f"Faiss Index Successfully Saved to '{faiss_index_path}'")
