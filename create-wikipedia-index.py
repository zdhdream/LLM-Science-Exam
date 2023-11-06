import faiss
import pickle
import pandas as pd
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from sentence_transformers import SentenceTransformer
from transformers.pipelines import pipeline
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import subprocess
from IPython.display import FileLink, display
import re
import scipy

# hf_model = pipeline("feature-extraction", model="model/distilbert-base-cased")
# kw_model = KeyBERT(model=hf_model)
model_name = "model/gte-small/"
sentence_transformer = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
parquet_folder = "data/wikipedia-20230701"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# file_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'number', 'o', 'other', 'p', 'q',
#               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
paraphs_parsed_dataset = load_from_disk("data/270K-Wikipedia-STEM-articles")
modified_texts = paraphs_parsed_dataset.map(lambda example:
                                            {'temp_text':
                                                 f"{example['title']} {example['section']} {example['text']}".replace(
                                                     '\n', " ").replace("'", "")},
                                            num_proc=2)["temp_text"]


# file_names = ['0_to_25000', '25000_to_50000', '50000_to_75000', '75000_to_100000', '100000_to_125000', '125000_to_131049']
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


def split_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences


top_k = 10


# summarizer = pipeline(task="summarization", model='model/t5-small')
# summarizer = pipeline(task="summarization", model='model/t5-small')
def extract_keywords(text):
    # keywords = kw_model.extract_keywords(text)
    # text = [word[0] for word in keywords]
    # result = summarizer(text, max_length=130, min_length=30, do_sample=False)
    documents = split_into_sentences(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    keywords_per_document = []
    for i in range(len(documents)):
        tfidf_row = tfidf_matrix[i].toarray()[0]
        top_indices = np.argsort(tfidf_row)[-top_k:][::-1]
        keywords = [feature_names[idx] for idx in top_indices]
        keywords_per_document.append(keywords)
    sentences = [' '.join(sentence) for sentence in keywords_per_document]
    return ' '.join(sentences)


# MODEL_NAME = "output_simcse_model"
# model = CustomModel(cfg=None, config_path=MODEL_NAME + '/config.pth', pretrained=False)
# state = torch.load(MODEL_NAME + '/model-gte-small_fold0_best.pth', map_location=torch.device('cpu'))
# model.load_state_dict(state['model'])
# model.eval()
# model.to(device)

# for idx, filename in enumerate(file_names):
#     if (idx + 1) >= 1:
#         document_embeddings = []
#
#         print(f"Processing file_id: {idx + 1} - file_name: {filename}.parquet ......")
#
#         parquet_path = os.path.join(parquet_folder, f"{filename}.parquet")
#         df = pd.read_parquet(parquet_path)
#
#         print(df.columns)
#         print("Sample text: ", df.iloc[0]["text"])
#
#         df['text'] = df['text'].apply(extract_keywords)
#         sentences = df["text"].tolist()
#         embeddings = sentence_transformer.encode(sentences, normalize_embeddings=True)
#         # embeddings = get_model_feature(model, sentences)
#         document_embeddings.extend(embeddings)
#
#         del df, sentences
#         # document_embeddings = [embedding.numpy() for embedding in document_embeddings]
#         document_embeddings = np.array(document_embeddings).astype("float32")
#         index = faiss.IndexFlatIP(document_embeddings.shape[1])
#         index.add(document_embeddings)
#         faiss_index_path = f"data/generation_data/wikipedia_embeddings_collection_{idx + 1}_{filename}.index"
#         faiss.write_index(index, faiss_index_path)
#
#         print(f"Faiss index saved to '{faiss_index_path}'")

#
# target_index = faiss.IndexFlatL2(384)
# for idx, filename in enumerate(file_names):
#     index_filename = f'data/generation_data/wikipedia_embeddings_collection_{idx + 1}_{filename}.index'
#     index_to_merge = faiss.read_index(index_filename)
#     num_vectors = index_to_merge.ntotal
#     for i in range(num_vectors):
#         vec = index_to_merge.reconstruct(i).reshape(-1, 384)
#         vec = np.array(vec).astype("float32")
#         target_index.add(vec)
# faiss.write_index(target_index, "data/index/gte-small-summary.index")


chunk_size = 100000
for idx in tqdm(range(0, len(modified_texts), chunk_size)):
    document_embeddings = []
    chunk_modified_text = modified_texts[idx: idx + chunk_size]
    sentences = list(chunk_modified_text)
    embeddings = sentence_transformer.encode(sentences, normalize_embeddings=True)
    document_embeddings.extend(embeddings)
    del sentences
    document_embeddings = np.array(document_embeddings).astype("float32")
    index = faiss.IndexFlatIP(document_embeddings.shape[1])
    index.add(document_embeddings)
    faiss_index_path = f"data/generation_data/wikipedia_embeddings_collection_{idx + 1}_{chunk_size}.index"
    faiss.write_index(index, faiss_index_path)
    print(f"Faiss index save to '{faiss_index_path}'")

target_index = faiss.IndexFlatL2(384)
for idx in tqdm(range(0, len(modified_texts), chunk_size)):
    index_filename = f'data/generation_data/wikipedia_embeddings_collection_{idx + 1}_{chunk_size}.index'
    index_to_merge = faiss.read_index(index_filename)
    num_vectors = index_to_merge.ntotal
    for i in range(num_vectors):
        vec = index_to_merge.reconstruct(i).reshape(-1, 384)
        vec = np.array(vec).astype("float32")
        target_index.add(vec)
faiss.write_index(target_index, "data/index/gte-small-270k.index")
