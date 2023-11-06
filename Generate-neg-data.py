import pandas as pd
import numpy as np
import random

import torch.cuda
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

MODEL = 'model/all-mpnet-base-v2'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 384
BATCH_SIZE = 1
# tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
# model = AutoModelForMultipleChoice.from_pretrained(CFG.model_path)
# 初始化DeBERTa分词器和模型
model = SentenceTransformer(MODEL, device="cuda:0")
model.max_seq_length = MAX_LENGTH
model = model.half()


def main():
    pos_data = pd.concat([
        pd.read_csv("data/llm-science-exam-dataset-w-context/15k_gpt3.5-turbo.csv"),
        pd.read_csv("data/llm-science-exam-dataset-w-context/5900_examples.csv"),
        pd.read_csv("data/llm-science-exam-dataset-w-context/6000_train_examples.csv"),
        pd.read_csv("data/llm-science-exam-dataset-w-context/extra_train_set.csv"),
        pd.read_csv("data/llm-science-exam-dataset-w-context/stem_1k_v1.csv")
    ])

    neg_data = pd.DataFrame(columns=pos_data.columns)

    n = 5  # 指定采样个数

    # 遍历数据的每一行
    for _, row in pos_data.iterrows():
        current_prompt = row['prompt']
        current_context = row['context']
        if current_context is None:
            continue
        current_context_embedding = model.encode(current_context, batch_size=1, device=DEVICE,
                                                 show_progress_bar=True,
                                                 convert_to_tensor=True, normalize_embeddings=True).half()
        current_context_embedding = current_context_embedding.unsqueeze(0)
        current_context_embedding = current_context_embedding.detach().cpu().numpy()

        train_embeddings = model.encode(pos_data.context.values, batch_size=32, device=DEVICE,
                                        show_progress_bar=True,
                                        convert_to_tensor=True, normalize_embeddings=True).half()
        train_embeddings = train_embeddings.detach().cpu().numpy()

        # 计算当前行'context' 与其他行'context' 之间的相似度
        similarities = cosine_similarity(current_context_embedding, train_embeddings)

        # 获取相似度低于0.5 的行索引
        low_similarity_indices = np.where(similarities < 0.5)[1]

        # 从低相似度的行中随机选择 5 行
        random_indices = random.sample(list(low_similarity_indices), min(n, len(low_similarity_indices)))

        # 遍历选择的行,生成负样本
        for idx in random_indices:
            # 创建新的样本，保持其他列不变，只替换 'prompt' 和 'context'
            new_sample = pos_data.iloc[idx].copy()
            new_sample['prompt'] = current_prompt
            new_sample['context'] = current_context

            # 将新样本添加到负样本数据框中
            neg_data = neg_data.append(new_sample, ignore_index=True)

    # 保存负样本数据框为 CSV 文件
    neg_data.to_csv('data/neg.csv', index=False)


if __name__ == "__main__":
    main()
