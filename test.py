from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from faiss import write_index, read_index
import ctypes

libc = ctypes.CDLL("libc.so.6")

MODEL = 'model/sentence-transformers_all-MiniLM-L6-v2'
DEVICE = 0
MAX_LENGTH = 384
BATCH_SIZE = 16


def main():
    vaild_df = pd.read_csv('data/ext-data/merge/train_qa_merge_all.csv')
    options = {
        'option_1': 'A',
        'option_2': 'B',
        'option_3': 'C',
        'option_4': 'D',
        'option_5': 'E',
    }
    vaild_df = vaild_df.rename(
        columns={'question': 'prompt', 'option_1': 'A', 'option_2': 'B', 'option_3': 'C', 'option_4': 'D',
                 'option_5': 'E'})
    vaild_df['answer'] = vaild_df['answer'].map(options)
    vaild_df = vaild_df[['prompt', 'A', 'B', 'C', 'D', 'E', 'answer']]
    vaild_df = vaild_df.sample(frac=1)
    vaild_df = vaild_df[:2000]
    vaild_df = vaild_df.reset_index(drop=True)
    vaild_df.to_csv('data/gpt4_valid.csv')



if __name__ == "__main__":
    main()
