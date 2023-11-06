
import os
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=0)
    parser.add_argument("--num_pq_files", type=int, default=-1)
    args = parser.parse_args()
    
    output_dir = f"science_texts"
    os.makedirs(output_dir, exist_ok=True)
    
    accelerator = Accelerator()


    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    batch_size = 64

    
    files = list(map(str, Path("/kaggle/input/wiki-20220301-en").glob("*.parquet")))
    
    if args.num_pq_files > 0:
        files = files[:args.num_pq_files]
    
    ds = load_dataset("parquet", data_files=files, split="train")

    def tokenize(examples, idxs):
        """
        Only need first 500 characters of text.
        Add length to sort and minimize padding
        """
        tokenized =  tokenizer(
            [x[:500] for x in examples["text"]], 
            padding=False, 
            truncation=True, 
            max_length=512,
            )

        tokenized["length"] = [len(t) for t in tokenized.input_ids]
        
        tokenized["idx"] = idxs

        return tokenized

    tds = ds.map(tokenize, batched=True, num_proc=4, with_indices=True)
    tds = tds.sort("length")
    
    # Ignore text with small number of tokens
    tds = tds.filter(lambda x: x["length"] > 100, num_proc=4)

    collator = DataCollatorWithPadding(tokenizer)

    def collate(examples):

        inputs = []
        for x in examples:
            inputs.append({"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]})

        batch = collator(inputs)

        batch["text"] = [x["text"] for x in examples]  
        batch["idx"] = torch.tensor([x["idx"] for x in examples])
        

        return batch
    

    dl = DataLoader(
        tds, 
        collate_fn=collate, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=4, 
        pin_memory=True,
        )

    model, dl = accelerator.prepare(model, dl)


    idxs = []
    count = 0
    with torch.inference_mode():
        for i, batch in tqdm(enumerate(dl)):
            out = model(batch["input_ids"], batch["attention_mask"])
            
            logits = accelerator.gather_for_metrics(out.logits)
            
            batch_idxs = accelerator.gather_for_metrics(batch["idx"])
            
            
            if accelerator.is_main_process:
                
                for (_, pos_proba), idx in zip(logits.softmax(-1), batch_idxs.cpu().tolist()):
                    if pos_proba > 0.95:
                        idxs.append(idx)               
                
            
            # Save every 100
            if (i+1) % 200 == 0:
                filename = f"{output_dir}/batch{i}.pq"
                Dataset.from_dict({"text": ds.select(idxs)["text"]}).to_parquet(filename)
                idxs = []

    
    if len(idxs):
        filename = f"{output_dir}/batch{i}.pq"
        Dataset.from_dict({"text": ds.select(idxs)["text"]}).to_parquet(filename)

if __name__ == "__main__":
    main()
