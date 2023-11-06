import torch
import os
import random
import pandas as pd
import numpy as np
import tqdm
import math
from sklearn.metrics import average_precision_score


def set_seed(seed=int):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state


def map3(y_true, y_pred):
    y_true = y_true.to_numpy()
    m = (y_true.reshape((-1, 1)) == y_pred)
    return np.mean(np.where(m.any(axis=1), m.argmax(axis=1) + 1, np.inf) ** (-1))


def compute_metrics(eval_predictions):
    predictions = eval_predictions.predictions
    label_ids = eval_predictions.label_ids

    def average_precision_at_k(actual, predicted, k):
        sorted_answer_indices = np.argsort(-predicted)
        top_answer_indices = sorted_answer_indices[:k]
        actual = int(actual)
        top_answer_indices = [int(i) for i in top_answer_indices]

        if actual in top_answer_indices:
            return [1, 0.5, 0.333333333333][top_answer_indices.index(actual)]
        else:
            return 0

    map_at_3_list = []
    for actual, predicted in zip(label_ids, predictions):
        ap_at_3 = average_precision_at_k(actual, predicted, k=3)
        map_at_3_list.append(ap_at_3)
    # Calculate the Mean Average Precision at 3 (MAP@3) using np.mean
    map_at_3 = np.mean(map_at_3_list)
    # Return a dictionary of metrics (including MAP@3)
    return {"MAP@3": map_at_3}


def getScore(trainer, df, token_ds):
    pred = trainer.predict(token_ds)

    map3_score = 0

    for index in tqdm(range(df.shape[0])):
        columns = df.iloc[index].values
        scores = -pred.predictions[index]
        predict = np.array(list("ABCDE"))[np.argsort(scores)][:3].tolist()
        if columns[6] in predict:
            map3_score += [1, 0.5, 0.333333333333][predict.index(columns[6])]
    map3_score /= df.shape[0]
    print(f'score = {map3_score}')

    return map3_score


def get_score(model, tokenizer, prompt, response):
    inputs = tokenizer(prompt + ' ' + response, return_tensors="pt", max_length=2048, padding='longest',
                       truncation=True).to('cuda')
    model.to('cuda')
    model.eval()
    with torch.autocast('cuda', dtype=torch.float16):
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    logits = outputs.logits

    return logits.item()


def get_top_3_winners(model, tokenizer, prompt, response_options):
    scores = []
    for index, response in enumerate(response_options):
        score = get_score(model, tokenizer, prompt, response)
        scores.append((index, score))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_3_winners = sorted_scores[:3]
    top_3_winners = [t[0] for t in top_3_winners]

    int_to_string = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E'
    }

    top_3_winners = [int_to_string[val] for val in top_3_winners]

    return top_3_winners


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
        user_preds = predictions[u]
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k + 1) * user_results[k]
    return map_at_3 / U





