import pandas as np
import torch
import numpy as np


class CFG:
    num_folds = 5
    selected_folds = [0]  # Folds to train on
    seed = 42

    model_path = 'model/deberta-v3-large-hf-weights'
    # training_args
    warmup_ratio =0.8
    learning_rate = 5e-6
    weight_decay = 0.01
    gradient_accumulation_steps = 6
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    epochs = 3
    output_dir = 'results/'


