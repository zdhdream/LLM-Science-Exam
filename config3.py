import pandas as np
import torch
import numpy as np


class CFG:
    num_folds = 5
    selected_folds = [0]  # Folds to train on
    seed = 42

    model_path = 'model/deberta-v3-large-hf-weights'
    # training_args
    warmup_ratio = 0.1
    learning_rate = 2e-5  # 5e-6 2e-6 2e-5
    weight_decay = 0.01
    decoder_lr = 1e-4
    gradient_accumulation_steps = 8
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 2
    epochs = 2
    output_dir = 'results/'

    # parameters
    is_freezingEmbedding1 = False
    is_freezingEmbedding2 = True
    is_settingParameters1 = True
    is_settingParameters2 = False
    use_SelfParameters = True
    use_Gradual_Unfreezing = False
    FREEZE_EMBEDDINGS = False
    FREEZE_LAYERS = 18  # 18
    NUM_TARIN_SAMPLES = 60347
    MAX_INPUT = 768
    num_training_steps = 7500
    num_warmup_steps = 1500

    # lora
    use_lora = False
    r = 8
    lora_alpha = 512
    lora_dropout = 0.1

    # aug
    use_shuffle_options = False
