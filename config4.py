import torch.cuda


class CFG4:
    num_folds = 5
    selected_folds = [0]  # Folds to train on
    seed = 42
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model_path = 'model/deberta-v3-large-hf-weights'
    # training_args
    warmup_ratio = 0.1
    learning_rate = 2e-5
    weight_decay = 0.01
    decoder_lr = 1e-4
    # gradient_accumulation_steps = 6
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 4
    GRAD_ACC = 8
    MAX_GRAD_NORM = None
    EARLY_STOPPING_EPOCH = 3
    epochs = 3
    awp_dir = 'awp_checkpoints/'
    adv_output = 'adv_checkpoints/'
    r_drop = 'r_drop_checkpoints/'
    log = 'log/'

    # parameters
    is_freezingEmbedding = False
    is_settingParameters1 = True
    is_settingParameters2 = False
    use_SelfParameters = True
    use_Gradual_Unfreezing = False
    FREEZE_EMBEDDINGS = False
    FREEZE_LAYERS = 15

    # Hyperparameter Search
    use_HyperparameterSearch = True

    # lora
    use_lora = False
    r = 8
    lora_alpha = 512
    lora_dropout = 0.1

    # awp
    USE_AMP = True
    adv_lr = 0.001
    adv_eps =  0.001

    print_freq = 20

