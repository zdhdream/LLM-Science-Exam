class CFG:
    model_name = 'model/deberta-v3-large-hf-weights'
    MAX_LEN = 1500
    seed = 42
    warmup_ratio = 0.1
    learning_rate = 2e-6
    decoder_lr = 1e-4
    weight_decay = 0.01
    gradient_accumulation_steps = 8
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    epoch = 3
    output_dir = "sequence/"

    # parameters
    is_freezingEmbedding1 = True
    is_freezingEmbedding2 = False
    is_settingParameters1 = True
    is_settingParameters2 = False
    use_SelfParameters = True
    use_Gradual_Unfreezing = False
    FREEZE_EMBEDDINGS = False
    FREEZE_LAYERS = 0  # 18

    # Hyperparameter Search
    use_HyperparameterSearch = True

    # lora
    use_lora = False
    r = 8
    lora_alpha = 512
    lora_dropout = 0.1

    # aug
    use_shuffle_options = False
