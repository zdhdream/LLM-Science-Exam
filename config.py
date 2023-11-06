import torch

class CFG1:
    seed = 42 # Random seed

    num_folds = 5 # Total folds
    selected_folds = [0] # Folds to train on
    Choices = ['A', 'B', 'C', 'D', 'E']

    epochs = 3 # Training epochs
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 4

    warmup_ratio = 0.3
    learning_rate = 5e-6 # [3e-5, 1e-5, 1.75e-5]
    weight_decay = 0.01

    # augmentation
    use_shuffle_options = False # Augmentation (Shuffle Options)
    is_addition = False
    desired_rows = 35000
    use_eda = True
    eda_rows = 500
    # stop words list
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
                  'ours', 'ourselves', 'you', 'your', 'yours',
                  'yourself', 'yourselves', 'he', 'him', 'his',
                  'himself', 'she', 'her', 'hers', 'herself',
                  'it', 'its', 'itself', 'they', 'them', 'their',
                  'theirs', 'themselves', 'what', 'which', 'who',
                  'whom', 'this', 'that', 'these', 'those', 'am',
                  'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'having', 'do', 'does', 'did',
                  'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                  'because', 'as', 'until', 'while', 'of', 'at',
                  'by', 'for', 'with', 'about', 'against', 'between',
                  'into', 'through', 'during', 'before', 'after',
                  'above', 'below', 'to', 'from', 'up', 'down', 'in',
                  'out', 'on', 'off', 'over', 'under', 'again',
                  'further', 'then', 'once', 'here', 'there', 'when',
                  'where', 'why', 'how', 'all', 'any', 'both', 'each',
                  'few', 'more', 'most', 'other', 'some', 'such', 'no',
                  'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                  'very', 's', 't', 'can', 'will', 'just', 'don',
                  'should', 'now', '']

    # cv
    # (1) merge ext_df and df_train[fold!=@fold] as train_dataset, df_train[fold==fold] as valid_dataset (is_merge=True, external_data=True)
    # (2) merge ext_df and df_train, random choice some data as train dataset, another as valid dataset (is_merge=True, external_data=False)
    # (3) ext_df as train dataset, df_train as valid dataset (is_merge=False, only_valid=True)
    is_merage = False
    only_valid = True
    external_data = False  # External data flag


    # optimizer
    is_freezingEmbedding = True
    SelfDefine1 = False # 效果差
    SelfDefine2 = False
    use_self_optimizer = True
    total_training_steps = 1000
    optim = "Adam"
    lr_scheduler_type='cosine'

    # lora
    use_lora = True
    r = 8
    lora_alpha = 512
    lora_dropout = 0.1



    model_path = 'model/deberta-v3-large-hf-weights'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = "./results"
    best_dir = "./save_checkpoints"

