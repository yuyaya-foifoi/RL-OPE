CFG_DICT = {
    "TRAIN": {
        "TYPE" : 'CVAE',
        "BATCHSIZE": 4096 * 2,
        "EVAL_BATCHSIZE": 4096 * 2,
        'LOSS_FUNC': 'BCE',
        "EPOCHS": 300,
        "LOG_INTERVAL": 50,
        "WORKER": 2,
        "LR": 1e-3,
    },
    "AUGMENTATION": {
    },
    "LOG": {"SAVE_DIR_NAME": "cvae_0922_500_000"},
}