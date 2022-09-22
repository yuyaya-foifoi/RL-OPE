import os
import shutil

from configs.config import CFG_DICT

SAVE_PATH = "./logs/{model_type}/{dir_name}".format(
    model_type=CFG_DICT["TRAIN"]["TYPE"],
    dir_name=CFG_DICT["LOG"]["SAVE_DIR_NAME"],
)

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)


def save_config():
    shutil.copy("./configs/config.py", SAVE_PATH)


def save_src():
    path = os.path.join(SAVE_PATH, "src")
    if not os.path.exists(path):
        shutil.copytree("./src", path)
