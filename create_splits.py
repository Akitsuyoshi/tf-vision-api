import argparse
import glob
import os
import random
import pathlib
import shutil
from os.path import dirname

import numpy as np
from utils import get_module_logger

def split(data_dir, seed=123):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    
    refs: https://github.com/jfilter/split-folders/blob/master/splitfolders/split.py
    """
    # Split data into train/val/test, 64%/16%/20% each
    ratio = {
        'train': 0.64,
        'val': 0.16,
        'test': 0.2
    }
    files = [f for f in pathlib.Path(data_dir).iterdir() if f.is_file() and not f.name.startswith('.')]
    random.seed(seed)
    random.shuffle(files)
    # Get data/waymo directory
    parent_dir = dirname(dirname(data_dir))

    # Split files by ratio
    train_idx = int(ratio['train'] * len(files))
    val_idx = train_idx + int(ratio['val'] * len(files))
    folder_file_map = {
        'train': files[: train_idx],
        'val': files[train_idx: val_idx],
        'test': files[val_idx:],
    }

    for folder, files in folder_file_map.items():
        full_path = os.path.join(parent_dir, folder)
        pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
        logger.info('Started to copy ' + folder + ' folder')
        for f in files:
            shutil.copy2(f, full_path)
        logger.info(folder + ' folder is completed')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)