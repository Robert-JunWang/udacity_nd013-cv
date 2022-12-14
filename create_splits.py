import argparse
import glob
import os
import random

import numpy as np
import shutil

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """

    train_dir = os.path.join(destination, 'train')
    val_dir = os.path.join(destination, 'val') 
    test_dir = os.path.join(destination, 'test') 

    if not os.path.exists(train_dir):
       os.makedirs(train_dir)
    if not os.path.exists(val_dir):
       os.makedirs(val_dir)
    if not os.path.exists(test_dir):
       os.makedirs(test_dir)

    all_files = [filename for filename in glob.glob(f'{source}/*.tfrecord')]
    np.random.shuffle(all_files)

    # Train size: 80%, validation size: 20%
    train_files, val_files = np.split(all_files, [int(len(all_files)*0.8)])

    logger.info('Total record: {}, train files: {}, val files: {}'.format(len(all_files), len(train_files), len(val_files)))

    # split the tf records by symbolically linking the files 
    for src in train_files:
        dst = os.path.join(train_dir, os.path.basename(src))
        os.symlink(src, dst)

    for src in val_files:
        dst = os.path.join(val_dir, os.path.basename(src))
        os.symlink(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)