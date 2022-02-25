# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import sys
import os

path = os.getcwd() + "/src"

sys.path.append(path)
from models.train_model import train

#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())

def load_data():
    olivetti = fetch_olivetti_faces()
    return olivetti

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)


    dataset = load_data()
    sss = StratifiedShuffleSplit()
    sss.get_n_splits(dataset.data, dataset.target)

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
    train_valid_idx, test_idx = next(strat_split.split(dataset.data, dataset.target))
    X_train_valid = dataset.data[train_valid_idx]
    y_train_valid = dataset.target[train_valid_idx]
    X_test = dataset.data[test_idx]
    y_test = dataset.target[test_idx]

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
    train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid))
    X_train = X_train_valid[train_idx]
    y_train = y_train_valid[train_idx]
    X_valid = X_train_valid[valid_idx]
    y_valid = y_train_valid[valid_idx]

    train(X_train_valid, X_train, X_valid, X_test, y_train, y_valid, y_train_valid)
    #print(X_train.shape, y_train.shape)
    #print(X_valid.shape, y_valid.shape)
    #print(X_test.shape, y_test.shape)

    #print(dataset)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main("../models/train_model.py", "../models/test_model.py")
