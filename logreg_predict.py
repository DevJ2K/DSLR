import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Colors import *
import argparse
from utils import *
import pickle
from logreg_train import LogisticRegression

def main():
    parser = argparse.ArgumentParser(description="Use a dataset to predict the output of our logistic regression model.")
    parser.add_argument(
        "-db", "--dataset",
        type=str,
        help="Path to the dataset you want to predict.",
        default='datasets/dataset_test.csv')
    parser.add_argument(
        "-w", "--weights",
        type=str,
        help="Path to the file containing the weights to be used for the prediction.",
        default='weights.pkl')


    args = parser.parse_args()

    print(args.dataset)
    print(args.weights)

    df = pd.read_csv(args.dataset)
    clean_dataset(df)
    classes = { value: index for index, value in enumerate(df['Hogwarts House'].unique()) }
    y = df.replace({'Hogwarts House': classes})['Hogwarts House']
    x_train = df.select_dtypes('float64')
    x_train = min_max_scaling(x_train)

    log_reg = LogisticRegression(X=x_train, y=y, epochs=None, learning_rate=None)

    try:
        log_reg.load_weights(args.weights)
    except Exception as e:
        print_error(e)
        exit(1)
    # print(log_reg.models)
    log_reg.predict()


if __name__ == "__main__":
    main()
