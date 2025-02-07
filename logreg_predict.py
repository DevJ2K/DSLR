import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Colors import *
import argparse
from utils import *
from logreg_train import LogisticRegression
import csv

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

    df = pd.read_csv(args.dataset)

    df = df.select_dtypes('float')
    df.dropna(axis=1, how='all', inplace=True)
    clean_dataset(df)
    df.fillna(df.mean(), inplace=True)
    df = min_max_scaling(df)

    log_reg = LogisticRegression(X=df, y=[], epochs=None, learning_rate=None)

    try:
        log_reg.load_weights(args.weights)
    except Exception as e:
        print_error(e)
        exit(1)

    predictions = log_reg.predict()
    with open('houses.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        field = ["Index", "Hogwarts House"]
        writer.writerow(field)
        for i in range(len(predictions)):
            writer.writerow([i, log_reg.predict_to_str(predictions[i])])

    print(f"{GREEN}Prediction file '{BHGREEN}houses.csv{GREEN}' has been successfully created !{RESET}")

    # for i in range(len(prediction)):
    #     print_info(f"Prediction for student {i}: {log_reg.predict_to_str(prediction[i])}")
    # print_info("Prediction: " + str(log_reg.predict()))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_error(e)
