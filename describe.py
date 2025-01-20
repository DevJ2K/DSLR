from Colors import *
from utils import print_error
from Data import Data, DataError
import csv
import numpy
import os


def main(file: str) -> None:
    data = Data(file=file)
    data.describe()

if __name__ == "__main__":
    main("datasets/dataset_train.csv")
    main("datasets/dataset_test.csv")
