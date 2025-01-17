from Colors import *
from utils import print_error
import csv
import os

class DataError(Exception):
    pass

class Data:
    def __init__(self, file: str):
        filepath = os.path.abspath(file)
        print(f"{BHWHITE}Reading file '{BHGREEN}{filepath}{BHWHITE}'...{RESET}")
        try:
            with open(filepath, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                print(reader.fieldnames)
                # for row in reader:
                #     print(row)
                pass
        except Exception as e:
            print_error(e)
        pass


if __name__ == "__main__":
    Data("datasets/dataset_train.csv")
    # Data("notright")
