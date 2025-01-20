from Colors import *
from utils import print_error
import csv
import numpy
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
                for row in reader:
                    print(row)
                    break
                pass
        except Exception as e:
            print_error(e)
        pass

    def describe(self):
        pass

    def visualize(self, type_of_visualization: str):
        match type_of_visualization:
            case "histogram":
                return
            case "scatter_plot":
                return
            case "pair_plot":
                return
    


if __name__ == "__main__":
    Data("datasets/dataset_train.csv")
    Data("datasets/dataset_test.csv")
    # Data("notright")
