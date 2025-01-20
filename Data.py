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
        self.brut_data = []
        self.fieldnames = []
        print(f"{BHWHITE}Reading file '{BHGREEN}{filepath}{BHWHITE}'...{RESET}")
        try:
            with open(filepath, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                self.fieldnames = [field for field in reader.fieldnames]
                self.brut_data = [row for row in reader]
        except Exception as e:
            print_error(e)
        pass

    def describe(self):
        if (len(self.fieldnames) == 0 or len(self.brut_data) == 0):
            print(f"{BHYELLOW}No data to describe...{RESET}")
            return
        numerical_field = []
        print(self.brut_data[0])
        for field in self.fieldnames:
            try:
                float(self.brut_data[0][field])
                numerical_field.append(field)
            except:
                pass
        print(self.fieldnames)
        print(numerical_field)

    def visualize(self, type_of_visualization: str):
        match type_of_visualization:
            case "histogram":
                return
            case "scatter_plot":
                return
            case "pair_plot":
                return
    


if __name__ == "__main__":
    data = Data("datasets/dataset_train.csv")
    data.describe()
    # Data("datasets/dataset_test.csv")
    # Data("notright")
