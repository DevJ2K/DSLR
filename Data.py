from Colors import *
from utils import print_error
import numpy as np
import os
import pandas as pd

class DataError(Exception):
    pass

class Data:
    def __init__(self, file: str):
        filepath = os.path.abspath(file)

        self.brut_data = []
        self.fieldnames = []
        print(f"{BHWHITE}Reading file '{BHGREEN}{filepath}{BHWHITE}'...{RESET}")
        try:
            self.df = pd.read_csv(filepath)
            self.df_only_nb = self.df.select_dtypes(include='float64')
        except Exception as e:
            raise DataError()

    def describe(self):
        print(self.df)
        print(self.df_only_nb)



        # print(self.df["Index"])
        # print(self.df["Index"].dtype)
        # print(type(self.df["Index"]))

        # print(pd.DataFrame(data=self.df, dtype=np.float64).dtypes)
        # print(self.df_only_nb)
    # def describe(self):
    #     if (len(self.fieldnames) == 0 or len(self.brut_data) == 0):
    #         print(f"{BHYELLOW}No data to describe...{RESET}")
    #         return
    #     numerical_field = []
    #     print(self.brut_data[0])
    #     for field in self.fieldnames:
    #         try:
    #             float(self.brut_data[0][field])
    #             numerical_field.append(field)
    #         except:
    #             pass
    #     print(self.fieldnames)
    #     print(numerical_field)

    def visualize(self, type_of_visualization: str):
        match type_of_visualization:
            case "histogram":
                return
            case "scatter_plot":
                return
            case "pair_plot":
                return
    


if __name__ == "__main__":
    try:
        data = Data("datasets/dataset_train.csv")
        data.describe()
    except Exception as e:
        print_error(e)
    # Data("datasets/dataset_test.csv")
    # Data("notright")
