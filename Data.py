from Colors import *
from utils import print_error
from DescribeSerie import DescribeSerie
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
            print(f"{BHRED}Fail to read file '{RED}{filepath}{BHRED}'.{RESET}")
            raise DataError()

    def describe(self):

        df = self.df_only_nb
        describe_value = {
            "column_names": [],
            "count": {},
            "mean": {},
            "std": {},
            "min": {},
            "25%": {},
            "50%": {},
            "75%": {},
            "max": {}
        }

        print(BHYELLOW, "[DESCRIBE] : Pandas", RESET, sep="")
        # print(df.describe())
        print(df['Arithmancy'].describe())


        print(BHYELLOW, "[DESCRIBE] : Own", RESET, sep="")
        import random
        for name in df:
            describe_value['column_names'].append(name)
            describe_value['count'][name] = random.uniform(0, 10000)
            describe_value['mean'][name] = 0
            describe_value['std'][name] = 0
            describe_value['min'][name] = 0
            describe_value['25%'][name] = 0
            describe_value['50%'][name] = 0
            describe_value['75%'][name] = 0
            describe_value['max'][name] = 0

        print(df['Arithmancy'].describe())
        describe_serie = DescribeSerie(df['Arithmancy'])
        print(describe_serie.count())
        # print(describe_serie.mean())
        # print(describe_serie.standard_deviation())
        # print(describe_serie.min())
        # print(describe_serie.percentile(percentage=25))
        # print(describe_serie.percentile(percentage=50))
        # print(describe_serie.percentile(percentage=75))
        # print(describe_serie.max())


        # Describe.from_dict(describe_value)
        # print(describe_value)

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
