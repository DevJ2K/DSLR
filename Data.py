from Colors import *
from DescribeSerie import DescribeSerie
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
            print(f"{BHRED}Failed to read file: '{RED}{filepath}{BHRED}'{RESET}")
            raise DataError()

    def describe(self, outfile: str = None, FULL_FIELD: bool = False):

        df = self.df_only_nb
        describe_values = {
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
        if FULL_FIELD:
            describe_values["variance"] = {}
            describe_values["sum"] = {}

        # print(BHYELLOW, "[DESCRIBE] : Pandas", RESET, sep="")
        # print(df.describe())
        # print(BHYELLOW, "[DESCRIBE] : Own", RESET, sep="")

        for name in df:
            describe_values['column_names'].append(name)
            describeSerie = DescribeSerie(df[name])
            describe_values['count'][name] = describeSerie.count()
            describe_values['mean'][name] = describeSerie.mean()
            describe_values['std'][name] = describeSerie.standard_deviation()
            describe_values['min'][name] = describeSerie.min()
            describe_values['25%'][name] = describeSerie.percentile(25)
            describe_values['50%'][name] = describeSerie.percentile(50)
            describe_values['75%'][name] = describeSerie.percentile(75)
            describe_values['max'][name] = describeSerie.max()
            if FULL_FIELD:
                describe_values['variance'][name] = describeSerie.variance()
                describe_values['sum'][name] = describeSerie.sum(describeSerie.getSeries())

        if (outfile != None):
            with open(outfile, "w") as fd:
                DescribeSerie.from_dict(describe_values, file=fd)
        else:
            DescribeSerie.from_dict(describe_values, file=None)


    def visualize(self, type_of_visualization: str):
        match type_of_visualization:
            case "histogram":
                return
            case "scatter_plot":
                return
            case "pair_plot":
                return



if __name__ == "__main__":
    # try:
        data = Data("datasets/dataset_train.csv")
        # data = Data("datasets/dataset_test.csv")
        data.describe()
        data.describe(FULL_FIELD=True)
    # except Exception as e:
    #     print_error(e)
    # Data("datasets/dataset_test.csv")
    # Data("notright")
