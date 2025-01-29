import pandas as pd
import math

class DescribeSerie:
    def __init__(self, serie: pd.Series):
        self.__serie = pd.Series(filter(lambda x: not math.isnan(x), serie))

    def no_value(self, function):
        def wrapper():
            print("Here1")
            if (len(self.__serie) == 0):
                return float('nan')
            return function()
        return wrapper()

    @no_value
    def count(self) -> float:
        print("Here")
        return float(len(self.__serie))
        

    def mean(self) -> float:
        if (len(self.__serie) == 0):
            return float('nan')
        pass

    def standard_deviation(self) -> float:
        if (len(self.__serie) == 0):
            return float('nan')
        pass

    def min(self) -> float:
        if (len(self.__serie) == 0):
            return float('nan')
        pass

    def max(self) -> float:
        if (len(self.__serie) == 0):
            return float('nan')
        pass

    def percentile(self, percentage: int) -> float:
        if (len(self.__serie) == 0):
            return float('nan')
        pass

    @staticmethod
    def from_dict(values: dict):
        values_width = {}
        desc_width = 6
        descriptions: list[str] = list(filter(lambda x: x != "column_names", values.keys()))
        column_names: list[str] = values['column_names']

        for name in column_names:
            max_width = len(name)
            for desc in descriptions:
                number_str = f"{values[desc][name]:.6f}"
                max_width = max(max_width, len(number_str))
            values_width[name] = max_width + 2

        print(" " * desc_width, end="")
        for name in column_names:
            print(name.rjust(values_width[name]), end="")
        print()
        for desc in descriptions:
            print(desc.ljust(desc_width), end="")
            for name in column_names:
                number_str = f"{values[desc][name]:.6f}"
                print(number_str.rjust(values_width[name]), end="")
            print()