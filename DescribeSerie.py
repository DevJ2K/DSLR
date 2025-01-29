import pandas as pd
import math

class DescribeSerie:
    def __init__(self, serie: pd.Series):
        self.__serie = pd.Series(filter(lambda x: not math.isnan(x), serie))

    def count(self) -> float:
        return float(len(self.__serie))
        

    def mean(self) -> float:
        pass

    def standard_deviation(self) -> float:
        pass

    def min(self) -> float:
        pass

    def max(self) -> float:
        pass

    def percentile(self, pourcentage: int) -> float:
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