import pandas as pd
import math

class DescribeSerie:
    def __init__(self, serie: pd.Series):
        self.__serie = pd.Series(filter(lambda x: not math.isnan(x), serie))

    def nan_checker(function):
        def wrapper(self):
            if (len(self.__serie) == 0):
                return float('nan')
            return function(self)
        return wrapper

    # @nan_checker
    def count(self) -> float:
        return float(len(self.__serie))


    @nan_checker
    def mean(self) -> float:
        total_sum = 0
        for num in self.__serie:
            total_sum += num
        return total_sum / len(self.__serie)

    def variance(self) -> float:
        """
    Calculates and returns the variance of a tuple
    """
        mean = self.mean()
        return sum(((xi - mean) ** 2 for xi in self.__serie)) / (len(self.__serie) - 1)

    @nan_checker
    def standard_deviation(self) -> float:
        return self.variance() ** 0.5

    @nan_checker
    def min(self) -> float:
        current_min = self.__serie[0]
        for num in self.__serie:
            if num < current_min:
                current_min = num
        return current_min

    @nan_checker
    def max(self) -> float:
        current_max = self.__serie[0]
        for num in self.__serie:
            if num > current_max:
                current_max = num
        return current_max

    # @nan_checker
    def percentile(self, percentage: int) -> float:
        if (len(self.__serie) == 0):
            return float('nan')
        tab = list(self.__serie)
        tab.sort()
        percentage /= 100
        pos = percentage * (len(tab) - 1)
        rest_pos = pos - math.floor(pos)
        int_pos = math.floor(pos)
        # print(pos)
        # print(int_pos)
        # print(rest_pos)

        if int_pos + 1 == len(tab):
            next_pos = int_pos
        else:
            next_pos = int_pos + 1

        # print(next_pos)
        nb = tab[next_pos] - tab[int_pos]
        nb_to_add = nb * rest_pos
        # print(nb, nb_to_add)
        return tab[int_pos] + nb_to_add

    @staticmethod
    def from_dict(values: dict, file = None):
        values_width = {}
        desc_width = 5
        descriptions: list[str] = list(filter(lambda x: x != "column_names", values.keys()))
        column_names: list[str] = values['column_names']

        for name in column_names:
            max_width = len(name)
            for desc in descriptions:
                number_str = f"{abs(values[desc][name]):.6f}"
                max_width = max(max_width, len(number_str))
            values_width[name] = max_width + 2

        print(" " * desc_width, end="", file=file)
        for name in column_names:
            print(name.rjust(values_width[name]), end="", file=file)
        print(file=file)
        for desc in descriptions:
            print(desc.ljust(desc_width), end="", file=file)
            for name in column_names:
                number = values[desc][name]
                if math.isnan(number):
                    number_str = "NaN"
                elif number == 0:
                    number_str = "0.0"
                else:
                    number_str = f"{values[desc][name]:.6f}"
                print(number_str.rjust(values_width[name]), end="", file=file)
            print(file=file)
